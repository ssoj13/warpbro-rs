//! Loopback REST transport. UI and HTTP publish the same typed events on Playa's EventBus.
//! Reply channels carry acknowledgements only; there is no parallel application-command queue.
use playa_events::EventEmitter;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::Read,
    sync::{
        Arc, Mutex, PoisonError, Weak,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
use tiny_http::{Header, Method, Request, Response};
const LIMIT: u64 = 1 << 20;
const TIMEOUT: Duration = Duration::from_secs(10);

pub enum Reply {
    Json(Value),
    Bytes {
        content_type: &'static str,
        data: Vec<u8>,
    },
}
struct Receipt {
    deadline: Instant,
    cancelled: AtomicBool,
    reply: Mutex<Option<mpsc::SyncSender<Result<Reply, String>>>>,
}
impl Receipt {
    fn complete(&self, result: Result<Reply, String>) {
        let sender = self
            .reply
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(sender) = sender {
            let _ = sender.try_send(result);
        }
    }
}
/// A typed action or query. Widgets and HTTP use this same EventBus event.
/// Clones share one completion; an event can never acknowledge twice or block the UI.
#[derive(Clone)]
pub struct Event<C> {
    pub command: C,
    receipt: Option<Arc<Receipt>>,
}
impl<C> Event<C> {
    pub fn ui(command: C) -> Self {
        Self {
            command,
            receipt: None,
        }
    }
    pub fn expired(&self) -> bool {
        self.receipt
            .as_ref()
            .is_some_and(|r| r.cancelled.load(Ordering::Acquire) || Instant::now() >= r.deadline)
    }
    pub fn respond(&self, result: Result<Reply, String>) {
        if let Some(receipt) = &self.receipt {
            receipt.complete(result);
        }
    }
}
pub type Decoder<C> = Arc<dyn Fn(&str, Value) -> Result<C, String> + Send + Sync>;
pub struct Server {
    port: u16,
    stop: Arc<AtomicBool>,
    pending: Arc<Mutex<Vec<Weak<Receipt>>>>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Server {
    pub fn start<C: Clone + Send + Sync + 'static>(
        port: u16,
        events: EventEmitter,
        decode: Decoder<C>,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<Self, String> {
        let listener = tiny_http::Server::http(("127.0.0.1", port))
            .map_err(|e| format!("REST API 127.0.0.1:{port}: {e}"))?;
        let port = listener
            .server_addr()
            .to_ip()
            .ok_or("REST listener has no IP address")?
            .port();
        let stop = Arc::new(AtomicBool::new(false));
        let pending = Arc::new(Mutex::new(Vec::<Weak<Receipt>>::new()));
        // These are HTTP sockets waiting for a parser, not application commands.
        let (requests, receive) = mpsc::sync_channel::<Request>(8);
        let receive = Arc::new(Mutex::new(receive));
        for index in 0..4 {
            let receive = receive.clone();
            let events = events.clone();
            let decode = decode.clone();
            let wake = wake.clone();
            let stop = stop.clone();
            let pending = pending.clone();
            thread::Builder::new()
                .name(format!("rest-{index}"))
                .spawn(move || {
                    loop {
                        let request = receive
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .recv();
                        let Ok(request) = request else {
                            break;
                        };
                        handle(request, &events, &decode, &wake, &stop, &pending);
                    }
                })
                .map_err(|e| e.to_string())?;
        }
        let stopping = stop.clone();
        let thread = thread::Builder::new()
            .name("rest-listener".into())
            .spawn(move || {
                while !stopping.load(Ordering::Acquire) {
                    match listener.recv_timeout(Duration::from_millis(50)) {
                        Ok(Some(request)) => match requests.try_send(request) {
                            Ok(()) => {}
                            Err(
                                mpsc::TrySendError::Full(request)
                                | mpsc::TrySendError::Disconnected(request),
                            ) => answer(
                                request,
                                503,
                                Err("REST parser queue is full or stopped".into()),
                            ),
                        },
                        Ok(None) => {}
                        Err(_) => break,
                    }
                }
                drop(requests);
                drop(listener);
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            port,
            stop,
            pending,
            thread: Some(thread),
        })
    }
    pub fn port(&self) -> u16 {
        self.port
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let pending =
            std::mem::take(&mut *self.pending.lock().unwrap_or_else(PoisonError::into_inner));
        for receipt in pending.into_iter().filter_map(|w| w.upgrade()) {
            receipt.cancelled.store(true, Ordering::Release);
            receipt.complete(Err("Application is shutting down".into()));
        }
        // Body readers hold no listening socket and cannot hold the host open.
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCommand {
    event_type: String,
    #[serde(default)]
    payload: Value,
}
fn handle<C: Clone + Send + Sync + 'static>(
    mut request: Request,
    events: &EventEmitter,
    decode: &Decoder<C>,
    wake: &Arc<dyn Fn() + Send + Sync>,
    stop: &AtomicBool,
    pending: &Mutex<Vec<Weak<Receipt>>>,
) {
    if request.headers().iter().any(|h| h.field.equiv("Origin")) {
        answer(
            request,
            403,
            Err("Browser-origin requests are not allowed".into()),
        );
        return;
    }
    let path = request.url().split('?').next().unwrap_or("").to_owned();
    if request.method() == &Method::Get && path == "/api/health" {
        answer(
            request,
            200,
            Ok(Reply::Json(json!({"pid":std::process::id(),"protocol":1}))),
        );
        return;
    }
    let wire = if request.method() == &Method::Get && path.starts_with("/api/") {
        Ok(WireCommand {
            event_type: format!("get.{}", path[5..].replace('/', ".")),
            payload: Value::Null,
        })
    } else if request.method() == &Method::Post && path.starts_with("/api/") {
        let mut body = String::new();
        match request
            .as_reader()
            .take(LIMIT + 1)
            .read_to_string(&mut body)
        {
            Err(e) => Err(e.to_string()),
            Ok(_) if body.len() as u64 > LIMIT => Err("REST body exceeds 1 MiB".into()),
            Ok(_) if path == "/api/event" => serde_json::from_str(&body).map_err(|e| e.to_string()),
            Ok(_) => {
                let payload = if body.trim().is_empty() {
                    Ok(json!({}))
                } else {
                    serde_json::from_str(&body)
                };
                payload
                    .map(|payload| WireCommand {
                        event_type: path[5..].replace('/', "."),
                        payload,
                    })
                    .map_err(|e| e.to_string())
            }
        }
    } else {
        answer(request, 404, Err("Unknown REST route".into()));
        return;
    };
    let command = match wire.and_then(|w| decode(&w.event_type, w.payload)) {
        Ok(command) => command,
        Err(error) => {
            answer(request, 400, Err(error));
            return;
        }
    };
    if stop.load(Ordering::Acquire) {
        answer(request, 503, Err("Application is shutting down".into()));
        return;
    }
    let (reply, receive) = mpsc::sync_channel(1);
    let receipt = Arc::new(Receipt {
        deadline: Instant::now() + TIMEOUT,
        cancelled: AtomicBool::new(false),
        reply: Mutex::new(Some(reply)),
    });
    {
        let mut pending = pending.lock().unwrap_or_else(PoisonError::into_inner);
        if stop.load(Ordering::Acquire) {
            drop(pending);
            answer(request, 503, Err("Application is shutting down".into()));
            return;
        }
        pending.retain(|r| r.strong_count() > 0);
        pending.push(Arc::downgrade(&receipt));
    }
    let event = Event {
        command,
        receipt: Some(receipt.clone()),
    };
    if events.try_emit(event).is_err() {
        answer(
            request,
            503,
            Err("EventBus is full; action was not published".into()),
        );
        return;
    }
    wake();
    match receive.recv_timeout(TIMEOUT) {
        Ok(result) => answer(request, if result.is_ok() { 200 } else { 400 }, result),
        Err(error) => {
            receipt.cancelled.store(true, Ordering::Release);
            answer(
                request,
                504,
                Err(format!(
                    "Application did not acknowledge the event: {error}"
                )),
            );
        }
    }
}
fn answer(request: Request, status: u16, result: Result<Reply, String>) {
    let (kind, data) = match result {
        Ok(Reply::Bytes { content_type, data }) => (content_type, data),
        Ok(Reply::Json(value)) => ("application/json", value.to_string().into_bytes()),
        Err(error) => (
            "application/json",
            json!({"success":false,"error":error})
                .to_string()
                .into_bytes(),
        ),
    };
    let response = Response::from_data(data)
        .with_status_code(status)
        .with_header(Header::from_bytes("Content-Type", kind).expect("static HTTP header"));
    let _ = request.respond(response);
}

#[cfg(test)]
mod tests {
    use super::*;
    use playa_events::{EventBus, downcast_event};
    use std::io::{Read, Write};
    fn server(bus: &EventBus) -> Server {
        Server::start(
            0,
            bus.emitter(),
            Arc::new(|name, data| Ok(json!({"name":name,"data":data}))),
            Arc::new(|| {}),
        )
        .unwrap()
    }
    fn send(port: u16, headers: &str, body: &str) -> String {
        let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        write!(socket,"POST /api/viewport/orbit HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n{}",body.len(),headers,body).unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).unwrap();
        response
    }
    #[test]
    fn http_publishes_typed_event_and_waits_for_application_acknowledgement() {
        let bus = EventBus::new();
        let server = server(&bus);
        let port = server.port();
        let client = thread::spawn(move || send(port, "", "{}"));
        let start = Instant::now();
        while bus.queue_len() == 0 {
            assert!(start.elapsed() < Duration::from_secs(2));
            thread::yield_now();
        }
        let events = bus.poll();
        assert_eq!(events.len(), 1);
        let event = downcast_event::<Event<Value>>(&events[0]).unwrap();
        assert_eq!(event.command["name"], "viewport.orbit");
        assert!(!event.expired());
        event.respond(Err("Bad orbit".into()));
        event.clone().respond(Ok(Reply::Json(json!({}))));
        let response = client.join().unwrap();
        assert!(response.contains("400 Bad Request"));
        assert!(response.contains("Bad orbit"));
        let port = server.port();
        drop(server);
        let _reopened =
            Server::start(port, bus.emitter(), Arc::new(|_, v| Ok(v)), Arc::new(|| {})).unwrap();
    }
    #[test]
    fn full_event_bus_refuses_http_action_without_eviction() {
        let bus = EventBus::new();
        let server = server(&bus);
        let mut accepted = 0;
        while bus.try_emit(Event::ui(json!(accepted))).is_ok() {
            accepted += 1;
        }
        let response = send(server.port(), "", "{}");
        assert!(response.contains("503 Service Unavailable"));
        assert_eq!(bus.poll().len(), accepted);
    }
    #[test]
    fn browser_origin_requests_never_reach_the_bus() {
        let bus = EventBus::new();
        let server = server(&bus);
        let response = send(server.port(), "Origin: https://example.com\r\n", "{}");
        assert!(response.contains("403 Forbidden"));
        assert_eq!(bus.queue_len(), 0);
    }
    #[test]
    fn incomplete_body_does_not_block_shutdown_or_port_reuse() {
        let bus = EventBus::new();
        let server = server(&bus);
        let port = server.port();
        let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        socket
            .write_all(
                b"POST /api/event HTTP/1.1\r\nHost: localhost\r\nContent-Length: 99\r\n\r\n{",
            )
            .unwrap();
        thread::sleep(Duration::from_millis(60));
        let start = Instant::now();
        drop(server);
        assert!(start.elapsed() < Duration::from_secs(1));
        let _reopened =
            Server::start(port, bus.emitter(), Arc::new(|_, v| Ok(v)), Arc::new(|| {})).unwrap();
    }
}
