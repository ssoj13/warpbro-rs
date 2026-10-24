//! Loopback REST transport over a bounded command bus. Only the host applies commands.
//! HTTP replies acknowledge the host's result, never merely a successful enqueue.
use serde::Deserialize;
use serde_json::{Value, json};
use std::io::Read;
use std::sync::{
    Arc, Mutex, PoisonError,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};
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

/// A deferred request. Expired requests cannot mutate the host after a client timed out.
pub struct Call {
    pub event_type: String,
    pub payload: Value,
    deadline: Instant,
    cancelled: Arc<AtomicBool>,
    reply: mpsc::SyncSender<Result<Reply, String>>,
}
impl Call {
    pub fn expired(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || Instant::now() >= self.deadline
    }
    pub fn respond(self, result: Result<Reply, String>) {
        let _ = self.reply.send(result);
    }
}

pub struct Server {
    pub calls: mpsc::Receiver<Call>,
    port: u16,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Server {
    pub fn start(port: u16, wake: Arc<dyn Fn() + Send + Sync>) -> Result<Self, String> {
        let listener = tiny_http::Server::http(("127.0.0.1", port))
            .map_err(|e| format!("REST API 127.0.0.1:{port}: {e}"))?;
        let port = listener
            .server_addr()
            .to_ip()
            .ok_or("REST listener has no IP address")?
            .port();
        let (send, calls) = mpsc::sync_channel(64);
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        // A fixed pool: slow clients never create an unbounded number of threads.
        let (requests, receive) = mpsc::sync_channel::<Request>(8);
        let receive = Arc::new(Mutex::new(receive));
        for index in 0..4 {
            let receive = receive.clone();
            let send = send.clone();
            let wake = wake.clone();
            let stopping = stop.clone();
            // Workers may be reading an incomplete client body. They own no listener
            // and must never prevent the host from shutting down.
            thread::Builder::new()
                .name(format!("rest-{index}"))
                .spawn(move || {
                    loop {
                        let request = receive
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .recv();
                        let Ok(request) = request else { break };
                        handle(request, &send, &wake, &stopping);
                    }
                })
                .map_err(|e| e.to_string())?;
        }
        let thread = thread::Builder::new()
            .name("rest-listener".into())
            .spawn(move || {
                while !stopping.load(Ordering::Acquire) {
                    match listener.recv_timeout(Duration::from_millis(50)) {
                        Ok(Some(request)) => {
                            if let Err(mpsc::TrySendError::Full(request)) =
                                requests.try_send(request)
                            {
                                answer(request, 503, Err("REST request queue is full".into()));
                            }
                        }
                        Ok(None) => {}
                        Err(_) => break,
                    }
                }
                drop(requests);
                drop(listener);
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            calls,
            port,
            stop,
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
        // Return terminal errors before waiting for the listener to release its port.
        for call in self.calls.try_iter() {
            call.respond(Err("Application is shutting down".into()));
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Event {
    event_type: String,
    #[serde(default)]
    payload: Value,
}

fn handle(
    mut request: Request,
    bus: &mpsc::SyncSender<Call>,
    wake: &Arc<dyn Fn() + Send + Sync>,
    stop: &AtomicBool,
) {
    // Browser pages may send loopback POSTs even when CORS hides the response.
    if request
        .headers()
        .iter()
        .any(|header| header.field.equiv("Origin"))
    {
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
            Ok(Reply::Json(json!({"pid":std::process::id(), "protocol":1}))),
        );
        return;
    }
    let event = if request.method() == &Method::Get && path.starts_with("/api/") {
        Ok(Event {
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
            Err(error) => Err(error.to_string()),
            Ok(_) if body.len() as u64 > LIMIT => Err("REST body exceeds 1 MiB".into()),
            Ok(_) if path == "/api/event" => serde_json::from_str(&body).map_err(|e| e.to_string()),
            Ok(_) => {
                let payload = if body.trim().is_empty() {
                    Ok(json!({}))
                } else {
                    serde_json::from_str(&body)
                };
                payload
                    .map(|payload| Event {
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
    let event = match event {
        Ok(event) => event,
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
    let cancelled = Arc::new(AtomicBool::new(false));
    let call = Call {
        event_type: event.event_type,
        payload: event.payload,
        deadline: Instant::now() + TIMEOUT,
        cancelled: cancelled.clone(),
        reply,
    };
    match bus.try_send(call) {
        Ok(()) => {}
        Err(mpsc::TrySendError::Full(_)) => {
            answer(request, 503, Err("Command bus is full".into()));
            return;
        }
        Err(mpsc::TrySendError::Disconnected(_)) => {
            answer(
                request,
                503,
                Err("Application command bus has stopped".into()),
            );
            return;
        }
    }
    wake();
    match receive.recv_timeout(TIMEOUT) {
        Ok(result) => answer(request, if result.is_ok() { 200 } else { 400 }, result),
        Err(error) => {
            cancelled.store(true, Ordering::Release);
            answer(
                request,
                504,
                Err(format!(
                    "Application did not acknowledge the command: {error}"
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
    use std::io::{Read, Write};
    #[test]
    fn replies_follow_application_acknowledgement_and_stop_releases_port() {
        let server = Server::start(0, Arc::new(|| {})).unwrap();
        let port = server.port();
        let client = thread::spawn(move || {
            let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            socket.write_all(b"POST /api/viewport/orbit HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
            let mut response = String::new();
            socket.read_to_string(&mut response).unwrap();
            response
        });
        let call = server.calls.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(call.event_type, "viewport.orbit");
        assert!(!call.expired());
        call.respond(Err("Bad orbit".into()));
        let response = client.join().unwrap();
        assert!(response.contains("400 Bad Request"));
        assert!(response.contains("Bad orbit"));
        drop(server);
        let _next = Server::start(port, Arc::new(|| {})).unwrap();
    }
    #[test]
    fn incomplete_body_does_not_block_shutdown_or_port_reuse() {
        let server = Server::start(0, Arc::new(|| {})).unwrap();
        let port = server.port();
        let mut socket = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        socket
            .write_all(b"POST /api/event HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{")
            .unwrap();
        // Give the worker time to begin the body read while keeping the socket open.
        thread::sleep(Duration::from_millis(100));
        let (done, stopped) = mpsc::channel();
        let shutdown = thread::spawn(move || {
            drop(server);
            done.send(()).unwrap();
        });
        let result = stopped.recv_timeout(Duration::from_secs(2));
        if result.is_err() {
            // Release the blocked reader even when testing a regression.
            drop(socket);
            shutdown.join().unwrap();
            panic!("REST shutdown waited for an incomplete request body");
        }
        shutdown.join().unwrap();
        let next = Server::start(port, Arc::new(|| {})).unwrap();
        drop(next);
        drop(socket);
    }

    #[test]
    fn browser_origin_requests_never_reach_the_bus() {
        let server = Server::start(0, Arc::new(|| {})).unwrap();
        let mut socket = std::net::TcpStream::connect(("127.0.0.1", server.port())).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        socket.write_all(b"POST /api/app/exit HTTP/1.1\r\nHost: localhost\r\nOrigin: https://example.com\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).unwrap();
        assert!(response.contains("403 Forbidden"));
        assert!(server.calls.try_recv().is_err());
    }
}
