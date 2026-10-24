//! Live render-worker diagnostics. Available while the renderer is busy, via REST and the UI.
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex, PoisonError};
use std::time::Instant;

#[derive(Default)]
struct State {
    compiling: Option<(String, Instant)>,
    pipelines: VecDeque<(String, f64)>,
    cache_hits: u64,
    cache_misses: u64,
    cache_writes: u64,
}
static WAKE: Mutex<Option<crate::render_service::Wake>> = Mutex::new(None);
pub fn set_wake(wake: crate::render_service::Wake) {
    *WAKE.lock().unwrap_or_else(PoisonError::into_inner) = Some(wake);
}
pub fn wake() {
    let wake = WAKE.lock().unwrap_or_else(PoisonError::into_inner).clone();
    if let Some(wake) = wake {
        wake.wake();
    }
}
static STATE: LazyLock<Mutex<State>> = LazyLock::new(Default::default);

pub struct PipelineTimer {
    name: String,
    start: Instant,
}
impl PipelineTimer {
    pub fn start(name: &str, family: Option<u32>) -> Self {
        let name = format!("{name} (family {family:?})");
        let start = Instant::now();
        STATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .compiling = Some((name.clone(), start));
        log::info!("WGPU compile started: {name}");
        wake();
        Self { name, start }
    }
}
impl Drop for PipelineTimer {
    fn drop(&mut self) {
        let seconds = self.start.elapsed().as_secs_f64();
        log::info!("WGPU compile finished: {} in {seconds:.3}s", self.name);
        let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
        state.compiling = None;
        state.pipelines.push_back((self.name.clone(), seconds));
        while state.pipelines.len() > 64 {
            state.pipelines.pop_front();
        }
    }
}
pub fn compiling() -> Option<(String, f64)> {
    STATE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .compiling
        .as_ref()
        .map(|(name, start)| (name.clone(), start.elapsed().as_secs_f64()))
}
pub fn thumbnail(hit: bool, write: bool) {
    let mut state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    if write {
        state.cache_writes += 1;
    } else if hit {
        state.cache_hits += 1;
    } else {
        state.cache_misses += 1;
    }
}
pub fn snapshot() -> Value {
    let state = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    json!({
        "thumbnail_cache":{"hits":state.cache_hits,"misses":state.cache_misses,"writes":state.cache_writes},
        "compiling":state.compiling.as_ref().map(|(name,start)| json!({"entry":name,"seconds":start.elapsed().as_secs_f64()})),
        "pipelines":state.pipelines.iter().map(|(name,seconds)| json!({"entry":name,"seconds":seconds})).collect::<Vec<_>>()
    })
}
