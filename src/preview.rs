//! Playa transport and worker-owned, byte-budgeted native presentation cache.
use crate::{
    render_service::{Command, Frame, RenderEvent, RenderService},
    scene::Scene,
    world::NodeId,
};
use playa_engine::{
    colour::ColorSpaceId,
    core::{CacheManager, GlobalFrameCache},
    entities::{
        CacheStrategy,
        frame::{Frame as PlayaFrame, PixelBuffer, PixelFormat, Premult},
    },
};
use playa_player::clock::Clock;
use std::sync::Arc;

/// All transport entry points use this policy; draft frames are completed at their
/// own one-sample target and never masquerade as final-quality cache entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewMode {
    Play,
    CacheThenPlay,
    DraftCacheThenPlay,
}
impl PreviewMode {
    pub fn cache_all(self) -> bool {
        self != Self::Play
    }
    pub fn samples(self, final_samples: u32) -> u32 {
        if self == Self::DraftCacheThenPlay {
            1
        } else {
            final_samples
        }
    }
}

/// Frozen animation and presentation settings for one preview cache generation.
#[derive(Clone)]
pub struct PreviewRequest {
    pub generation: u64,
    pub scene: Arc<Scene>,
    pub first: u32,
    pub last: u32,
    pub fps: f32,
    pub width: usize,
    pub height: usize,
    pub spp: u32,
    pub seed: u32,
    pub output_hdr: bool,
    pub white_nits: f32,
    pub cache_fraction: f64,
    pub reserve_gb: f64,
}
impl PreviewRequest {
    /// Restarting transport is a scheduling change, not a pixel-content change.
    pub(crate) fn compatible_content(&self, other: &Self) -> bool {
        let scene_matches = Arc::ptr_eq(&self.scene, &other.scene)
            || match (&self.scene.document, &other.scene.document) {
                (Some(left), Some(right)) => left == right,
                (None, None) => self.scene == other.scene,
                _ => false,
            };
        scene_matches
            && self.cache_bounds() == other.cache_bounds()
            && self.width == other.width
            && self.height == other.height
            && self.spp == other.spp
            && self.seed == other.seed
            && self.output_hdr == other.output_hdr
            && self.white_nits == other.white_nits
            && self.cache_fraction == other.cache_fraction
            && self.reserve_gb == other.reserve_gb
    }
    /// Transport range does not define pixel identity. Cache covers the authoring
    /// work area so selection-only preview changes reuse the same frames.
    pub fn cache_bounds(&self) -> (u32, u32) {
        self.scene
            .document
            .as_ref()
            .map_or((self.first, self.last), |doc| {
                (doc.first.min(self.first), doc.last.max(self.last))
            })
    }
    pub fn frame_count(&self) -> Result<u32, String> {
        self.last
            .checked_sub(self.first)
            .and_then(|n| n.checked_add(1))
            .filter(|n| *n <= 100001)
            .ok_or("Preview range must contain 1..=100001 frames".into())
    }
    pub fn validate(&self) -> Result<(), String> {
        self.frame_count()?;
        let (first, last) = self.cache_bounds();
        if last
            .checked_sub(first)
            .and_then(|n| n.checked_add(1))
            .is_none_or(|n| n > 100001)
        {
            return Err("Preview cache range must contain 1..=100001 frames".into());
        }
        if !self.fps.is_finite()
            || self.fps <= 0.0
            || !self.white_nits.is_finite()
            || self.white_nits <= 0.0
        {
            return Err("Preview FPS and reference white must be positive and finite".into());
        }
        if !self.cache_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.cache_fraction)
            || self.cache_fraction == 0.0
            || !self.reserve_gb.is_finite()
            || self.reserve_gb < 0.0
        {
            return Err("Invalid preview memory budget".into());
        }
        crate::render_service::validate_size(self.width, self.height, self.spp)
    }
}

/// UI adapter. Playa owns the playback clock; cache access and rendering stay on the worker.
#[derive(Default)]
pub struct PreviewController {
    request: Option<PreviewRequest>,
    clock: Option<Clock>,
    pending: Option<Command>,
    begin: Option<Command>,
    recycled: [Option<Arc<Frame>>; 3],
    caching: bool,
    completed: u32,
    bytes: usize,
    error: Option<String>,
    awaiting_frame: bool,
    pub resident: Arc<[u32]>,
}
impl PreviewController {
    /// Adopt a completed ordinary viewport frame without changing transport or
    /// copying any pixel buffer on the UI thread.
    pub fn cache_viewport(
        &mut self,
        mut request: PreviewRequest,
        number: u32,
        frame: Arc<Frame>,
    ) -> Result<(), String> {
        request.validate()?;
        if self
            .request
            .as_ref()
            .is_none_or(|old| !old.compatible_content(&request))
        {
            let mut clock = Clock::new(request.fps, request.frame_count()? as i32);
            clock.set_loop(true);
            clock.set_position((number.saturating_sub(request.first)) as i32);
            self.clock = Some(clock);
            self.request = Some(request.clone());
            self.resident = Arc::default();
            self.caching = false;
            self.awaiting_frame = false;
            self.error = None;
            self.begin = None;
        }
        if let Some(active) = &self.request {
            request.generation = active.generation;
        }
        self.pending = Some(Command::CacheViewport {
            request,
            number,
            frame,
        });
        Ok(())
    }
    pub fn cached_request(&self) -> Option<PreviewRequest> {
        self.request.clone()
    }
    pub fn cached_spp(&self) -> Option<u32> {
        self.request.as_ref().map(|r| r.spp)
    }
    pub fn displaying_preview(&self) -> bool {
        self.caching || self.playing() || self.awaiting_frame
    }
    pub fn start(&mut self, request: PreviewRequest, cache_then_play: bool) -> Result<(), String> {
        request.validate()?;
        if self
            .request
            .as_ref()
            .is_none_or(|old| !old.compatible_content(&request))
        {
            self.resident = Arc::default();
        }
        let mut clock = Clock::new(request.fps, request.frame_count()? as i32);
        clock.set_loop(true);
        if !cache_then_play {
            clock.play();
        }
        self.begin = Some(Command::BeginPreview {
            request: request.clone(),
            cache_all: cache_then_play,
        });
        self.pending = None;
        self.request = Some(request);
        self.clock = Some(clock);
        self.caching = cache_then_play;
        self.completed = 0;
        self.bytes = 0;
        self.error = None;
        self.awaiting_frame = !cache_then_play;
        Ok(())
    }
    /// Sends replaceable work without waiting and returns a newly advanced absolute frame.
    pub fn update(&mut self, dt: f32, service: &RenderService) -> Option<u32> {
        if service.preview_stopped() {
            self.cancel();
            self.pending = None;
            if self.error.as_deref() != Some("CUDA worker is unavailable") {
                self.error = Some("CUDA worker is unavailable".into());
            }
            return None;
        }
        for slot in &mut self.recycled {
            if let Some(frame) = slot.take() {
                if let Err(Command::RecyclePreviewFrame { frame }) =
                    service.try_preview_command(Command::RecyclePreviewFrame { frame })
                {
                    *slot = Some(frame);
                }
            }
        }
        if let Some(command) = self.begin.take() {
            if let Err(command) = service.try_preview_command(command) {
                self.begin = Some(command);
                return None;
            }
        }
        if let Some(command) = self.pending.take() {
            if let Err(command) = service.try_preview_command(command) {
                self.pending = Some(command);
                return None;
            }
        }
        let request = self.request.as_ref()?;
        let clock = self.clock.as_mut()?;
        if self.caching || self.awaiting_frame || !dt.is_finite() || dt < 0.0 || !clock.tick(dt) {
            return None;
        }
        let number = request.first + clock.position() as u32;
        self.pending = Some(Command::SeekPreview {
            generation: request.generation,
            number,
        });
        self.awaiting_frame = true;
        Some(number)
    }
    pub fn handle(&mut self, event: &RenderEvent) -> Option<Arc<Frame>> {
        let generation = self.request.as_ref()?.generation;
        match event {
            RenderEvent::PreviewFrame {
                generation: id,
                number,
                frame,
            } if *id == generation && Some(*number) == self.position() => {
                self.awaiting_frame = false;
                Some(frame.clone())
            }
            RenderEvent::PreviewProgress {
                generation: id,
                completed,
                total,
                bytes,
                resident,
            } if *id == generation => {
                self.completed = (*completed).min(*total);
                self.bytes = *bytes;
                self.resident = resident.clone();
                None
            }
            RenderEvent::PreviewReady { generation: id } if *id == generation => {
                self.caching = false;
                if let Some(clock) = &mut self.clock {
                    clock.set_position(0);
                    clock.play();
                }
                self.seek(self.request.as_ref()?.first);
                None
            }
            RenderEvent::PreviewFailed {
                generation: id,
                error,
            } if *id == generation => {
                self.error = Some(error.clone());
                self.cancel();
                None
            }
            _ => None,
        }
    }
    pub fn pause(&mut self) {
        if let Some(clock) = &mut self.clock {
            clock.pause();
        }
        if self.caching {
            self.cancel();
        }
    }
    pub fn set_loop(&mut self, enabled: bool) {
        if let Some(clock) = &mut self.clock {
            clock.set_loop(enabled);
        }
    }
    pub fn resume(&mut self) {
        if let Some(clock) = &mut self.clock {
            clock.play();
        }
    }
    pub fn seek(&mut self, number: u32) {
        if let (Some(request), Some(clock)) = (&self.request, &mut self.clock) {
            let number = number.clamp(request.first, request.last);
            clock.set_position((number - request.first) as i32);
            self.pending = Some(Command::SeekPreview {
                generation: request.generation,
                number,
            });
            self.awaiting_frame = true;
        }
    }
    pub fn cancel(&mut self) {
        self.begin = None;
        if let Some(request) = &self.request {
            self.pending = Some(Command::CancelPreview {
                generation: request.generation,
            });
        }
        if let Some(clock) = &mut self.clock {
            clock.pause();
        }
        self.caching = false;
        self.request = None;
        self.resident = Arc::default();
        self.clock = None;
        self.awaiting_frame = false;
    }
    /// Return a retired/rejected presentation to the worker after upload staging releases it.
    pub fn recycle(&mut self, frame: Arc<Frame>) {
        if self
            .recycled
            .iter()
            .flatten()
            .any(|old| Arc::ptr_eq(old, &frame))
        {
            return;
        }
        if let Some(slot) = self.recycled.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(frame);
        }
    }
    pub fn position(&self) -> Option<u32> {
        Some(self.request.as_ref()?.first + self.clock.as_ref()?.position() as u32)
    }
    pub fn running(&self) -> bool {
        self.caching || self.playing()
    }
    pub fn playing(&self) -> bool {
        self.clock.as_ref().is_some_and(Clock::playing)
    }
    pub fn caching(&self) -> bool {
        self.caching
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    pub fn progress(&self) -> (u32, u32, usize) {
        (
            self.completed,
            self.request
                .as_ref()
                .and_then(|r| r.frame_count().ok())
                .unwrap_or(0),
            self.bytes,
        )
    }
}

#[derive(Clone)]
struct FrameMetadata {
    light_kind: crate::color::DisplayLight,
    samples: u32,
    converged: bool,
    last_ms: f32,
    last_spp: u32,
    unresolved: f32,
    denoised_samples: u32,
    denoise_ms: f32,
    denoise_error: Option<String>,
}
/// All native pixel ownership belongs to Playa's cache. Metadata cannot keep evicted pixels alive.
pub(crate) struct PreviewCache {
    pub request: PreviewRequest,
    pub manager: Arc<CacheManager>,
    cache: GlobalFrameCache,
    metadata: Vec<Option<FrameMetadata>>,
    codes_id: NodeId,
    linear_id: NodeId,
}
impl PreviewCache {
    pub fn new(request: PreviewRequest, cache_all: bool) -> Result<Self, String> {
        request.validate()?;
        let (cache_first, cache_last) = request.cache_bounds();
        let count = (cache_last - cache_first + 1) as usize;
        let manager = Arc::new(CacheManager::new(
            request.cache_fraction,
            request.reserve_gb,
        ));
        let required = request
            .width
            .checked_mul(request.height)
            .and_then(|n| n.checked_mul(20))
            .and_then(|n| n.checked_mul(request.frame_count().ok()? as usize))
            .ok_or("Preview cache size overflow")?;
        if cache_all && required > manager.mem().1 {
            return Err(format!(
                "Preview range needs {} MiB; cache budget is {} MiB. Reduce preview resolution or increase the cache budget.",
                required / 1048576,
                manager.mem().1 / 1048576
            ));
        }
        let cache =
            GlobalFrameCache::new(count.saturating_mul(2), manager.clone(), CacheStrategy::All);
        Ok(Self {
            request,
            manager,
            cache,
            metadata: vec![None; count],
            codes_id: NodeId::new(),
            linear_id: NodeId::new(),
        })
    }
    pub fn restart(&mut self, request: PreviewRequest, cache_all: bool) -> Result<(), String> {
        request.validate()?;
        let required = request
            .width
            .checked_mul(request.height)
            .and_then(|n| n.checked_mul(20))
            .and_then(|n| n.checked_mul(request.frame_count().ok()? as usize))
            .ok_or("Preview cache size overflow")?;
        if cache_all && required > self.manager.mem().1 {
            return Err("Preview range exceeds cache budget; reduce preview resolution or increase the cache budget".into());
        }
        self.manager.increment_generation();
        self.request = request;
        Ok(())
    }
    pub fn index(&self, number: u32) -> Option<i32> {
        let (first, last) = self.request.cache_bounds();
        (number >= first && number <= last).then(|| (number - first) as i32)
    }
    pub fn bytes(&self) -> usize {
        self.manager.mem().0
    }
    pub fn count(&self) -> u32 {
        (self.request.first..=self.request.last)
            .filter(|n| self.contains(*n))
            .count() as u32
    }
    pub fn resident(&self) -> Arc<[u32]> {
        let (first, last) = self.request.cache_bounds();
        (first..=last)
            .filter(|n| self.contains(*n))
            .collect::<Vec<_>>()
            .into()
    }
    pub fn store(&mut self, number: u32, frame: &Frame, epoch: u64) -> Result<(), String> {
        let index = self
            .index(number)
            .ok_or("Preview frame is outside its range")?;
        if let Some(error) = &frame.colour_error {
            return Err(error.clone());
        }
        if frame.width != self.request.width
            || frame.height != self.request.height
            || frame.pixels.len() != frame.width * frame.height
            || frame.light.len() != frame.width * frame.height
            || !frame.complete(self.request.spp)
        {
            return Err("Viewport frame does not match preview quality or dimensions".into());
        }
        let meta = FrameMetadata {
            light_kind: frame.light_kind,
            samples: frame.samples,
            converged: frame.converged,
            last_ms: frame.last_ms,
            last_spp: frame.last_spp,
            unresolved: frame.unresolved,
            denoised_samples: frame.denoised_samples,
            denoise_ms: frame.denoise_ms,
            denoise_error: frame.denoise_error.clone(),
        };
        let codes = frame
            .pixels
            .iter()
            .copied()
            .flat_map(u32::to_le_bytes)
            .collect();
        let linear = frame.light.iter().flatten().copied().collect();
        let codes = PlayaFrame::from_cpu_buffer(
            PixelBuffer::U8(codes),
            PixelFormat::Rgba8,
            frame.width,
            frame.height,
            ColorSpaceId::Srgb,
            Premult::Opaque,
        )
        .map_err(|e| e.to_string())?;
        let linear = PlayaFrame::from_cpu_buffer(
            PixelBuffer::F32(linear),
            PixelFormat::RgbaF32,
            frame.width,
            frame.height,
            ColorSpaceId::LinearSrgb,
            Premult::Opaque,
        )
        .map_err(|e| e.to_string())?;
        self.cache.insert(self.codes_id.into(), index, codes, epoch);
        self.cache
            .insert(self.linear_id.into(), index, linear, epoch);
        self.metadata[index as usize] = Some(meta);
        Ok(())
    }
    pub fn contains(&self, number: u32) -> bool {
        self.index(number).is_some_and(|index| {
            self.metadata[index as usize].is_some()
                && self.cache.get_status(self.codes_id.into(), index).is_some()
                && self
                    .cache
                    .get_status(self.linear_id.into(), index)
                    .is_some()
        })
    }
    pub fn get_into(&self, number: u32, frame: &mut Frame) -> Option<()> {
        let index = self.index(number)?;
        let meta = self.metadata[index as usize].as_ref()?;
        let codes = self.cache.get(self.codes_id.into(), index)?.cpu_raster()?.0;
        let linear = self
            .cache
            .get(self.linear_id.into(), index)?
            .cpu_raster()?
            .0;
        let (PixelBuffer::U8(codes), PixelBuffer::F32(linear)) = (codes.as_ref(), linear.as_ref())
        else {
            return None;
        };
        frame.pixels.clear();
        frame.pixels.extend(
            codes
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]])),
        );
        frame.light.clear();
        frame
            .light
            .extend(linear.chunks_exact(4).map(|v| [v[0], v[1], v[2], v[3]]));
        let sdr = Arc::get_mut(&mut frame.sdr_bytes)?;
        sdr.clear();
        sdr.extend_from_slice(codes);
        let hdr = Arc::get_mut(&mut frame.hdr_bytes)?;
        hdr.clear();
        if self.request.output_hdr {
            let gain = if meta.light_kind.hdr() {
                100.0 / self.request.white_nits
            } else {
                1.0
            };
            for pixel in &frame.light {
                for value in [
                    crate::color::oetf(pixel[0] * gain),
                    crate::color::oetf(pixel[1] * gain),
                    crate::color::oetf(pixel[2] * gain),
                    1.0,
                ] {
                    hdr.extend_from_slice(&value.to_ne_bytes());
                }
            }
        }
        frame.generation = self.request.generation;
        frame.preview = true;
        frame.width = self.request.width;
        frame.height = self.request.height;
        frame.radiance.clear();
        frame.light_kind = meta.light_kind;
        frame.colour_error = None;
        frame.denoised_samples = meta.denoised_samples;
        frame.denoise_ms = meta.denoise_ms;
        frame.denoise_error.clone_from(&meta.denoise_error);
        frame.samples = meta.samples;
        frame.converged = meta.converged;
        frame.last_ms = meta.last_ms;
        frame.last_spp = meta.last_spp;
        frame.unresolved = meta.unresolved;
        Some(())
    }
}

/// A discarded preview cache (content change, draft/final switch) hands its rasters to
/// Playa's reaper thread. Dropping `GlobalFrameCache` itself would free them inline on the
/// render worker, which stalls it (Playa measured ~400 ms for 4 GB of frames).
impl Drop for PreviewCache {
    fn drop(&mut self) {
        self.cache.clear_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> PreviewRequest {
        PreviewRequest {
            generation: 19,
            scene: Arc::new(Scene::preset(crate::params::FAMILY_BULB)),
            first: 10,
            last: 12,
            fps: 24.0,
            width: 1,
            height: 1,
            spp: 8,
            seed: 3,
            output_hdr: true,
            white_nits: 203.0,
            cache_fraction: 0.01,
            reserve_gb: 0.0,
        }
    }
    fn frame() -> Frame {
        Frame {
            generation: 19,
            preview: true,
            width: 1,
            height: 1,
            pixels: vec![0xff123456],
            light: vec![[-2.0, 0.125, 40.0, 1.0]],
            radiance: vec![],
            light_kind: crate::color::DisplayLight::Absolute { peak_nits: 1000.0 },
            colour_error: None,
            denoised_samples: 8,
            denoise_ms: 0.5,
            denoise_error: None,
            samples: 8,
            converged: false,
            last_ms: 3.0,
            last_spp: 2,
            unresolved: 0.0,
            sdr_bytes: Arc::new(vec![]),
            hdr_bytes: Arc::new(vec![]),
        }
    }
    #[test]
    fn incomplete_frames_are_rejected_but_one_sample_draft_is_complete() {
        let mut final_cache = PreviewCache::new(request(), false).unwrap();
        let mut partial = frame();
        partial.samples = 7;
        let epoch = final_cache.manager.current_epoch();
        assert!(final_cache.store(10, &partial, epoch).is_err());
        assert!(final_cache.resident().is_empty());
        let mut draft_request = request();
        draft_request.spp = PreviewMode::DraftCacheThenPlay.samples(8);
        assert!(!draft_request.compatible_content(&request()));
        let mut draft = PreviewCache::new(draft_request, true).unwrap();
        partial.samples = 1;
        draft
            .store(10, &partial, draft.manager.current_epoch())
            .unwrap();
        assert_eq!(draft.resident().as_ref(), &[10]);
        assert!(final_cache.resident().is_empty());
        final_cache.store(10, &frame(), epoch).unwrap();
        assert_eq!(final_cache.resident().as_ref(), &[10]);
        assert!(final_cache.store(10, &partial, epoch).is_err());
        let mut output = frame();
        final_cache.get_into(10, &mut output).unwrap();
        assert_eq!(output.samples, 8);
    }

    #[test]
    fn selected_transport_range_reuses_work_area_frame_identity() {
        let mut full = request();
        Arc::make_mut(&mut full.scene).document = Some(Box::new(
            crate::world::WorldDocument::from_scene(&full.scene),
        ));
        let mut cache = PreviewCache::new(full.clone(), false).unwrap();
        cache
            .store(10, &frame(), cache.manager.current_epoch())
            .unwrap();
        let index = cache.index(10);
        let mut selection = full.clone();
        selection.first = 11;
        selection.last = 12;
        selection.generation += 1;
        assert!(full.compatible_content(&selection));
        cache.restart(selection, false).unwrap();
        assert_eq!(cache.index(10), index);
        assert_eq!(cache.count(), 0, "progress counts only the selected range");
        assert_eq!(
            cache.resident().as_ref(),
            &[10],
            "coverage includes the whole work area"
        );
    }

    #[test]
    fn transport_restart_retains_native_cache_and_content_changes_do_not_match() {
        let original = request();
        let mut cache = PreviewCache::new(original.clone(), true).unwrap();
        let epoch = cache.manager.current_epoch();
        cache.store(10, &frame(), epoch).unwrap();
        let mut restart = original.clone();
        restart.generation += 1;
        assert!(original.compatible_content(&restart));
        cache.restart(restart, true).unwrap();
        assert_eq!(cache.manager.current_epoch(), epoch);
        assert_eq!(cache.count(), 1);
        let mut out = frame();
        cache.get_into(10, &mut out).unwrap();
        assert_eq!(out.generation, 20);
        assert_eq!(out.light, [[-2.0, 0.125, 40.0, 1.0]]);
        let mut changed = original.clone();
        changed.spp += 1;
        assert!(!original.compatible_content(&changed));
        let mut changed = original.clone();
        Arc::make_mut(&mut changed.scene).camera.yaw_degrees += 1.0;
        assert!(!original.compatible_content(&changed));
        let mut world = original.clone();
        Arc::make_mut(&mut world.scene).document = Some(Box::new(
            crate::world::WorldDocument::from_scene(&original.scene),
        ));
        let mut restart = world.clone();
        restart.generation += 1;
        Arc::make_mut(&mut restart.scene).camera.yaw_degrees += 1.0;
        assert!(
            world.compatible_content(&restart),
            "frozen authoring document owns evaluation, not transient head camera"
        );
        Arc::make_mut(&mut restart.scene)
            .document
            .as_mut()
            .unwrap()
            .fps += 1.0;
        assert!(!world.compatible_content(&restart));
    }
    #[test]
    fn start_then_seek_retains_begin_and_cancel_clears_transport() {
        let mut controller = PreviewController::default();
        controller.start(request(), true).unwrap();
        controller.seek(12);
        assert!(matches!(
            controller.begin,
            Some(Command::BeginPreview { .. })
        ));
        assert!(matches!(
            controller.pending,
            Some(Command::SeekPreview { number: 12, .. })
        ));
        assert_eq!(controller.position(), Some(12));
        controller.cancel();
        assert!(controller.begin.is_none());
        assert!(matches!(
            controller.pending,
            Some(Command::CancelPreview { generation: 19 })
        ));
        assert_eq!(controller.position(), None);
        controller.resume();
        assert!(!controller.running());
    }
    #[test]
    fn playback_rejects_same_generation_stale_seek_and_old_generation_events() {
        let mut controller = PreviewController::default();
        controller.start(request(), false).unwrap();
        controller.seek(12);
        let frame = Arc::new(frame());
        assert!(
            controller
                .handle(&RenderEvent::PreviewFrame {
                    generation: 19,
                    number: 10,
                    frame: frame.clone()
                })
                .is_none()
        );
        assert!(controller.awaiting_frame);
        assert!(
            controller
                .handle(&RenderEvent::PreviewFrame {
                    generation: 18,
                    number: 12,
                    frame: frame.clone()
                })
                .is_none()
        );
        assert!(
            controller
                .handle(&RenderEvent::PreviewFrame {
                    generation: 19,
                    number: 12,
                    frame
                })
                .is_some()
        );
        assert!(!controller.awaiting_frame);
        controller.handle(&RenderEvent::PreviewFailed {
            generation: 19,
            error: "test failure".into(),
        });
        assert!(!controller.running());
        assert_eq!(controller.position(), None);
        assert_eq!(controller.error(), Some("test failure"));
    }
    #[test]
    fn native_hdr_cache_preserves_values_and_reuses_presentation_allocations() {
        let mut cache = PreviewCache::new(request(), true).unwrap();
        let epoch = cache.manager.current_epoch();
        cache.store(10, &frame(), epoch).unwrap();
        assert_eq!(cache.count(), 1);
        assert_eq!(cache.bytes(), 20);
        let mut out = frame();
        cache.get_into(10, &mut out).unwrap();
        assert_eq!(out.light, [[-2.0, 0.125, 40.0, 1.0]]);
        assert_eq!(out.pixels, [0xff123456]);
        assert_eq!(
            out.hdr_bytes.as_ref(),
            &crate::render_service::hdr_canvas_bytes(&out.light, 100.0 / 203.0)
        );
        let pointers = (
            out.pixels.as_ptr(),
            out.light.as_ptr(),
            out.sdr_bytes.as_ptr(),
            out.hdr_bytes.as_ptr(),
        );
        for _ in 0..32 {
            cache.get_into(10, &mut out).unwrap();
        }
        assert_eq!(
            pointers,
            (
                out.pixels.as_ptr(),
                out.light.as_ptr(),
                out.sdr_bytes.as_ptr(),
                out.hdr_bytes.as_ptr()
            )
        );
        cache.manager.increment_generation();
        assert!(cache.contains(10), "seek is not a content invalidation");
    }
    #[test]
    fn range_at_u32_max_and_preflight_reject_invalid_or_unbudgeted_work() {
        let mut req = request();
        req.first = u32::MAX - 2;
        req.last = u32::MAX;
        assert_eq!(req.frame_count().unwrap(), 3);
        let cache = PreviewCache::new(req, false).unwrap();
        assert_eq!(cache.index(u32::MAX), Some(2));
        let mut req = request();
        req.first = 20;
        assert!(req.validate().is_err());
        let mut req = request();
        req.width = 16384;
        req.height = 4096;
        req.cache_fraction = 1e-12;
        assert!(
            PreviewCache::new(req, true).is_err(),
            "preflight must reject before allocating pixels"
        );
    }
}
