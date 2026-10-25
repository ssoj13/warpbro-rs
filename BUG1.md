# BUG1 - ffmpeg-rs now pulls OxideAV (oxideav-av1 / -vp9 / -core) git deps; AV1/VP9 CPU decode is being replaced

From: ffmpeg-rs session, 2026-10-24. Affects: every crate here that depends on `ssoj13/ffmpeg-rs`
(`av-codec` (default-features = false), `av-format-mov`, tools/mp4-quality - probably no oxideav, verify).

## What changed in ffmpeg-rs
1. The week audit (`ffmpeg-rs/.agents/plan10/WEEK_AUDIT_2026-10-23.md`, A1/A2) found the AV1 and VP9 CPU decoders
   that landed 2026-10-16..23 were invented, not ports: real AV1 files decoded to garbage, VP9 gave grey frames on any
   error, the VP9 encoder was tuned against that invented decoder. They are being DELETED. Until the replacement lands,
   `av1` / `vp9` without `-hwaccel vulkan` return `AVERROR(ENOSYS)` (C behaviour for `av1`, `av1dec.c:673`), never a
   frame. Vulkan AV1/VP9 decode and the rav1e AV1 encoder are unchanged.
2. The replacement is OxideAV (MIT, pure Rust, spec clean-room): our forks `ssoj13/oxideav-av1`, `ssoj13/oxideav-vp9`,
   `ssoj13/oxideav-core`, branch `ffrs` (av1/vp9) / `master` (core), consumed as SSH git deps exactly like the other
   ssoj13 deps. They are optional behind `av-codec-av1/sw` and `av-codec-vp9/sw`, enabled by `av-codec`'s default
   features `dec-av1` / `dec-vp9`. So you get them if anything in your graph uses `av-codec` with default features
   (`av-decode`, `av-player`, `av-graph`, `av-encoder`, `av-ffargs` all do); `av-codec` with `default-features = false`
   and no `dec-av1`/`dec-vp9` does not.
3. `oxideav-core` is resolved from our fork by the forks themselves - there is NO `[patch]` you need to copy.

## What you need to do
- [ ] After the ffmpeg-rs `main` commit that mentions "oxideav" is pushed: `cargo update -p av-codec` (or the
      ffmpeg-rs crate you depend on) so the lock picks up the new ffmpeg-rs rev and the three oxideav git sources;
      then build/test with `--locked`. Commit the new `Cargo.lock`.
- [ ] Check what you actually get: `cargo tree -i oxideav-core -e normal`. Expect either nothing (no AV1/VP9 software
      decode in your graph) or exactly one `oxideav-core` from `ssh://git@github.com/ssoj13/oxideav-core.git`.
      Two `oxideav-core` entries (one from crates.io) = report it back to ffmpeg-rs, do not patch around it.
- [ ] Do NOT add your own `oxideav-*` deps or a `[patch.crates-io] oxideav-core` - ffmpeg-rs owns that wiring.
- [ ] If you call AV1 or VP9 CPU decoding through ffmpeg-rs (playback of AV1/VP9 files, thumbnails, probing
      that decodes a frame): until the wrapper lands it now fails with ENOSYS instead of returning garbage. Surface the
      error to the user; do not fall back silently. If you need it working sooner, say so to the ffmpeg-rs session.
- [ ] If you offered a VP9 encode/export option backed by ffmpeg-rs `vp9` encoding: that encoder is removed (it was
      invented); remove the option until the OxideAV-backed encoder lands.
- [ ] Reply in this file (append a "Status" line: done / blocked + why).

Plans: `ffmpeg-rs/crates/av-codec-av1/AGENTS.md`, `ffmpeg-rs/crates/av-codec-vp9/AGENTS.md`, `ffmpeg-rs/.agents/oxideav.md`.
Licence: OxideAV is MIT; the notice ships in `ffmpeg-rs/THIRD_PARTY.md` - include it if you redistribute binaries.
