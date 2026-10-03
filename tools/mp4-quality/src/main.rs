//! Standalone CPU-only quality/stride/timestamp regression using frac's locked codec ref.
use av_codec_core::{AVCodecContext, AVPacket};
use av_util_frame::{AVFrame, av_frame_alloc, av_frame_get_buffer};
use av_util_pixfmt::{AVColorSpace as C, AVPixelFormat as P};
use std::{
    fs::{self, File},
    path::Path,
    time::Instant,
};
type R<T> = Result<T, String>;
const N: usize = 32;
const FPS: (u32, u32) = (24000, 1001);
const COLR: &[u8] = &[b'n', b'c', b'l', b'x', 0, 1, 0, 13, 0, 1, 0];
fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
fn frame(w: usize, h: usize, format: P) -> R<AVFrame> {
    let mut frame = av_frame_alloc();
    frame.width = w as i32;
    frame.height = h as i32;
    frame.format = format as i32;
    av_frame_get_buffer(&mut frame, 0).map_err(err)?;
    Ok(*frame)
}
fn scaler(
    w: usize,
    h: usize,
    src: P,
    dst: P,
    src_full: bool,
    dst_full: bool,
) -> R<Box<av_swscale::SwsContext>> {
    let mut sws = av_swscale::sws_alloc_context();
    sws.set_src(w, h, src);
    sws.set_dst(w, h, dst);
    sws.set_colorspace(C::AVCOL_SPC_BT709);
    sws.set_range(src_full, dst_full);
    av_swscale::sws_init_context(&mut sws).map_err(err)?;
    Ok(sws)
}
fn convert(sws: &av_swscale::SwsContext, src: &AVFrame, format: P) -> R<AVFrame> {
    let mut dst = frame(src.width as usize, src.height as usize, format)?;
    av_swscale::sws_scale_frame(sws, &mut dst, src).map_err(err)?;
    dst.pts = src.pts;
    dst.duration = src.duration;
    Ok(dst)
}
/// Fine grayscale detail, colored multi-pixel structures, moving edges and row markers.
fn pattern(w: usize, h: usize, number: usize) -> Vec<u8> {
    let mut rgb = vec![0; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let sx = x + number * 2;
            let sy = y + number;
            let fine = if (sx / 2 + sy / 2) % 2 == 0 { 28 } else { -28 };
            let gray = (120 + fine + ((sx * 11 + sy * 7) % 31) as i32 - 15).clamp(0, 255) as u8;
            let mut p = [gray; 3];
            if y > h / 3 && y < 2 * h / 3 {
                p = match (sx / 8) % 4 {
                    0 => [210, 55, 35],
                    1 => [35, 180, 65],
                    2 => [45, 65, 215],
                    _ => [200, 180, 55],
                };
            }
            if (x + w - number % w) % w < 2 {
                p = [235; 3];
            }
            if x < 2 {
                p = [30 + (y * 3 % 190) as u8, 40, 200];
            }
            if x >= w - 2 {
                p = [200, 30 + (y * 5 % 190) as u8, 40];
            }
            rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&p);
        }
    }
    rgb
}
fn rgb_frame(w: usize, h: usize, number: usize, rgb: &[u8]) -> R<AVFrame> {
    let mut src = frame(w, h, P::RGB24)?;
    let stride = src.linesize[0] as usize;
    let plane = src.data[0]
        .as_mut()
        .and_then(|b| b.data_mut())
        .ok_or("RGB plane")?;
    // Deliberately nonzero padding proves row addressing, especially width 66.
    plane.fill(0xcd);
    if rgb.len() != checked_rgb_len(w, h)? {
        return Err("RGB source byte count mismatch".into());
    }
    for y in 0..h {
        plane[y * stride..y * stride + w * 3].copy_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
    }
    src.pts = number as i64 * i64::from(FPS.1);
    src.duration = i64::from(FPS.1);
    Ok(src)
}
fn plane(frame: &AVFrame, index: usize, width: usize, height: usize) -> R<Vec<u8>> {
    let buffer = frame.data[index].as_ref().ok_or("missing plane")?.data();
    let stride = usize::try_from(frame.linesize[index]).map_err(err)?;
    let mut result = Vec::with_capacity(width * height);
    for y in 0..height {
        result.extend_from_slice(
            buffer
                .get(y * stride..y * stride + width)
                .ok_or("short plane")?,
        );
    }
    Ok(result)
}
struct Expected {
    rgb: Vec<u8>,
    baseline: Vec<u8>,
    yuv: [Vec<u8>; 3],
}
fn checked_rgb_len(w: usize, h: usize) -> R<usize> {
    if w == 0 || h == 0 || w > 16384 || h > 16384 || !w.is_multiple_of(2) || !h.is_multiple_of(2) {
        return Err("RGB sequence dimensions must be positive, even and at most 16384".into());
    }
    w.checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or("RGB byte count overflow".into())
}
fn load_rgb_sequence(dir: &Path, w: usize, h: usize) -> R<Vec<Vec<u8>>> {
    let len = checked_rgb_len(w, h)?;
    (0..N)
        .map(|n| {
            let path = dir.join(format!("frame.{n:06}.rgb"));
            let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            if bytes.len() != len {
                return Err(format!(
                    "{}: expected {len} RGB24 bytes, found {}",
                    path.display(),
                    bytes.len()
                ));
            }
            Ok(bytes)
        })
        .collect()
}
fn expected(w: usize, h: usize, raw: Option<&[Vec<u8>]>) -> R<Vec<Expected>> {
    let to_yuv = scaler(w, h, P::RGB24, P::YUV420P, true, false)?;
    let to_rgb = scaler(w, h, P::YUV420P, P::RGB24, false, true)?;
    (0..N)
        .map(|n| {
            let bytes = raw.map_or_else(|| pattern(w, h, n), |frames| frames[n].clone());
            let rgb = rgb_frame(w, h, n, &bytes)?;
            let yuv = convert(&to_yuv, &rgb, P::YUV420P)?;
            let baseline = convert(&to_rgb, &yuv, P::RGB24)?;
            Ok(Expected {
                rgb: bytes,
                baseline: plane(&baseline, 0, w * 3, h)?,
                yuv: [
                    plane(&yuv, 0, w, h)?,
                    plane(&yuv, 1, w / 2, h / 2)?,
                    plane(&yuv, 2, w / 2, h / 2)?,
                ],
            })
        })
        .collect()
}
/// Convert encoder timestamps into the muxer's implicit decode-zero media clock.
struct PacketTimeline {
    first_dts: Option<i64>,
    min_pts: i64,
    max_end: i64,
}
impl Default for PacketTimeline {
    fn default() -> Self {
        Self {
            first_dts: None,
            min_pts: i64::MAX,
            max_end: i64::MIN,
        }
    }
}
impl PacketTimeline {
    fn record(&mut self, packet: &AVPacket) -> R<()> {
        self.first_dts.get_or_insert(packet.dts);
        self.min_pts = self.min_pts.min(packet.pts);
        self.max_end = self.max_end.max(
            packet
                .pts
                .checked_add(packet.duration)
                .ok_or("timestamp overflow")?,
        );
        Ok(())
    }
    fn window(&self) -> R<(i64, i64)> {
        let origin = self.first_dts.ok_or("no encoded packets")?;
        let start = self
            .min_pts
            .checked_sub(origin)
            .ok_or("edit start overflow")?;
        let end = self
            .max_end
            .checked_sub(origin)
            .ok_or("edit end overflow")?;
        if start < 0 || end <= start {
            return Err("invalid presentation window".into());
        }
        Ok((start, end))
    }
}
struct Encoded {
    packets: usize,
    bytes: u64,
    ms: f64,
    stride: i32,
}
fn drain(
    ctx: &mut AVCodecContext,
    writer: &mut av_format_movenc::MovWriter<File>,
    header: &mut bool,
    w: usize,
    h: usize,
    timing: &mut String,
    count: &mut usize,
    timeline: &mut PacketTimeline,
) -> R<()> {
    loop {
        let mut packet = AVPacket::new();
        match ctx.avcodec_receive_packet(&mut packet) {
            Ok(()) => {
                if !*header {
                    writer
                        .add_video(
                            av_format_core::Codec::Hevc,
                            &av_codec::hvcc_from_au(packet.data()).map_err(err)?,
                            w as u32,
                            h as u32,
                            FPS.0,
                            &av_format_core::UNITY_MATRIX,
                        )
                        .map_err(err)?;
                    writer.set_video_colr(COLR).map_err(err)?;
                    *header = true;
                }
                let delta = i32::try_from(packet.pts - packet.dts).map_err(err)?;
                if delta < 0 {
                    return Err("negative CTS".into());
                }
                if packet.duration != i64::from(FPS.1) {
                    return Err(format!("duration {}", packet.duration));
                }
                timeline.record(&packet)?;
                timing.push_str(&format!(
                    "{},{},{},{},{},{}\n",
                    *count,
                    packet.pts,
                    packet.dts,
                    packet.duration,
                    packet.data().len(),
                    packet.flags
                ));
                writer
                    .write_sample(
                        0,
                        &av_codec::annexb_to_length_prefixed(packet.data()),
                        FPS.1,
                        delta,
                        packet.flags & av_codec_core::AV_PKT_FLAG_KEY != 0,
                    )
                    .map_err(err)?;
                *count += 1;
            }
            Err(av_util_core::AvError::Posix(11)) | Err(av_util_core::AvError::Eof) => {
                return Ok(());
            }
            Err(e) => return Err(err(e)),
        }
    }
}
fn encode(
    path: &Path,
    w: usize,
    h: usize,
    qp: u8,
    preset: &str,
    trim: bool,
    refs: &[Expected],
) -> R<Encoded> {
    let mut ctx = av_codec_core::avcodec_alloc_context3();
    ctx.width = w as i32;
    ctx.height = h as i32;
    ctx.pix_fmt = P::YUV420P;
    ctx.flags |= av_codec_core::AV_CODEC_FLAG_GLOBAL_HEADER | av_codec_core::AV_CODEC_FLAG_QSCALE;
    ctx.global_quality = i32::from(qp) * av_codec_core::FF_QP2LAMBDA;
    let entry = av_codec::avcodec_find_encoder_by_name("hevc_kvz").ok_or("encoder registry")?;
    ctx.open_with_opts((entry.make)(), &[("preset", preset)])
        .map_err(err)?;
    let sws = scaler(w, h, P::RGB24, P::YUV420P, true, false)?;
    let mut writer =
        av_format_movenc::MovWriter::new(File::create(path).map_err(err)?).map_err(err)?;
    let mut timing = String::from("packet,pts,dts,duration,bytes,flags\n");
    let mut header = false;
    let mut packets = 0;
    let mut timeline = PacketTimeline::default();
    let stride = rgb_frame(w, h, 0, &refs[0].rgb)?.linesize[0];
    if !ctx.extradata.is_empty() {
        writer
            .add_video(
                av_format_core::Codec::Hevc,
                &av_codec::hvcc_from_au(&ctx.extradata).map_err(err)?,
                w as u32,
                h as u32,
                FPS.0,
                &av_format_core::UNITY_MATRIX,
            )
            .map_err(err)?;
        writer.set_video_colr(COLR).map_err(err)?;
        header = true;
    }
    let start = Instant::now();
    for n in 0..N {
        let src = rgb_frame(w, h, n, &refs[n].rgb)?;
        let dst = convert(&sws, &src, P::YUV420P)?;
        ctx.avcodec_send_frame(Some(&dst)).map_err(err)?;
        drain(
            &mut ctx,
            &mut writer,
            &mut header,
            w,
            h,
            &mut timing,
            &mut packets,
            &mut timeline,
        )?;
    }
    ctx.avcodec_send_frame(None).map_err(err)?;
    drain(
        &mut ctx,
        &mut writer,
        &mut header,
        w,
        h,
        &mut timing,
        &mut packets,
        &mut timeline,
    )?;
    // Keep the B-frame CTS values intact; trim their decode preroll from presentation.
    // set_video_edits takes media-clock (start,end), not duration or movie-clock ticks.
    let window = timeline.window()?;
    if window.1 - window.0 != N as i64 * i64::from(FPS.1) {
        return Err("encoded presentation duration".into());
    }
    if trim {
        writer.set_video_edits(&[window]).map_err(err)?;
    }
    writer.finish().map_err(err)?;
    fs::write(path.with_extension("packets.csv"), timing).map_err(err)?;
    if packets != N {
        return Err(format!("packet count {packets} != {N}"));
    }
    Ok(Encoded {
        packets,
        bytes: fs::metadata(path).map_err(err)?.len(),
        ms: start.elapsed().as_secs_f64() * 1000.0,
        stride,
    })
}
#[derive(Default)]
struct ErrorSum {
    sum: f64,
    count: u64,
}
impl ErrorSum {
    fn add(&mut self, a: &[u8], b: &[u8]) -> R<()> {
        if a.len() != b.len() {
            return Err("metric dimensions".into());
        }
        for (&a, &b) in a.iter().zip(b) {
            let d = f64::from(a) - f64::from(b);
            self.sum += d * d;
            self.count += 1;
        }
        Ok(())
    }
    fn psnr(&self) -> f64 {
        if self.sum == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0 * 255.0 / (self.sum / self.count as f64)).log10()
        }
    }
}
#[derive(Default)]
struct Metrics {
    rgb: ErrorSum,
    baseline: ErrorSum,
    yuv: [ErrorSum; 3],
    edge: ErrorSum,
    frames: usize,
    seen: [bool; N],
}
fn collect(
    ctx: &mut AVCodecContext,
    sws: &av_swscale::SwsContext,
    refs: &[Expected],
    m: &mut Metrics,
    w: usize,
    h: usize,
) -> R<()> {
    loop {
        let mut decoded = av_frame_alloc();
        match ctx.avcodec_receive_frame(&mut decoded) {
            Ok(()) => {
                if decoded.width != w as i32
                    || decoded.height != h as i32
                    || decoded.format != P::YUV420P as i32
                {
                    return Err(format!(
                        "decoded geometry {}x{} format{}",
                        decoded.width, decoded.height, decoded.format
                    ));
                }
                if decoded.pts < 0 || decoded.pts % i64::from(FPS.1) != 0 {
                    return Err(format!("decoded PTS {}", decoded.pts));
                }
                let n = decoded.pts as usize / FPS.1 as usize;
                if n >= N || m.seen[n] {
                    return Err(format!("duplicate/out-of-range frame {n}"));
                }
                m.seen[n] = true;
                let reference = &refs[n];
                let rgb = convert(sws, &decoded, P::RGB24)?;
                let rgb = plane(&rgb, 0, w * 3, h)?;
                m.rgb.add(&reference.rgb, &rgb)?;
                m.baseline.add(&reference.rgb, &reference.baseline)?;
                for c in 0..3 {
                    let (cw, ch) = if c == 0 { (w, h) } else { (w / 2, h / 2) };
                    m.yuv[c].add(&reference.yuv[c], &plane(&decoded, c, cw, ch)?)?;
                }
                for y in 0..h {
                    for x in [0, 1, w - 2, w - 1] {
                        let i = (y * w + x) * 3;
                        m.edge.add(&reference.baseline[i..i + 3], &rgb[i..i + 3])?;
                    }
                }
                m.frames += 1;
            }
            Err(av_util_core::AvError::Posix(11)) | Err(av_util_core::AvError::Eof) => {
                return Ok(());
            }
            Err(e) => return Err(err(e)),
        }
    }
}
fn decode(path: &Path, w: usize, h: usize, refs: &[Expected], trim: bool) -> R<Metrics> {
    let mut demux = av_format_mov::Demuxer::open(path).map_err(|e| format!("demux open: {e:?}"))?;
    if demux.width() != w as u32
        || demux.height() != h as u32
        || demux.timescale() != FPS.0
        || demux.sample_count() != N
    {
        return Err("MP4 dimensions/count/timescale mismatch".into());
    }
    if demux.colr_box() != Some(COLR) {
        return Err("MP4 colour tags mismatch".into());
    }
    let mut pts = (0..N)
        .map(|n| demux.display_pts(n).map_err(err))
        .collect::<R<Vec<_>>>()?;
    pts.sort_unstable();
    let first = pts[0];
    if trim
        && (first != 0
            || demux.start_time() != 0
            || demux.presented_duration().map_err(err)? != N as u64 * u64::from(FPS.1)
            || demux.presented_frame_count().map_err(err)? != N
            || demux.edit_list().iter().any(|edit| edit.media_time < 0))
    {
        return Err(format!(
            "presentation origin/duration/frame/edit mismatch: first_pts={first} start={} duration={} frames={} edits={:?}",
            demux.start_time(),
            demux.presented_duration().map_err(err)?,
            demux.presented_frame_count().map_err(err)?,
            demux.edit_list()
        ));
    }
    eprintln!(
        "timing file={} first_pts={} timescale={} coded_ticks={} presented_ticks={} presented_frames={} start_time={} edits={:?}",
        path.display(),
        first,
        demux.timescale(),
        demux.total_duration(),
        demux.presented_duration().map_err(err)?,
        demux.presented_frame_count().map_err(err)?,
        demux.start_time(),
        demux.edit_list()
    );
    for (n, pts) in pts.iter().enumerate() {
        if *pts != first + n as i64 * i64::from(FPS.1) {
            return Err("MP4 presentation gaps".into());
        }
    }
    let mut ctx = av_codec_core::avcodec_alloc_context3();
    let entry = av_codec::avcodec_find_decoder_by_name("hevc").ok_or("HEVC decoder registry")?;
    ctx.extradata = demux.parameter_sets_annexb();
    ctx.open((entry.make)())
        .map_err(|e| format!("decoder open: {e:?}"))?;
    let sws = scaler(w, h, P::YUV420P, P::RGB24, false, true)?;
    let mut m = Metrics::default();
    let mut timing = String::from("sample,pts,dts,duration\n");
    let mut previous_dts = None;
    for n in 0..N {
        let pts = demux.display_pts(n).map_err(err)?;
        let dts = demux.sample_dts(n).map_err(err)?;
        let duration = demux.sample_duration(n).map_err(err)?;
        if duration != FPS.1 || previous_dts.is_some_and(|prev| dts <= prev) {
            return Err("MP4 decode timing".into());
        }
        previous_dts = Some(dts);
        timing.push_str(&format!("{n},{pts},{dts},{duration}\n"));
        let sample = demux.read_sample(n).map_err(err)?;
        // ffmpeg-rs HEVC init retains extradata; parameter sets enter its parser in-band.
        let bytes = if n == 0 {
            let mut bytes = demux.parameter_sets_annexb();
            bytes.extend_from_slice(&sample);
            bytes
        } else {
            sample
        };
        let mut packet = AVPacket::new();
        av_codec_core::ff_get_encode_buffer(&mut packet, bytes.len(), 0)
            .map_err(err)?
            .copy_from_slice(&bytes);
        packet.pts = pts - first;
        packet.dts = dts - first;
        packet.duration = i64::from(duration);
        ctx.avcodec_send_packet(&packet)
            .map_err(|e| format!("send sample {n} pts{pts}: {e:?}"))?;
        collect(&mut ctx, &sws, refs, &mut m, w, h)
            .map_err(|e| format!("receive after sample {n}: {e}"))?;
    }
    ctx.avcodec_send_packet(&AVPacket::new()).map_err(err)?;
    collect(&mut ctx, &sws, refs, &mut m, w, h)?;
    if m.frames != N || !m.seen.iter().all(|seen| *seen) {
        return Err(format!("decoded frame count {}", m.frames));
    }
    fs::write(path.with_extension("demux.csv"), timing).map_err(err)?;
    Ok(m)
}
fn main() -> R<()> {
    let mut args = std::env::args().skip(1);
    let Some(out) = args.next() else {
        eprintln!(
            "Usage: frac-mp4-quality OUTPUT_DIRECTORY [--legacy] [--rgb-sequence DIR WIDTH HEIGHT] (CPU only, 32 frames per encode)"
        );
        return Ok(());
    };
    let mut trim = true;
    let mut sequence = None;
    while let Some(option) = args.next() {
        match option.as_str() {
            "--legacy" => trim = false,
            "--rgb-sequence" if sequence.is_none() => {
                let dir = args
                    .next()
                    .ok_or("--rgb-sequence requires DIR WIDTH HEIGHT")?;
                let w = args
                    .next()
                    .ok_or("missing RGB width")?
                    .parse::<usize>()
                    .map_err(err)?;
                let h = args
                    .next()
                    .ok_or("missing RGB height")?
                    .parse::<usize>()
                    .map_err(err)?;
                let frames = load_rgb_sequence(Path::new(&dir), w, h)?;
                sequence = Some((w, h, frames));
            }
            _ => return Err(format!("unknown or duplicate option {option}")),
        }
    }
    let out = Path::new(&out);
    fs::create_dir_all(out).map_err(err)?;
    let mut csv = String::from(
        "width,height,frames,qp,preset,rgb_psnr,conversion_only_rgb_psnr,y_psnr,u_psnr,v_psnr,edge_vs_conversion_psnr,bytes,encode_ms,rgb_stride,packets\n",
    );
    let dimensions = sequence
        .as_ref()
        .map_or_else(|| vec![(66, 50), (320, 180)], |(w, h, _)| vec![(*w, *h)]);
    for (w, h) in dimensions {
        let raw = sequence.as_ref().map(|(_, _, frames)| frames.as_slice());
        let refs = expected(w, h, raw).map_err(|e| format!("reference {w}x{h}: {e}"))?;
        let mut results = Vec::new();
        for (qp, preset) in [(27, "veryfast"), (18, "medium")] {
            let path = out.join(format!("detail-{w}x{h}-qp{qp}-{preset}.mp4"));
            let encoded = encode(&path, w, h, qp, preset, trim, &refs)
                .map_err(|e| format!("encode {w}x{h} {qp}/{preset}: {e}"))?;
            let metrics = decode(&path, w, h, &refs, trim)
                .map_err(|e| format!("decode {w}x{h} {qp}/{preset}: {e}"))?;
            let row = format!(
                "{w},{h},{},{qp},{preset},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.3},{},{}\n",
                metrics.frames,
                metrics.rgb.psnr(),
                metrics.baseline.psnr(),
                metrics.yuv[0].psnr(),
                metrics.yuv[1].psnr(),
                metrics.yuv[2].psnr(),
                metrics.edge.psnr(),
                encoded.bytes,
                encoded.ms,
                encoded.stride,
                encoded.packets
            );
            print!("{row}");
            csv.push_str(&row);
            results.push(metrics);
            fs::write(out.join("summary.csv"), &csv).map_err(err)?;
        }
        if sequence.is_none() && results[1].yuv[0].psnr() <= results[0].yuv[0].psnr() {
            return Err(format!("{w}x{h}: QP18 medium did not improve luma PSNR"));
        }
        let low = results[0].yuv[1].sum + results[0].yuv[2].sum;
        let high = results[1].yuv[1].sum + results[1].yuv[2].sum;
        if sequence.is_none() && high >= low {
            return Err(format!("{w}x{h}: QP18 medium did not improve chroma error"));
        }
    }
    Ok(())
}
