//! CUDA, Vulkan Video, OCIO and OIDN tests share one physical adapter.
//!
//! Naming contract: a test that needs a CUDA device is named `cuda_*`; one that needs another
//! GPU API (a wgpu adapter, Vulkan Video) is named `gpu_*`. Hosted CI has no GPU and runs the
//! ordinary suite with `--skip cuda_ --skip gpu_` (`bootstrap.py` GPU_TEST_PREFIXES); a device
//! test under another name fails there ("Render worker is unavailable", "Vulkan Video
//! unavailable", a wgpu validation error). libtest's `--skip` is a substring match, so the
//! prefixes must not occur anywhere else in a test name either (`wgpu_*` contains `gpu_`):
//! [`device_prefixes_only_start_test_names`] enforces both rules.
use std::sync::{Mutex, MutexGuard};

/// Hold until all GPU objects and background render workers from a test have been dropped.
/// A per-module lock cannot isolate CUDA/render workers from Vulkan Video device creation.
pub fn lock() -> MutexGuard<'static, ()> {
    static ADAPTER: Mutex<()> = Mutex::new(());
    ADAPTER.lock().unwrap_or_else(|error| error.into_inner())
}

/// Blank out comments, string literals and character literals while preserving newlines and code
/// tokens so attribute, signature and brace-balanced body checks never match prose or literals.
fn strip_non_code(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
            continue;
        }
        if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let mut depth = 1usize;
            out.extend_from_slice(b"  ");
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    out.extend_from_slice(b"  ");
                    i += 2;
                } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    out.extend_from_slice(b"  ");
                    i += 2;
                } else {
                    out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
                    i += 1;
                }
            }
            continue;
        }
        let raw_start = match (b, bytes.get(i + 1)) {
            (b'r', _) => Some(i + 1),
            (b'b' | b'c', Some(&b'r')) => Some(i + 2),
            _ => None,
        };
        let boundary = i == 0 || (!bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_');
        if let Some(mut j) = raw_start
            && boundary
        {
            let mut hashes = 0usize;
            while bytes.get(j) == Some(&b'#') {
                hashes += 1;
                j += 1;
            }
            if bytes.get(j) == Some(&b'"') {
                out.resize(out.len() + (j - i + 1), b' ');
                i = j + 1;
                while i < bytes.len() {
                    let suffix = bytes.get(i + 1..i + 1 + hashes);
                    let hashes_match = suffix.is_some_and(|s| s.iter().all(|&c| c == b'#'));
                    if bytes[i] == b'"' && hashes_match {
                        out.resize(out.len() + hashes + 1, b' ');
                        i += hashes + 1;
                        break;
                    }
                    out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
                    i += 1;
                }
                continue;
            }
        }
        if b == b'"' {
            out.push(b' ');
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    out.push(b' ');
                    out.push(if bytes[i + 1] == b'\n' { b'\n' } else { b' ' });
                    i += 2;
                    continue;
                }
                let end = bytes[i] == b'"';
                out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
                i += 1;
                if end {
                    break;
                }
            }
            continue;
        }
        if b == b'\'' {
            if bytes.get(i + 1) == Some(&b'\\') {
                let escaped = bytes.get(i + 2).copied();
                let end = bytes.len().min(i + 32);
                let tail = bytes.get(i + 3..end).unwrap_or_default();
                if escaped.is_some_and(|c| c != b'\n')
                    && let Some(rel) = tail.iter().position(|&c| c == b'\'')
                    && !tail[..rel].contains(&b'\n')
                {
                    out.resize(out.len() + rel + 4, b' ');
                    i += rel + 4;
                    continue;
                }
            } else if let Some(ch) = src.get(i + 1..).and_then(|s| s.chars().next()) {
                let char_bytes = ch.len_utf8();
                if ch != '\'' && ch != '\n' && bytes.get(i + 1 + char_bytes) == Some(&b'\'') {
                    out.resize(out.len() + char_bytes + 2, b' ');
                    i += char_bytes + 2;
                    continue;
                }
            }
        }
        out.push(b);
        i += 1;
    }
    String::from_utf8(out).expect("ASCII masking preserves valid UTF-8")
}

fn strip_leading_attrs(mut line: &str) -> &str {
    line = line.trim_start();
    while let Some(rest) = line.strip_prefix("#[") {
        let mut depth = 1usize;
        let mut end = None;
        for (idx, b) in rest.bytes().enumerate() {
            match b {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(idx + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(after) = end else {
            break;
        };
        line = rest[after..].trim_start();
    }
    line
}

fn parse_fn_name(mut line: &str) -> Option<&str> {
    if let Some(rest) = line.strip_prefix("pub") {
        let rest = rest.trim_start();
        line = if let Some(after_vis) = rest.strip_prefix('(') {
            after_vis.split_once(')')?.1.trim_start()
        } else {
            rest
        };
    }
    for kw in ["async ", "unsafe "] {
        if let Some(rest) = line.strip_prefix(kw) {
            line = rest.trim_start();
        }
    }
    let rest = line.strip_prefix("fn ")?.trim_start();
    let name = rest.split(['(', '<']).next()?.trim();
    (!name.is_empty()).then_some(name)
}

const GPU_MARKERS: [&str; 10] = [
    "new_gpu(",
    "RenderService::spawn(",
    "Gpu::new(",
    "Gpu::with_engine(",
    "WarpBroComputeEngine::new(",
    "WgpuEngine::new(",
    "render_wgpu::warmup(",
    "render_bench::run(",
    "shared_device(",
    "CudaContext::new(",
];

fn find_lock_call(text: &str, in_self: bool) -> Option<usize> {
    if let Some(pos) = text.find("test_gpu::lock(") {
        return Some(pos);
    }
    if !in_self {
        return None;
    }
    text.match_indices("lock(")
        .find(|&(idx, _)| {
            if idx == 0 {
                return true;
            }
            let prev = text.as_bytes()[idx - 1];
            !prev.is_ascii_alphanumeric() && prev != b'_' && prev != b'.'
        })
        .map(|(idx, _)| idx)
}

fn is_valid_lock_stmt(stmt: &str, in_self: bool) -> bool {
    let Some(pos) = find_lock_call(stmt, in_self) else {
        return false;
    };
    if !stmt[pos..].trim_end().ends_with("lock()") {
        return false;
    }
    let before = &stmt[..pos];
    let Some((_, after_let)) = before.rsplit_once("let ") else {
        return false;
    };
    let Some((pat, rhs_prefix)) = after_let.split_once('=') else {
        return false;
    };
    if rhs_prefix.contains(['(', '|', '{', '}']) {
        return false;
    }
    let ident = pat.split(':').next().unwrap_or("").trim();
    let ident = ident.strip_prefix("mut ").map_or(ident, str::trim_start);
    !ident.is_empty() && ident != "_"
}

fn holds_adapter_lock(body: &str, in_self: bool) -> bool {
    let mut depth = 0isize;
    let mut opened = false;
    let mut stmt_start = None;
    for (idx, b) in body.bytes().enumerate() {
        match b {
            b'{' => {
                if !opened {
                    opened = true;
                    depth = 1;
                    stmt_start = Some(idx + 1);
                } else {
                    if depth == 1 {
                        stmt_start = None;
                    }
                    depth += 1;
                }
            }
            b'}' if opened => {
                depth -= 1;
                if depth <= 0 {
                    break;
                }
                if depth == 1 {
                    stmt_start = Some(idx + 1);
                }
            }
            b';' if opened && depth == 1 => {
                if let Some(start) = stmt_start
                    && is_valid_lock_stmt(&body[start..idx], in_self)
                {
                    return true;
                }
                stmt_start = Some(idx + 1);
            }
            _ => {}
        }
    }
    false
}

/// Enforces both halves of the GPU test contract over every `#[test]` in `src/`:
/// 1. Only tests starting with `gpu_` / `cuda_` may contain those substrings (`--skip` is a
///    substring match, so a GPU-free `wgpu_*` test would be silently skipped).
/// 2. Every test touching `test_gpu::lock` or a GPU/render entry point starts with `gpu_` /
///    `cuda_`, and every `gpu_*` / `cuda_*` test binds `test_gpu::lock()` to a named guard to
///    serialize shared-adapter access for the whole test scope.
#[test]
fn device_prefixes_only_start_test_names() {
    const PREFIXES: [&str; 2] = ["gpu_", "cuda_"];
    let mut bad = Vec::new();
    let mut dirs = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).expect("read src directory") {
            let path = entry.expect("read src entry").path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            if path.extension().is_none_or(|ext| ext != "rs") {
                continue;
            }
            let raw = std::fs::read_to_string(&path).expect("read Rust source file");
            let code = strip_non_code(&raw);
            let lines: Vec<&str> = code.lines().collect();
            let in_self = path.file_name().is_some_and(|f| f == "test_gpu.rs");
            let in_bench = path.file_name().is_some_and(|f| f == "render_bench.rs");
            let mut idx = 0;
            let mut test_next = false;
            while idx < lines.len() {
                let trimmed = lines[idx].trim();
                if trimmed.contains("#[test]") || trimmed.contains("::test]") {
                    test_next = true;
                }
                let after_attrs = strip_leading_attrs(trimmed);
                if test_next && let Some(name) = parse_fn_name(after_attrs) {
                    test_next = false;
                    let start_line = idx + 1;
                    let mut depth = 0isize;
                    let mut opened = false;
                    let mut body = String::new();
                    while idx < lines.len() {
                        body.push_str(lines[idx]);
                        body.push('\n');
                        for ch in lines[idx].chars() {
                            if ch == '{' {
                                depth += 1;
                                opened = true;
                            } else if ch == '}' {
                                depth -= 1;
                            }
                        }
                        idx += 1;
                        if opened && depth <= 0 {
                            break;
                        }
                    }
                    let loc = format!("{}:{start_line}: `{name}`", path.display());
                    let device = PREFIXES.iter().any(|p| name.starts_with(p));
                    if !device && PREFIXES.iter().any(|p| name.contains(p)) {
                        bad.push(format!("{loc} contains a device prefix after index 0"));
                    }
                    let mentions_lock = find_lock_call(&body, in_self).is_some();
                    let holds_lock = holds_adapter_lock(&body, in_self);
                    let marker_gpu = GPU_MARKERS.iter().any(|m| body.contains(m));
                    let bench_gpu = in_bench
                        && (body.contains("super::run(") || body.contains("super::run_families("));
                    let opens_gpu = marker_gpu || bench_gpu;
                    if !device && (mentions_lock || opens_gpu) {
                        bad.push(format!("{loc} uses GPU without `cuda_`/`gpu_` prefix"));
                    }
                    if device && !holds_lock {
                        bad.push(format!("{loc} requires a named `test_gpu::lock()`"));
                    }
                    continue;
                }
                idx += 1;
            }
        }
    }
    assert!(
        bad.is_empty(),
        "GPU test naming / adapter-lock contract violation:\n{}",
        bad.join("\n")
    );
}

#[test]
fn lexer_and_lock_ratchet_cover_escapes_and_wildcard_guards() {
    let sample = concat!(
        "fn check(c: char) -> bool { matches!(c, '\\''|'\"'|'\\u{0010_FFFF}') }\n",
        "fn utf8(c: char) -> bool { matches!(c, '·'|'\"') && matches!(c, '·'|'{') }\n",
        "const S: &str = cr#\"#[test] fn wgpu_fake() {}\"#;\n",
    );
    let masked = strip_non_code(sample);
    assert!(!masked.contains("wgpu_fake"));
    assert!(masked.contains("fn utf8(c: char)"));
    assert_eq!(masked.matches('{').count(), masked.matches('}').count());
    let ok = |s| holds_adapter_lock(s, false);
    let ok_self = |s| holds_adapter_lock(s, true);
    assert!(ok("fn gpu_t() { let _g = crate::test_gpu::lock(); }"));
    assert!(ok_self("fn gpu_t() { let mut g = lock(); }"));
    assert!(!ok("fn gpu_t() { let _ = crate::test_gpu::lock(); }"));
    assert!(!ok("fn gpu_t() { let _: () = crate::test_gpu::lock(); }"));
    assert!(!ok("fn gpu_t() { crate::test_gpu::lock(); }"));
    assert!(!ok("fn gpu_t() { { let g = crate::test_gpu::lock(); } }"));
    assert!(!ok("fn gpu_t() { if c { let g = test_gpu::lock(); } }"));
    assert!(!ok("fn gpu_t() { let f = || test_gpu::lock(); }"));
}
