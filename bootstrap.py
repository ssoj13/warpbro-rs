#!/usr/bin/env python3
"""
bootstrap.py - Unified local build/run/check script for WarpBro.

Cross-platform-ish (Linux / WSL2), Python 3, stdlib only. Adapted from the gitnexus-rs
bootstrap.

WarpBro builds through cuda-oxide: its GPU kernels (fractal-rs `fractal-kernels`) are Rust compiled to PTX by the
rustc_codegen_cuda backend, which `cargo oxide` enables. Builds warm the CUDA driver's JIT
cache and open the WGPU device with its shader module and post pipelines before reporting
success, so the next workspace launch does not wait for a cold compile.
Plain `cargo build` cannot link them (build.rs stops it with an explanation). This script checks the toolchain first, then drives
`cargo oxide`.

Toolchain (see README.md):
    NVIDIA driver (on WSL2: the Windows driver, never a Linux one inside WSL)
    CUDA Toolkit 13.x (nvcc, libNVVM, nvJitLink)      CUDA_HOME, default /usr/local/cuda
    LLVM llc 21+ and clang + libclang headers         CUDA_OXIDE_LLC, default newest llc-2x
    rustup toolchain stable (rust-toolchain.toml) + rust-src rustc-dev llvm-tools
    cargo-oxide from our fork at CUDA_OXIDE_REV       python bootstrap.py d --fix

Commands:
    d(octor)      Check the toolchain (--fix installs the rustup / cargo-oxide parts)
    b(uild)       cargo oxide build  -> target/release/WarpBro
    r(un)         Build, then run the browser (args after `--` go to WarpBro)
    g(allery)     Render every preset:  g [DIR [W H SPP]]   (default gallery 1920 1080 256)
    bench         Time every preset:    bench [W H SPP]     (default 960 540 32)
    docs          Rebuild the README images in docs/ (needs uv for Pillow)
    i(nstall)     Copy the release binary to ~/.local/bin/WarpBro
    c(heck)       cargo fmt --check + cargo clippy
    k(ernels)     Import verified native assembly receipts; --assemble explicitly invokes ptxas
    ci            Hosted CI without a GPU: toolchain, check, tests (--skip cuda_ gpu_), build, dist/*.zip
    cl(ean)       cargo clean (+ stray *.ll / *.ptx dumps in the repo root)
    h(elp)        Print help

Examples:
    python bootstrap.py d --fix
    python bootstrap.py b
    python bootstrap.py r
    python bootstrap.py r -- --bench 1920 1080 64
    python bootstrap.py g out 3840 2160 1024
    python bootstrap.py i
    python bootstrap.py ci
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).parent.resolve()
IS_WINDOWS = platform.system() == "Windows"
IS_WSL = "microsoft" in platform.release().lower()

BIN_NAME = "WarpBro.exe" if IS_WINDOWS else "WarpBro"
RELEASE_BIN = ROOT_DIR / "target" / "release" / BIN_NAME
INSTALL_DIR = Path.home() / ".local" / "bin"

CUDA_OXIDE_GIT = "ssh://git@github.com/ssoj13/cuda-rust-windows.git"
CUDA_OXIDE_REV = "9370f8c91eba499a880a5769ecd11f39fa65ffbb"
RUST_COMPONENTS = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "rustfmt", "llvm-tools"]
# Release builds (`--cubins`) also embed cubins for these GPUs (RTX 20/30/40/50), assembled
# from the sm_75 PTX by cuda-oxide into one fat binary ahead of that PTX: those GPUs skip the
# first-launch JIT, any other GPU or an older driver still JIT-compiles the PTX.
CUBIN_ARCHS = "sm_75,sm_86,sm_89,sm_120"
MIN_LLVM = 21
MIN_CUDA_MAJOR = 13
DIST_DIR = ROOT_DIR / "dist"
# Test-name prefixes of device tests (src/test_gpu.rs): CUDA, and other GPU APIs (wgpu, Vulkan Video).
GPU_TEST_PREFIXES = ("cuda_", "gpu_")


class C:
    RST = "\033[0m"
    RED = "\033[91m"
    GRN = "\033[92m"
    YLW = "\033[93m"
    CYN = "\033[96m"
    WHT = "\033[97m"

    @classmethod
    def init(cls) -> None:
        if IS_WINDOWS:
            os.system("")


def fmt_time(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60000:
        return f"{ms / 1000:.1f}s"
    mins = int(ms // 60000)
    secs = (ms % 60000) / 1000
    return f"{mins}m{secs:.0f}s"


def header(text: str) -> None:
    line = "=" * 60
    print(f"\n{C.CYN}{line}\n{text}\n{line}{C.RST}")


def step(text: str) -> None:
    print(f"  {C.WHT}{text}{C.RST}")


def ok(text: str) -> None:
    print(f"  {C.GRN}[OK] {text}{C.RST}")


def warn(text: str) -> None:
    print(f"  {C.YLW}[WARN] {text}{C.RST}")


def err(text: str) -> None:
    print(f"  {C.RED}[ERR] {text}{C.RST}")


def run(args: list[str], cwd: Path | None = None, capture: bool = False, env: dict | None = None) -> tuple[int, str, float]:
    start = time.perf_counter()
    try:
        result = subprocess.run(args, cwd=cwd or ROOT_DIR, capture_output=capture, text=True, encoding="utf-8", errors="replace", env=env or build_env())
    except FileNotFoundError:
        return 127, f"{args[0]}: not found", 0.0
    elapsed_ms = (time.perf_counter() - start) * 1000
    output = (result.stdout or "") + (result.stderr or "") if capture else ""
    return result.returncode, output, elapsed_ms


def which(cmd: str) -> Path | None:
    found = shutil.which(cmd, path=build_env().get("PATH"))
    return Path(found) if found else None


# =============================================================================
# environment
# =============================================================================

def pinned_toolchain() -> str:
    text = (ROOT_DIR / "rust-toolchain.toml").read_text(encoding="utf-8")
    m = re.search(r'channel\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else "stable"


def newest_llc() -> Path | None:
    """The newest `llc-NN` (NN >= MIN_LLVM) on PATH, else a plain `llc` if new enough."""
    path = os.environ.get("PATH", "")
    best: tuple[int, Path] | None = None
    for d in path.split(os.pathsep):
        p = Path(d)
        if not p.is_dir():
            continue
        for f in p.glob("llc-*"):
            m = re.fullmatch(r"llc-(\d+)", f.name)
            if m and int(m.group(1)) >= MIN_LLVM and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), f)
    if best:
        return best[1]
    plain = shutil.which("llc")
    return Path(plain) if plain else None


_ENV: dict | None = None


def windows_toolchain_env(env: dict) -> dict:
    """Compile the isolated vcv-rs helper before any CUDA build dependency runs."""
    cargo = shutil.which("cargo", path=env.get("PATH"))
    if not cargo:
        raise RuntimeError("Rust/Cargo not found; install Rust from https://rustup.rs/")
    target = ROOT_DIR / "target" / "bootstrap"
    command = [cargo, "+stable", "build", "--manifest-path", str(ROOT_DIR / "xtask" / "Cargo.toml"),
               "--release", "--target-dir", str(target)]
    result = subprocess.run(command, cwd=ROOT_DIR, env=env)
    if result.returncode:
        raise RuntimeError("Failed to build the vcv-rs toolchain helper")
    result = subprocess.run([str(target / "release" / "frac-toolchain.exe")],
                            cwd=ROOT_DIR, env=env, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise RuntimeError("MSVC environment setup failed (vcv-rs)")
    return json.loads(result.stdout)


def build_env() -> dict:
    """os.environ plus the defaults cargo-oxide needs (CUDA_HOME, PATH, CUDA_OXIDE_LLC)."""
    global _ENV
    if _ENV is None:
        env = dict(os.environ)
        extra = [str(Path.home() / ".cargo" / "bin")]
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
        if IS_WINDOWS:
            # Normalize case: Windows treats Path/PATH as the same variable.
            env = {k.upper(): v for k, v in env.items()}
            env.update(windows_toolchain_env(env))
        else:
            cuda = Path(env.get("CUDA_HOME", "/usr/local/cuda"))
            env.setdefault("CUDA_HOME", str(cuda))
            env["PATH"] = os.pathsep.join([str(cuda / "bin"), env["PATH"]])
        if "CUDA_OXIDE_LLC" not in env:
            llc = shutil.which("llc", path=env["PATH"]) if IS_WINDOWS else newest_llc()
            if llc:
                env["CUDA_OXIDE_LLC"] = str(llc)
        _ENV = env
    return _ENV


def check_cargo() -> bool:
    if not which("cargo"):
        err("Rust/Cargo not found")
        step("Install Rust from https://rustup.rs/")
        return False
    return True


# =============================================================================
# doctor
# =============================================================================

def installed_oxide_matches(output: str) -> bool:
    """Cargo reports git installs as package headings ending in (URL#short-SHA)."""
    for line in output.splitlines():
        match = re.fullmatch(r"cargo-oxide v[^ ]+ \((.+)\):", line.strip())
        if not match:
            continue
        source, separator, revision = match.group(1).rpartition("#")
        repository = source.split("?", 1)[0].removesuffix(".git")
        expected = CUDA_OXIDE_GIT.removesuffix(".git")
        return bool(separator and repository == expected and len(revision) >= 7
                    and CUDA_OXIDE_REV.startswith(revision))
    return False


def doctor(fix: bool, full: bool = True, gpu: bool = True) -> bool:
    """Check every toolchain piece; with `fix`, install the user-level ones (rustup, cargo-oxide).
    `gpu=False` (hosted CI) builds without a driver: no device check, no `cargo oxide doctor`."""
    env = build_env()
    passed = True

    # --- GPU / driver
    if gpu:
        code, out, _ = run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"], capture=True)
        if code == 0 and out.strip():
            ok(f"GPU: {out.strip().splitlines()[0]}")
        else:
            err("nvidia-smi failed: no NVIDIA driver / GPU visible")
            if IS_WSL:
                step("WSL2: install/update the driver on WINDOWS; never install nvidia-driver-* or cuda-drivers in WSL")
            passed = False
    else:
        step("GPU: not required (build-only)")

    # --- CUDA toolkit
    nvcc = which("nvcc")
    if nvcc:
        code, out, _ = run([str(nvcc), "--version"], capture=True)
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m and int(m.group(1)) >= MIN_CUDA_MAJOR:
            ok(f"CUDA {m.group(1)}.{m.group(2)} ({nvcc}), CUDA_HOME={env['CUDA_HOME']}")
        else:
            err(f"CUDA {MIN_CUDA_MAJOR}.x+ needed, found: {m.group(0) if m else 'unknown'}")
            passed = False
    else:
        err("nvcc not found (CUDA toolkit)")
        apt_hint("cuda-toolkit-13-3 libcurand-dev-13-3", repo=True)
        passed = False

    # --- LLVM llc + clang
    llc = env.get("CUDA_OXIDE_LLC")
    if llc and Path(llc).exists():
        ok(f"llc: {llc}")
    else:
        err(f"LLVM llc {MIN_LLVM}+ not found")
        apt_hint(f"llvm-{MIN_LLVM + 1} clang-{MIN_LLVM + 1} libclang-common-{MIN_LLVM + 1}-dev libclang-{MIN_LLVM + 1}-dev")
        passed = False
    clang = next((c for c in (f"clang-{v}" for v in range(30, MIN_LLVM - 1, -1)) if which(c)), None) or ("clang" if which("clang") else None)
    if clang:
        ok(f"clang: {clang}")
    else:
        err("clang not found (needed by bindgen)")
        passed = False

    # --- Rust toolchain
    if not check_cargo():
        return False
    channel = pinned_toolchain()
    code, out, _ = run(["rustup", "toolchain", "list"], capture=True)
    if channel in out:
        ok(f"rustup toolchain {channel}")
    elif fix:
        step(f"Installing {channel} ...")
        code, _, _ = run(["rustup", "toolchain", "install", channel, "--profile", "minimal"])
        passed &= code == 0
    else:
        err(f"rustup toolchain {channel} missing (python bootstrap.py d --fix)")
        passed = False
    code, out, _ = run(["rustup", "component", "list", "--installed", "--toolchain", channel], capture=True)
    missing = [c for c in RUST_COMPONENTS if not any(line.startswith(c) for line in out.splitlines())]
    if not missing:
        ok("components: " + " ".join(RUST_COMPONENTS))
    elif fix:
        step("Adding components: " + " ".join(missing))
        code, _, _ = run(["rustup", "component", "add", *missing, "--toolchain", channel])
        passed &= code == 0
    else:
        err("missing components: " + " ".join(missing) + " (python bootstrap.py d --fix)")
        passed = False

    # --- cargo-oxide: an existing upstream/path install can have the same version.
    oxide = which("cargo-oxide")
    code, installs, _ = run(["cargo", f"+{channel}", "install", "--list"], capture=True)
    if oxide and code == 0 and installed_oxide_matches(installs):
        ok(f"cargo-oxide: {oxide} ({CUDA_OXIDE_REV[:10]})")
    elif fix:
        step(f"Installing cargo-oxide from our fork at {CUDA_OXIDE_REV[:10]} ...")
        code, _, _ = run(["cargo", f"+{channel}", "install", "--force", "--locked",
                          "--git", CUDA_OXIDE_GIT, "--rev", CUDA_OXIDE_REV, "cargo-oxide"])
        passed &= code == 0
    else:
        err(f"cargo-oxide missing or from another source (need {CUDA_OXIDE_REV[:10]}; python bootstrap.py d --fix)")
        passed = False

    # --- cargo-oxide's own check (backend, libNVVM, nvJitLink, libdevice, ...)
    if full and gpu and passed:
        print()
        step("cargo oxide doctor")
        code, out, _ = run(["cargo", "oxide", "doctor"], capture=True)
        for line in out.splitlines():
            if "✓" in line or "✗" in line or "Environment" in line or "-" in line[:4]:
                print("    " + line)
        passed &= code == 0
    return passed


def apt_hint(packages: str, repo: bool = False) -> None:
    if repo:
        step("NVIDIA repo (WSL): https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb"
             if IS_WSL else "NVIDIA repo: https://developer.nvidia.com/cuda-downloads")
    step(f"sudo apt-get install -y {packages}")


def run_doctor(args: argparse.Namespace) -> int:
    header("DOCTOR")
    good = doctor(fix=args.fix)
    print()
    if good:
        ok("Toolchain ready")
    else:
        err("Toolchain incomplete (see above)")
    print()
    return 0 if good else 1


# =============================================================================
# build / run / render
# =============================================================================

def arch_args(args: argparse.Namespace) -> list[str]:
    """`--arch` only when given; otherwise cargo-oxide reads default-arch from
    .cargo/cuda-oxide.toml, the one place the shipped architecture is set."""
    arch = getattr(args, "arch", None)
    return ["--arch", arch] if arch else []


def build(args: argparse.Namespace) -> int:
    no_gpu = getattr(args, "no_gpu", False)
    if not doctor(fix=False, full=False, gpu=not no_gpu):
        err("Toolchain incomplete: python bootstrap.py d --fix")
        return 1
    cmd = ["cargo", "oxide", "build", *arch_args(args)]
    env = build_env()
    if getattr(args, "cubins", False):
        env["CUDA_OXIDE_CUBIN_ARCHS"] = CUBIN_ARCHS
        step(f"cargo oxide build (release; embedded PTX + cubins {CUBIN_ARCHS})")
    else:
        step("cargo oxide build (release; embedded PTX)")
    print()
    code, _, elapsed = run(cmd, env=env)
    if code != 0 or not RELEASE_BIN.is_file():
        err("Build failed")
        return code or 1
    if not (no_gpu or getattr(args, "skip_warmup", False)):
        # Prime the CUDA JIT cache and open the WGPU device for this exact executable, on
        # the same GPU the workspace uses. Fail here if a module or pipeline cannot initialize.
        step("Preparing CUDA kernels and WGPU pipelines for the next application launch ...")
        code, _, warmup_ms = run([str(RELEASE_BIN), "--warmup"])
        if code != 0:
            err("GPU warmup failed; see the module/pipeline error above")
            return code
        step(f"GPU warmup: {fmt_time(warmup_ms)}")
    ok(f"Build successful ({fmt_time(elapsed)})")
    step(f"Binary: {RELEASE_BIN}")
    print()
    return 0


def run_build(args: argparse.Namespace) -> int:
    header("BUILD")
    return build(args)


def run_app(args: argparse.Namespace, app_args: list[str]) -> int:
    header("RUN")
    code = build(args)
    if code != 0:
        return code
    step(f"{BIN_NAME} {' '.join(app_args)}".rstrip())
    code, _, _ = run([str(RELEASE_BIN), *app_args])
    return code


def run_gallery(args: argparse.Namespace) -> int:
    rest = args.rest or []
    out = rest[0] if rest else "gallery"
    dims = rest[1:4] if len(rest) > 1 else ["1920", "1080", "256"]
    header(f"GALLERY -> {out}/")
    return run_app(args, ["--gallery", out, *dims])


def run_bench(args: argparse.Namespace) -> int:
    dims = args.rest or ["960", "540", "32"]
    return run_app(args, ["--bench", *dims])


def run_docs(args: argparse.Namespace) -> int:
    """Regenerate docs/: gallery contact sheet, hero renders, UI screenshots, bench table."""
    header("DOCS")
    if not which("uv"):
        err("uv not found (used to run Pillow without touching system Python)")
        return 1
    code = build(args)
    if code != 0:
        return code
    tmp = ROOT_DIR / "target" / "docs-gallery"
    shutil.rmtree(tmp, ignore_errors=True)
    step("Rendering the gallery (1280x720, 256 spp) ...")
    code, out, _ = run([str(RELEASE_BIN), "--gallery", str(tmp), "1280", "720", "256"], capture=True)
    if code != 0:
        err("Gallery render failed")
        print(out)
        return code
    (ROOT_DIR / "docs" / "bench-1280x720.txt").write_text(out, encoding="utf-8")
    snaps = [
        ("ui-menger.png", {"FRAC_SNAP_PRESET": "13"}),
        ("ui-materials.png", {"FRAC_SNAP_PRESET": "9", "FRAC_SNAP_MATERIAL": "BrushedAluminium", "FRAC_SNAP_TAB": "materials"}),
    ]
    for name, extra in snaps:
        step(f"Window screenshot {name} ...")
        env = dict(build_env(), FRAC_SNAP=str(ROOT_DIR / "docs" / name), FRAC_SNAP_SPP="256", **extra)
        run([str(RELEASE_BIN)], env=env)
    script = r'''
import glob, os, sys
from PIL import Image
src, docs = sys.argv[1], sys.argv[2]
order = ["mandelbulb","mandelbox","quaternion-julia","kifs","kleinian","pseudo-kleinian","apollonian","hybrid",
         "bulb-power-12-gold","menger-sponge","quaternion-glass-coat","bulb-julia","bulb-twisted","octahedron-kifs",
         "hybrid-bulb-kifs","pseudo-kleinian-cave"]
w, h, cols = 400, 225, 4
sheet = Image.new("RGB", (cols * w, (len(order) + cols - 1) // cols * h))
for i, n in enumerate(order):
    sheet.paste(Image.open(f"{src}/{n}.png").resize((w, h), Image.LANCZOS), ((i % cols) * w, (i // cols) * h))
sheet.save(f"{docs}/gallery.jpg", quality=90)
for n in ["mandelbulb", "bulb-power-12-gold", "menger-sponge", "pseudo-kleinian-cave"]:
    Image.open(f"{src}/{n}.png").convert("RGB").save(f"{docs}/{n}.jpg", quality=92)
for f in glob.glob(f"{docs}/ui-*.png"):
    Image.open(f).convert("RGB").save(f[:-4] + ".jpg", quality=90)
    os.remove(f)
'''
    code, out, _ = run(["uv", "run", "-q", "--with", "pillow", "python", "-c", script, str(tmp), str(ROOT_DIR / "docs")], capture=True)
    if code == 0:
        ok("docs/ updated")
    else:
        err("Image processing failed")
        print(out)
    print()
    return code


def run_install(args: argparse.Namespace) -> int:
    header("INSTALL")
    if not RELEASE_BIN.is_file() or args.force_install:
        code = build(args)
        if code != 0:
            return code
    else:
        step(f"Using existing binary: {RELEASE_BIN} (-f to rebuild)")
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    dest = INSTALL_DIR / BIN_NAME
    shutil.copy2(RELEASE_BIN, dest)
    ok(f"Installed {dest}")
    if str(INSTALL_DIR) not in os.environ.get("PATH", "").split(os.pathsep):
        warn(f"{INSTALL_DIR} is not on PATH")
    print()
    return 0


def run_check(_args: argparse.Namespace) -> int:
    header("CHECK")
    passed = True

    step("Checking formatting ...")
    code, _, elapsed = run(["cargo", "fmt", "--check"])
    if code == 0:
        ok(f"Format OK ({fmt_time(elapsed)})")
    else:
        err("Format check failed (cargo fmt)")
        passed = False

    print()
    # clippy only type-checks (no codegen), so it runs on the plain backend; build.rs lets it
    # through (CLIPPY_ARGS).
    step("Running clippy ...")
    code, _, elapsed = run(["cargo", "clippy", "--release", "--all-targets", "--locked", "--", "-D", "warnings"])
    if code == 0:
        ok(f"Clippy OK ({fmt_time(elapsed)})")
    else:
        err("Clippy failed")
        passed = False

    print()
    if passed:
        ok("All checks passed")
    else:
        err("Some checks failed")
    print()
    return 0 if passed else 1


def platform_name() -> str:
    """`windows-x86_64` / `linux-x86_64`: the release archive's platform tag."""
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(machine, machine)
    return f"{'windows' if IS_WINDOWS else 'linux'}-{arch}"


def release_version() -> str:
    """The tag on a tag build (`v0.3.0`), else the crate version and commit (`0.2.0-abcdef1`)."""
    if os.environ.get("GITHUB_REF_TYPE") == "tag" and os.environ.get("GITHUB_REF_NAME"):
        return os.environ["GITHUB_REF_NAME"]
    manifest = (ROOT_DIR / "Cargo.toml").read_text(encoding="utf-8")
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', manifest)
    code, sha, _ = run(["git", "rev-parse", "--short", "HEAD"], capture=True)
    return f"{version.group(1) if version else '0.0.0'}-{sha.strip() if code == 0 else 'local'}"


def package() -> Path:
    """dist/warpbro-<version>-<platform>.zip: the release binary plus README and CHANGELOG,
    under one top-level folder of the same name."""
    import zipfile
    stem = f"warpbro-{release_version()}-{platform_name()}"
    DIST_DIR.mkdir(exist_ok=True)
    archive = DIST_DIR / f"{stem}.zip"
    archive.unlink(missing_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(RELEASE_BIN, f"{stem}/{BIN_NAME}")
        for doc in ("README.md", "CHANGELOG.md"):
            z.write(ROOT_DIR / doc, f"{stem}/{doc}")
    return archive


def run_ci(args: argparse.Namespace) -> int:
    """Everything a hosted, GPU-less runner can verify, in order, stopping at the first failure:
    toolchain, fmt + clippy, the ordinary suite without device tests (GPU_TEST_PREFIXES, see
    src/test_gpu.rs), the release build, and the release archive."""
    header("CI")
    args.no_gpu = True
    stages = [
        ("Toolchain", lambda: 0 if doctor(fix=True, full=False, gpu=False) else 1),
        ("Format and clippy", lambda: run_check(args)),
        # The same arch as the release build: equal backend flags let the build reuse every
        # crate the tests compiled (otherwise the whole graph builds twice).
        ("Tests without a GPU", lambda: run(["cargo", "oxide", "test", *arch_args(args), "--",
                                             "--release", "--locked", "--",
                                             *(a for p in GPU_TEST_PREFIXES for a in ("--skip", p))])[0]),
        ("Release build", lambda: build(args)),
    ]
    for name, stage in stages:
        step(f"--- {name}")
        code = stage()
        if code != 0:
            err(f"{name} failed")
            return code
    archive = package()
    ok(f"Packaged {archive.relative_to(ROOT_DIR)} ({archive.stat().st_size / 1e6:.1f} MB)")
    return 0


# =============================================================================
# kernels: registers, local memory, spills
# =============================================================================

KERNEL_PTX = ROOT_DIR / "target" / "oxide-ptx" / "fractal_kernels.ptx"
KERNEL_REPORT_DIR = ROOT_DIR / "target" / "oxide-ptx" / "kernels"


def ptx_call_graph(source: str) -> tuple[set[str], dict[str, set[str]]]:
    """Read entry names and direct device calls before ptxas inlines/removes helpers."""
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    declaration = re.compile(
        r"(?m)^\s*(?:\.(?:visible|extern|weak)\s+)*\.(entry|func)\s+"
        r"(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)\s*\("
    )
    call = re.compile(
        r"\bcall(?:\.\w+)*\s+(?:\([^;]*?\)\s*,\s*)?([A-Za-z_$][\w$]*)\s*,"
    )
    entries: set[str] = set()
    graph: dict[str, set[str]] = {}
    for match in declaration.finditer(source):
        body = source.find("{", match.end())
        terminator = source.find(";", match.end())
        if body < 0 or 0 <= terminator < body:  # External declaration, not a body.
            continue
        depth, cursor = 1, body + 1
        while cursor < len(source) and depth:
            depth += (source[cursor] == "{") - (source[cursor] == "}")
            cursor += 1
        if depth:
            raise RuntimeError(f"Unterminated PTX function {match[2]}")
        name = match[2]
        if name in graph:
            raise RuntimeError(f"Duplicate PTX function {name}")
        graph[name] = set(call.findall(source[body + 1:cursor - 1]))
        if match[1] == "entry":
            entries.add(name)
    if not entries:
        raise RuntimeError("No PTX entry points found; resource coverage cannot be verified")
    return entries, graph


def parse_ptxas_stats(output: str) -> dict[str, dict]:
    """Preserve ordered reports and per-symbol maxima, without guessing helper ownership.

    ptxas repeats a helper while reporting different entry compilations. A log block
    is validated only when it starts with its own unique entry properties and has no
    other entry properties. Its heading is provenance, never a helper owner claim.
    Parallel/interleaved blocks remain unattributed and retain every occurrence.
    """
    entry_pattern = re.compile(r"Compiling entry function '([^']+)'")
    compiled = [match[1] for match in entry_pattern.finditer(output)]
    if len(compiled) != len(set(compiled)):
        raise RuntimeError("Duplicate compiling-entry header")
    entries = set(compiled)
    occurrences: list[dict] = []
    blocks: list[dict] = []
    entry_sections: set[str] = set()
    block = current = None
    for line_number, line in enumerate(output.splitlines(), 1):
        if entry := entry_pattern.search(line):
            block = {"entry": entry[1], "occurrences": []}
            blocks.append(block)
            current = None
        properties = re.search(r"""Function properties for\s+['"]?([^'"\s:]+)""", line)
        if properties:
            name = properties[1]
            if name in entries:
                if name in entry_sections:
                    raise RuntimeError(f"Duplicate entry resource section for {name}")
                entry_sections.add(name)
            current = {
                "function": name, "ordinal": len(occurrences) + 1, "line": line_number,
                "compiler_context": name if name in entries else None,
                "log_block_entry": block["entry"] if block else None,
                "log_block_validated": False, "resources": {},
            }
            occurrences.append(current)
            if block is not None:
                block["occurrences"].append(current)
        for label, pattern in (
            ("frame", r"(\d+) bytes stack frame"),
            ("st", r"(\d+) bytes spill stores"),
            ("ld", r"(\d+) bytes spill loads"),
            ("regs", r"Used (\d+) registers"),
        ):
            matches = re.findall(pattern, line)
            if not matches:
                continue
            if current is None:
                raise RuntimeError(f"Resource field without function properties at line {line_number}")
            if len(matches) != 1 or label in current["resources"]:
                raise RuntimeError(f"Duplicate {label} field for {current['function']} at line {line_number}")
            current["resources"][label] = int(matches[0])

    for block in blocks:
        reports = block["occurrences"]
        reported_entries = [item["function"] for item in reports if item["function"] in entries]
        validated = bool(reports) and reports[0]["function"] == block["entry"] and reported_entries == [block["entry"]]
        if validated:
            seen: set[str] = set()
            for item in reports:
                if item["function"] in seen:
                    raise RuntimeError(f"Duplicate resource section for {item['function']} in entry log block {block['entry']}")
                seen.add(item["function"])
                item["log_block_validated"] = True

    stats: dict[str, dict] = {}
    for item in occurrences:
        row = stats.setdefault(item["function"], {"occurrences": []})
        row["occurrences"].append(item)
        for field, value in item["resources"].items():
            row[field] = max(row[field], value) if field in row else value
    return stats


def read_ptxas_receipt(
    receipt: Path, ptx: Path, arch: str, ptx_sha256: str,
    entries: set[str], calls: dict[str, set[str]],
) -> tuple[dict[str, dict], dict] | None:
    """Validate a production receipt; a different PTX hash is historical evidence."""
    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON field {key} in {receipt}")
            result[key] = value
        return result

    metadata = json.loads(receipt.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    fields = {"schema", "kind", "module", "host_target", "arch", "ptx_name", "ptx_sha256",
              "ptxas_path", "ptxas_sha256", "options", "entries", "raw_log",
              "assembly_succeeded", "resource_coverage_complete"}
    if not isinstance(metadata, dict) or metadata.keys() != fields:
        raise RuntimeError(f"Unknown or missing receipt fields: {receipt}")
    if type(metadata["schema"]) is not int or metadata["schema"] != 1 or metadata["kind"] != "ptxas-assembly-receipt":
        raise RuntimeError(f"Unsupported assembly receipt schema: {receipt}")
    for field in ("ptx_sha256", "ptxas_sha256"):
        if not isinstance(metadata[field], str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[field]):
            raise RuntimeError(f"Invalid {field} in {receipt}")
    if metadata["ptx_sha256"] != ptx_sha256:
        return None
    if (metadata["module"] != ptx.stem or metadata["ptx_name"] != ptx.name
            or metadata["arch"] != arch or not isinstance(metadata["host_target"], str)
            or not metadata["host_target"]):
        raise RuntimeError(f"Receipt module, PTX name, host or architecture mismatch: {receipt}")
    if metadata["assembly_succeeded"] is not True or metadata["resource_coverage_complete"] is not None:
        raise RuntimeError(f"Unsupported assembly outcome/coverage claim: {receipt}")
    inventory = metadata["entries"]
    if (not isinstance(inventory, list) or any(not isinstance(name, str) for name in inventory)
            or len(inventory) != len(set(inventory)) or set(inventory) != entries):
        raise RuntimeError(f"Missing, duplicate or unknown receipt entries: {receipt}")
    options = metadata["options"]
    if not isinstance(options, list) or any(not isinstance(option, str) for option in options):
        raise RuntimeError(f"Invalid compiler options in {receipt}")
    keys = [option.split("=", 1)[0] for option in options]
    allowed = {"--gpu-name", "--fmad", "--split-compile", "--verbose",
               "--generate-line-info", "--device-debug"}
    if (len(keys) != len(set(keys)) or set(keys) - allowed
            or f"--gpu-name={arch}" not in options or "--verbose" not in options
            or sum(option in ("--fmad=true", "--fmad=false") for option in options) != 1
            or ("--generate-line-info" in options and "--device-debug" in options)
            or any(option.startswith(("--verbose=", "--generate-line-info=", "--device-debug="))
                   for option in options)):
        raise RuntimeError(f"Unknown, duplicate or inconsistent compiler options: {receipt}")
    for option in options:
        if option.startswith("--split-compile"):
            value = option.partition("=")[2]
            if not re.fullmatch(r"[1-9][0-9]*", value) or int(value) > 0xffffffff:
                raise RuntimeError(f"Invalid split-compilation budget in {receipt}")
    if not isinstance(metadata["ptxas_path"], str):
        raise RuntimeError(f"Invalid assembler path in {receipt}")
    tool = Path(metadata["ptxas_path"])
    if not tool.is_absolute() or not tool.is_file():
        raise RuntimeError(f"Receipt assembler is unavailable: {tool}")
    with tool.open("rb") as source:
        tool_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if tool_sha256 != metadata["ptxas_sha256"]:
        raise RuntimeError(f"Receipt assembler hash mismatch: {receipt}")
    raw_name = metadata["raw_log"]
    if not isinstance(raw_name, str) or not raw_name or raw_name in (".", "..") or "/" in raw_name or "\\" in raw_name:
        raise RuntimeError(f"Invalid raw-log name in {receipt}")
    raw = receipt.parent / raw_name
    raw_bytes = raw.read_bytes()
    output = raw_bytes.decode("utf-8")
    compiled = re.findall(r"Compiling entry function '([^']+)' for '([^']+)'", output)
    if (len(compiled) != len(entries) or {name for name, _ in compiled} != entries
            or any(target != arch for _, target in compiled)):
        raise RuntimeError(f"Missing, duplicate, unknown or wrong-architecture raw entries: {raw}")
    stats = parse_ptxas_stats(output)
    coverage = kernel_resource_report(stats, entries, calls)
    if not coverage["complete"]:
        raise RuntimeError(f"Incomplete production receipt {receipt}: " + "; ".join(coverage["coverage_errors"]))
    return stats, {
        "resource_source": "production-aot-receipt",
        "ptxas_path": str(tool.resolve()), "ptxas_sha256": tool_sha256,
        "compiler_options": options, "host_target": metadata["host_target"],
        "raw_log": str(raw), "raw_log_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def retained_kernel_stats(
    ptx: Path, arch: str, ptx_sha256: str, entries: set[str], calls: dict[str, set[str]],
) -> tuple[dict[str, dict], dict]:
    """Accept equivalent repeated builds, reject conflicting evidence without guessing."""
    accepted = []
    for receipt in sorted(ptx.parent.glob(f"{ptx.stem}.*.ptxas-report/{arch}.ptxas.json")):
        if result := read_ptxas_receipt(receipt, ptx, arch, ptx_sha256, entries, calls):
            accepted.append((receipt, result))
    if not accepted:
        raise RuntimeError(f"No production {arch} receipt matches this PTX; build with native cubins, "
                           "or explicitly use k --assemble for a separate standalone assembly")
    def identity(result: tuple[dict, dict]) -> tuple[dict, dict]:
        stats, provenance = result
        variants = {
            name: sorted(tuple(sorted(item["resources"].items())) for item in row["occurrences"])
            for name, row in stats.items()
        }
        return variants, {key: value for key, value in provenance.items()
                          if key not in ("raw_log", "raw_log_sha256")}

    canonical = identity(accepted[0][1])
    if any(identity(result) != canonical for _, result in accepted[1:]):
        raise RuntimeError(f"Conflicting production {arch} receipts for this PTX: "
                           + ", ".join(str(path) for path, _ in accepted))
    stats, provenance = accepted[0][1]
    return stats, {
        **provenance, "receipts": [str(path) for path, _ in accepted],
        "raw_logs": [
            {"receipt": str(path), "path": result[1]["raw_log"], "sha256": result[1]["raw_log_sha256"]}
            for path, result in accepted
        ],
    }


def ptxas_kernel_stats(ptx: Path, arch: str, line_info: bool) -> tuple[dict[str, dict], Path, dict]:
    """Assemble PTX and retain entry/helper resources plus the raw compiler receipt."""
    KERNEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cubin = KERNEL_REPORT_DIR / f"{arch}.cubin"
    options = [f"-arch={arch}", "-O3", "--verbose", *(["--generate-line-info"] if line_info else [])]
    cmd = ["ptxas", *options, str(ptx), "-o", str(cubin)]
    code, output, _ = run(cmd, capture=True)
    cubin.with_suffix(".ptxas.txt").write_text(output, encoding="utf-8")
    if code != 0:
        raise RuntimeError(f"ptxas {arch} failed:\n{output}")
    return parse_ptxas_stats(output), cubin, {
        "resource_source": "standalone-ptxas", "compiler_options": options,
        "raw_log": str(cubin.with_suffix(".ptxas.txt")), "cubin": str(cubin),
    }


def kernel_resource_report(
    stats: dict[str, dict], entries: set[str], calls: dict[str, set[str]],
) -> dict:
    """Verify entry coverage and associate reported helpers with their PTX callers.

    ptxas can inline a PTX helper away. Such helpers are listed as unreported, not
    assigned invented zero resources; their surviving code belongs to the caller.
    Gates inspect every reported function, including helpers outside a parsed chain.
    """
    errors = [f"Missing entry resource report: {name}" for name in sorted(entries - stats.keys())]
    errors.extend(f"Unknown reported function: {name}" for name in sorted(stats.keys() - calls.keys()))
    occurrences = []
    for name, row in sorted(stats.items()):
        reports = row["occurrences"]
        if not reports:
            errors.append(f"No resource occurrences for {name}")
        if name in entries and len(reports) != 1:
            errors.append(f"Entry {name} has {len(reports)} resource occurrences; expected exactly one")
        required = {"frame", "st", "ld"} | ({"regs"} if name in entries else set())
        for report in reports:
            if missing := required - report["resources"].keys():
                errors.append(f"Incomplete resource report for {name} occurrence {report['ordinal']} "
                              f"(line {report['line']}): missing {', '.join(sorted(missing))}")
            occurrences.append(report)

    chains = {}
    for entry in sorted(entries):
        reachable: set[str] = set()
        pending = list(calls.get(entry, ()))
        while pending:
            name = pending.pop()
            if name not in reachable and name != entry:
                reachable.add(name)
                pending.extend(calls.get(name, ()))
        chains[entry] = {
            "ptx_callees": sorted(reachable),
            "reported_helpers": sorted(reachable & stats.keys()),
            "unreported_ptx_helpers": sorted(reachable - stats.keys()),
        }
    return {
        "complete": not errors,
        "coverage_errors": errors,
        "entries": chains,
        "functions": dict(sorted(stats.items())),
        "function_aggregation": "per-field maxima across occurrences; the maxima need not coexist in one variant",
        "helper_context_attribution": "unknown; log_block_entry records only the observed log heading, not a proven helper owner",
        "reported_occurrences": len(occurrences),
        "spilling_functions": sorted(name for name, row in stats.items() if row.get("st") or row.get("ld")),
        "local_frame_functions": sorted(name for name, row in stats.items() if row.get("frame")),
        "spilling_occurrences": sorted(
            (item for item in occurrences if item["resources"].get("st") or item["resources"].get("ld")),
            key=lambda item: item["ordinal"],
        ),
        "local_frame_occurrences": sorted(
            (item for item in occurrences if item["resources"].get("frame")),
            key=lambda item: item["ordinal"],
        ),
    }


def local_memory_sites(cubin: Path) -> dict[str, list[tuple[int, int, str]]]:
    """Every local load (LDL) and store (STL) of every function in `cubin`, attributed to the
    innermost source line with its inlining chain (`nvdisasm --print-line-info-inline`).
    Returns, per function, `(stores, loads, "file:line <- caller:line ...")`, busiest first."""
    code, sass, _ = run(["nvdisasm", "--print-line-info-inline", "--print-code", str(cubin)], capture=True)
    if code != 0:
        raise RuntimeError(f"nvdisasm failed:\n{sass[:2000]}")
    marker = re.compile(r'//## File "([^"]+)", line (\d+)((?: inlined at "[^"]+", line \d+)*)')
    inlined = re.compile(r'inlined at "([^"]+)", line (\d+)')
    access = re.compile(r"\b(STL|LDL)(?:\.\w+)*\s")
    name = lambda path: Path(path.replace("\\\\", "\\")).name  # noqa: E731
    counts: dict[str, dict[str, list[int]]] = {}
    function, where, in_marker = None, "?", False
    for line in sass.splitlines():
        if m := re.match(r"^\s*\.text\.(\S+):", line):
            function, in_marker = m.group(1), False
        elif m := marker.search(line):
            # A block of markers precedes an instruction; the first is the innermost.
            if not in_marker:
                where = " <- ".join([f"{name(m.group(1))}:{m.group(2)}",
                                     *(f"{name(f)}:{n}" for f, n in inlined.findall(m.group(3)))])
            in_marker = True
        else:
            in_marker = False
            if function and (m := access.search(line)):
                site = counts.setdefault(function, {}).setdefault(where, [0, 0])
                site[0 if m.group(1) == "STL" else 1] += 1
    return {f: sorted(((st, ld, w) for w, (st, ld) in sites.items()), key=lambda s: -(s[0] + s[1]))
            for f, sites in counts.items()}


def run_kernels(args: argparse.Namespace) -> int:
    """Report resources for all entries and helpers; missing entry coverage always fails.

    --no-spills gates compiler register-spill bytes only. --strict also rejects
    local frames (arrays/call ABI storage), retaining its stronger historical meaning.
    """
    header("KERNELS")
    assemble = getattr(args, "assemble", False)
    if args.lines and not assemble:
        err("--lines requires --assemble: production receipts retain resources, not temporary cubins")
        return 1
    archs = args.rest or CUBIN_ARCHS.split(",")
    if args.build or args.lines:
        env = build_env()
        if args.lines:
            # Line tables add .loc directives only; optimisation and register allocation are unchanged.
            env["CUDA_OXIDE_DEBUG"] = "line-tables"
        step("cargo oxide build" + (" (line tables)" if args.lines else ""))
        code, _, elapsed = run(["cargo", "oxide", "build", *arch_args(args)], env=env)
        if code != 0:
            err("Build failed")
            return code
        ok(f"PTX built ({fmt_time(elapsed)})")
    if not KERNEL_PTX.is_file():
        err(f"{KERNEL_PTX.relative_to(ROOT_DIR)} not found: build first (b, or k --build)")
        return 1
    ptx_bytes = KERNEL_PTX.read_bytes()
    entries, calls = ptx_call_graph(ptx_bytes.decode("utf-8"))
    ptx_sha256 = hashlib.sha256(ptx_bytes.removesuffix(b"\0")).hexdigest()
    KERNEL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    coverage_failed = False
    spills: list[str] = []
    frames: list[str] = []
    for arch in archs:
        try:
            if assemble:
                warn(f"{arch}: explicitly assembling standalone PTX; this is separate from production AOT")
                stats, cubin, provenance = ptxas_kernel_stats(KERNEL_PTX, arch, args.lines)
            else:
                stats, provenance = retained_kernel_stats(KERNEL_PTX, arch, ptx_sha256, entries, calls)
                step(f"{arch}: importing {len(provenance['receipts'])} verified production assembly receipt(s)")
                cubin = None
        except (OSError, ValueError, RuntimeError) as error:
            err(f"{arch}: {error}")
            return 1
        report = {
            "schema": 1, "arch": arch, "ptx": str(KERNEL_PTX),
            "ptx_sha256": ptx_sha256, "provenance": provenance,
            "call_graph_source": "PTX before ptxas optimization",
            **kernel_resource_report(stats, entries, calls),
        }
        report_path = KERNEL_REPORT_DIR / f"{arch}-resources.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\n  {arch}: {'function (E=entry, H=helper)':56} {'regs':>4} {'frame':>6} {'spill st':>8} {'spill ld':>8}")
        for name, row in sorted(stats.items(), key=lambda kv: (-kv[1].get("frame", 0), kv[0])):
            kind = "E" if name in entries else "H"
            print(f"  {kind} {name:58} {row.get('regs', '-'):>4} {row.get('frame', '-'):>6}"
                  f" {row.get('st', '-'):>8} {row.get('ld', '-'):>8}")
        for error in report["coverage_errors"]:
            err(f"{arch}: {error}")
        coverage_failed |= not report["complete"]
        spills.extend(f"{arch}/{name}" for name in report["spilling_functions"])
        frames.extend(f"{arch}/{name}" for name in report["local_frame_functions"])
        step(f"resource report: {report_path.relative_to(ROOT_DIR)}")
        if args.lines:
            report_path = KERNEL_REPORT_DIR / f"{arch}-local.txt"
            sites = local_memory_sites(cubin)
            with report_path.open("w", encoding="utf-8") as out:
                for function, rows in sorted(sites.items(), key=lambda kv: -sum(a + b for a, b, _ in kv[1])):
                    out.write(f"\n== {function}: STL={sum(r[0] for r in rows)} LDL={sum(r[1] for r in rows)}\n")
                    out.writelines(f"  STL={st:4} LDL={ld:4}  {w}\n" for st, ld, w in rows)
            step(f"local loads/stores by source line: {report_path.relative_to(ROOT_DIR)}")
    print()
    spill_gate = args.strict or args.no_spills
    if spills:
        (err if spill_gate else warn)(f"{len(spills)} function/arch pairs spill registers")
    if frames:
        (err if args.strict else warn)(f"{len(frames)} function/arch pairs keep local frames (separate from register spills)")
    if coverage_failed or (spill_gate and spills) or (args.strict and frames):
        return 1
    if not spills:
        ok("No reported entry or helper spills registers; all PTX entries covered")
    if not frames:
        ok("No reported entry or helper keeps a local frame")
    return 0


def run_clean(_args: argparse.Namespace) -> int:
    header("CLEAN")
    code, _, elapsed = run(["cargo", "clean"])
    if code == 0:
        ok(f"cargo clean ({fmt_time(elapsed)})")
    else:
        err("cargo clean failed")
        return code
    # cuda-oxide drops its pipeline dumps next to the manifest.
    stray = [p for pat in ("*.ll", "*.ptx") for p in ROOT_DIR.glob(pat)]
    for p in stray:
        p.unlink()
    if stray:
        ok(f"Removed {len(stray)} *.ll / *.ptx dumps")
    print()
    return 0


HELP_TEXT = """
FRAC-RS BUILD SYSTEM

Kernels are Rust compiled to PTX by cuda-oxide: always build through `cargo oxide`
(this script, or `cargo ob` / `cargo or`). Plain `cargo build` cannot link them.
On Windows this script prepares MSVC / SDK / CUDA via vcv-rs automatically.
Direct cargo aliases require an already configured Developer PowerShell.

COMMANDS
  d       doctor: check driver, CUDA, LLVM, clang, stable toolchain, cargo-oxide
            --fix   install the missing rustup toolchain / components / cargo-oxide
  b       build (cargo oxide build) -> target/release/WarpBro
  r       build + run the browser; arguments after `--` go to WarpBro
  g       render every preset:  g [DIR [W H SPP]]    (gallery 1920 1080 256)
  bench   time every preset:    bench [W H SPP]      (960 540 32)
  docs    regenerate docs/ (README images, bench table)
  i       install to ~/.local/bin (copies the release binary; -f rebuilds)
  c       cargo fmt --check + cargo clippy --all-targets -D warnings
  k       import verified production entry/helper resources:  k [ARCH...]
            --build   rebuild the PTX first     --lines   map local accesses (requires --assemble)
            --assemble  explicitly run standalone ptxas if production receipts are unavailable
            --no-spills  fail on register spills     --strict  also fail on local frames
            Missing entry/resource diagnostics always fail; frames and spills are distinct.
  ci      what hosted CI runs (no GPU): d --fix, c, tests --skip cuda_ gpu_, build, dist/*.zip
  cl      cargo clean (+ *.ll / *.ptx dumps)
  h       help

OPTIONS
  --arch sm_86     target architecture for b / r / ci (default: default-arch in .cargo/cuda-oxide.toml)
  --no-gpu         b: build without a GPU (no device check, no GPU warmup)
  --cubins         b / ci: also embed cubins for sm_75/86/89/120 (release archives; +1-2 min)
  --fix            doctor: install what can be installed without sudo
  -f, --force      install: rebuild first

CARGO ALIASES (.cargo/config.toml)
  cargo ob | cargo or | cargo ogallery DIR W H SPP | cargo obench W H SPP

EXAMPLES
  python bootstrap.py d --fix
  python bootstrap.py b
  python bootstrap.py r
  python bootstrap.py r -- --bench 1920 1080 64
  python bootstrap.py g out 3840 2160 1024
  python bootstrap.py docs
  python bootstrap.py i
  python bootstrap.py ci
  python bootstrap.py b --cubins
  python bootstrap.py k sm_86 --lines
"""

COMMANDS = ["d", "b", "r", "g", "bench", "docs", "i", "c", "k", "ci", "cl", "h"]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    C.init()

    argv = sys.argv[1:]
    app_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, app_args = argv[:i], argv[i + 1:]

    parser = argparse.ArgumentParser(
        description="WarpBro build system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS, default="h", help=", ".join(COMMANDS))
    parser.add_argument("rest", nargs="*", help="positional arguments for g / bench")
    parser.add_argument("--arch", help="CUDA architecture, e.g. sm_86")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Build without warming CUDA and WGPU (for packaging/cross-builds)")
    parser.add_argument("--no-gpu", action="store_true", help="build without a GPU (no device check, no warmup)")
    parser.add_argument("--cubins", action="store_true",
                        help=f"b / ci: also embed cubins for {CUBIN_ARCHS} (release archives)")
    parser.add_argument("--build", action="store_true", help="k: rebuild the PTX first")
    parser.add_argument("--assemble", action="store_true", help="k: explicitly assemble standalone PTX instead of importing production receipts")
    parser.add_argument("--lines", action="store_true", help="k: rebuild with line tables and map local accesses (requires --assemble)")
    parser.add_argument("--strict", action="store_true", help="k: fail on entry/helper local frames or register spills")
    parser.add_argument("--no-spills", action="store_true", help="k: fail on entry/helper register spills; report frames separately")
    parser.add_argument("--fix", action="store_true", help="doctor: install missing user-level tools")
    parser.add_argument("-f", "--force", dest="force_install", action="store_true", help="install: rebuild first")

    args = parser.parse_args(argv)

    if args.command == "h":
        print(HELP_TEXT)
        return 0

    if args.command == "r":
        return run_app(args, app_args)

    dispatch = {
        "d": run_doctor,
        "b": run_build,
        "g": run_gallery,
        "bench": run_bench,
        "docs": run_docs,
        "i": run_install,
        "c": run_check,
        "k": run_kernels,
        "ci": run_ci,
        "cl": run_clean,
    }
    handler = dispatch.get(args.command)
    if handler:
        return handler(args)

    print(HELP_TEXT)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, json.JSONDecodeError) as error:
        err(str(error))
        sys.exit(1)
