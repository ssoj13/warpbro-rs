#!/usr/bin/env python3
"""
bootstrap.py - Unified local build/run/check script for WarpBro.

Cross-platform-ish (Linux / WSL2), Python 3, stdlib only. Adapted from the gitnexus-rs
bootstrap.

WarpBro builds through cuda-oxide: its GPU kernels (src/gpu.rs) are Rust compiled to PTX by the
rustc_codegen_cuda backend, which `cargo oxide` enables. Builds warm the CUDA driver's JIT
cache before reporting success, so the next workspace launch does not wait for a cold compile.
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

CUDA_OXIDE_GIT = "ssh://git@github.com/ssoj13/cuda-oxide-windows.git"
CUDA_OXIDE_REV = "be40bf23b6636f2eb053cb7aa1b915fd707d25d5"
RUST_COMPONENTS = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "rustfmt", "llvm-tools"]
MIN_LLVM = 21
MIN_CUDA_MAJOR = 13
# CI builds without a GPU to detect: PTX for Turing (sm_75) and newer, JIT-compiled by the
# driver for the installed GPU on first launch.
CI_ARCH = "sm_75"
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

def build(args: argparse.Namespace) -> int:
    no_gpu = getattr(args, "no_gpu", False)
    if not doctor(fix=False, full=False, gpu=not no_gpu):
        err("Toolchain incomplete: python bootstrap.py d --fix")
        return 1
    cmd = ["cargo", "oxide", "build"]
    if getattr(args, "arch", None):
        cmd += ["--arch", args.arch]
    step("cargo oxide build (release; embedded PTX)")
    print()
    code, _, elapsed = run(cmd)
    if code != 0 or not RELEASE_BIN.is_file():
        err("Build failed")
        return code or 1
    if not (no_gpu or getattr(args, "skip_cuda_warmup", False)):
        # Prime the driver's cache for this exact executable, on the same GPU the
        # workspace uses. Fail here if its module/parameter ABI cannot initialize.
        step("Preparing CUDA kernels for the next application launch ...")
        code, _, warmup_ms = run([str(RELEASE_BIN), "--warmup-cuda"])
        if code != 0:
            err("CUDA warmup failed; see the module/parameter error above")
            return code
        step(f"CUDA warmup: {fmt_time(warmup_ms)}")
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
    src/test_gpu.rs), the release build for CI_ARCH, and the release archive."""
    header("CI")
    args.no_gpu = True
    args.arch = args.arch or CI_ARCH
    stages = [
        ("Toolchain", lambda: 0 if doctor(fix=True, full=False, gpu=False) else 1),
        ("Format and clippy", lambda: run_check(args)),
        ("Tests without a GPU", lambda: run(["cargo", "oxide", "test", "--", "--release", "--locked", "--",
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
  ci      what hosted CI runs (no GPU): d --fix, c, tests --skip cuda_ gpu_, build, dist/*.zip
  cl      cargo clean (+ *.ll / *.ptx dumps)
  h       help

OPTIONS
  --arch sm_86     target architecture for b / r / ci (default: detected GPU; ci: sm_75)
  --no-gpu         b: build without a GPU (no device check, no CUDA warmup)
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
"""

COMMANDS = ["d", "b", "r", "g", "bench", "docs", "i", "c", "ci", "cl", "h"]


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
    parser.add_argument("--skip-cuda-warmup", action="store_true",
                        help="Build without initializing CUDA (for packaging/cross-builds)")
    parser.add_argument("--no-gpu", action="store_true", help="build without a GPU (no device check, no warmup)")
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
