//! `frac-toolchain`: prints the native build environment as one JSON object of variables on
//! stdout. On Windows it is MSVC (VS 2022 / 2026), the Windows SDK and UCRT, CUDA and vcpkg,
//! found by `vcv-rs` and merged with the inherited PATH / INCLUDE / LIB / LIBPATH; elsewhere
//! an empty object. `bootstrap.py` builds and runs it (a separate workspace, so it compiles
//! without the application's CUDA dependencies) and starts cargo inside that environment.
use std::collections::BTreeMap;

#[cfg(windows)]
fn native_env() -> Result<BTreeMap<String, String>, String> {
    use vcv_rs::Arch;
    use vcv_rs::detect::{detect_sdk, detect_ucrt, detect_vs_range};
    use vcv_rs::env::{add_cuda, add_vcpkg, build_env, detect_vcpkg, probe_vcpkg};

    // Resolve CUDA before Visual Studio so a machine with multiple Visual Studio releases prefers
    // a compiler version accepted by the installed CUDA toolkit's `host_config.h`, falling back to
    // any VS <= 2026 when CUDA is absent or only a newer VS is installed (`cuda-oxide` uses rustc).
    let cuda = vcv_rs::detect_cuda();
    let cuda_vs = cuda
        .as_ref()
        .and_then(|c| c.msvc)
        .and_then(|r| detect_vs_range(r.min_vs_year(), r.max_vs_year()));
    // Say when CUDA's own range found nothing: nvcc may later reject the fallback compiler,
    // and that error names `host_config.h`, not the Visual Studio choice made here.
    if cuda_vs.is_none()
        && let Some(c) = &cuda
    {
        eprintln!(
            "bootstrap: no Visual Studio in CUDA {}'s supported range; falling back to any VS <= 2026",
            c.version
        );
    }
    let vs = cuda_vs
        .or_else(|| detect_vs_range(None, Some(2026)))
        .ok_or("Visual Studio C++ tools not found (VS 2022/2026 required)")?;
    let sdk = detect_sdk().ok_or("Windows SDK not found")?;
    let ucrt = detect_ucrt().ok_or("Universal CRT not found")?;
    let mut assembled = build_env(&vs, Some(&sdk), Some(&ucrt), Arch::X64, Arch::X64);
    eprintln!(
        "bootstrap: VS {} / MSVC {} (vcv-rs)",
        vs.version, vs.tools_ver
    );

    if let Some(cuda) = cuda {
        add_cuda(&mut assembled, &cuda, Arch::X64);
        assembled
            .vars
            .insert("CUDA_TOOLKIT_PATH".into(), cuda.root.display().to_string());
        eprintln!("bootstrap: CUDA {} ({})", cuda.version, cuda.root.display());
    }
    if let Some(vcpkg) = detect_vcpkg().and_then(|root| probe_vcpkg(&root, Arch::X64)) {
        add_vcpkg(&mut assembled, &vcpkg);
    }

    let mut env = assembled.vars;
    for (name, paths) in [
        ("PATH", assembled.path),
        ("INCLUDE", assembled.include),
        ("LIB", assembled.lib),
        ("LIBPATH", assembled.libpath),
    ] {
        if !paths.is_empty() {
            let mut paths = paths;
            if let Some(inherited) = std::env::var_os(name) {
                paths.extend(std::env::split_paths(&inherited));
            }
            env.insert(
                name.into(),
                // A lossy conversion would hand cargo a silently corrupted search path.
                std::env::join_paths(paths)
                    .map_err(|e| format!("Invalid {name}: {e}"))?
                    .into_string()
                    .map_err(|p| format!("{name} is not valid UTF-8: {}", p.display()))?,
            );
        }
    }
    Ok(env)
}

#[cfg(not(windows))]
fn native_env() -> Result<BTreeMap<String, String>, String> {
    Ok(BTreeMap::new())
}

fn main() {
    match native_env().and_then(|env| serde_json::to_string(&env).map_err(|e| e.to_string())) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("bootstrap: {error}");
            std::process::exit(1);
        }
    }
}
