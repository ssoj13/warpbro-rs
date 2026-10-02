use std::collections::BTreeMap;

#[cfg(windows)]
fn native_env() -> Result<BTreeMap<String, String>, String> {
    use vcv_rs::Arch;
    use vcv_rs::detect::{detect_sdk, detect_ucrt, detect_vs_range};
    use vcv_rs::env::{add_cuda, add_vcpkg, build_env, detect_vcpkg, probe_vcpkg};

    let vs = detect_vs_range(None, Some(2026))
        .ok_or("Visual Studio C++ tools not found (VS 2022/2026 required)")?;
    let sdk = detect_sdk().ok_or("Windows SDK not found")?;
    let ucrt = detect_ucrt().ok_or("Universal CRT not found")?;
    let mut assembled = build_env(&vs, Some(&sdk), Some(&ucrt), Arch::X64, Arch::X64);
    eprintln!(
        "bootstrap: VS {} / MSVC {} (vcv-rs)",
        vs.version, vs.tools_ver
    );

    if let Some(cuda) = vcv_rs::detect_cuda() {
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
                std::env::join_paths(paths)
                    .map_err(|e| format!("Invalid {name}: {e}"))?
                    .to_string_lossy()
                    .into_owned(),
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
    match native_env() {
        Ok(env) => println!("{}", serde_json::to_string(&env).unwrap()),
        Err(error) => {
            eprintln!("bootstrap: {error}");
            std::process::exit(1);
        }
    }
}
