//! The WGSL twin: [`wgsl_source`] and the function-name registry [`TWIN_FUNCTIONS`].
//!
//! The library is the constants header generated from [`crate::consts`] followed by one WGSL
//! file per Rust module (`src/wgsl/{math,fresnel,microfacet,diffuse,sheen,surface,sampling,
//! transmission,sample}.wgsl`), each mirroring its Rust module statement for statement. It
//! declares only `const`, `struct` and `fn` items prefixed `MX_`/`SS_`, `Mx`/`Ss` or
//! `mx_`/`ss_`, has no bindings and no entry point, and never calls the builtin `distance`, so
//! a consumer shader can prepend it to its own module (e.g. `consts + wgsl_source() + body`).
//!
//! Naming rule for twins: a WGSL `mx_*` function has a Rust function of the same name (the
//! MaterialX name); a WGSL `ss_*` function has a Rust function named without the `ss_` prefix
//! (e.g. `ss_eval_light` is [`crate::eval_light`]). Tests check the registry both ways.

use std::sync::OnceLock;

/// Every WGSL function of the library, per module in [`WGSL_FILES`] order: each module owns
/// its sorted list (`<module>::WGSL_TWINS`), so a wave that fills a module (e.g. R2's
/// `sampling` and `sample`) registers its twins without editing this file. The
/// `twin_registry_matches_wgsl_and_rust` test checks every list against its WGSL file and the
/// Rust module, both ways.
pub const TWIN_FUNCTIONS: &[(&str, &[&str])] = &[
    ("math", crate::math::WGSL_TWINS),
    ("fresnel", crate::fresnel::WGSL_TWINS),
    ("microfacet", crate::microfacet::WGSL_TWINS),
    ("diffuse", crate::diffuse::WGSL_TWINS),
    ("sheen", crate::sheen::WGSL_TWINS),
    ("surface", crate::surface::WGSL_TWINS),
    ("sampling", crate::sampling::WGSL_TWINS),
    ("transmission", crate::transmission::WGSL_TWINS),
    ("sample", crate::sample::WGSL_TWINS),
];

/// The WGSL files in library order, each paired with the Rust module it mirrors.
pub const WGSL_FILES: &[(&str, &str)] = &[
    ("math", include_str!("wgsl/math.wgsl")),
    ("fresnel", include_str!("wgsl/fresnel.wgsl")),
    ("microfacet", include_str!("wgsl/microfacet.wgsl")),
    ("diffuse", include_str!("wgsl/diffuse.wgsl")),
    ("sheen", include_str!("wgsl/sheen.wgsl")),
    ("surface", include_str!("wgsl/surface.wgsl")),
    ("sampling", include_str!("wgsl/sampling.wgsl")),
    ("transmission", include_str!("wgsl/transmission.wgsl")),
    ("sample", include_str!("wgsl/sample.wgsl")),
];

/// The complete WGSL library: [`crate::consts::wgsl_header`] followed by [`WGSL_FILES`]. Built
/// once per process.
pub fn wgsl_source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| {
        let mut source = crate::consts::wgsl_header();
        for (_, text) in WGSL_FILES {
            source.push('\n');
            source.push_str(text);
        }
        source
    })
}
