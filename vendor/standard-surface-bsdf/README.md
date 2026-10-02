# standard-surface-bsdf

Autodesk Standard Surface BSDF ported from MaterialX `v1.39.5-22-g47cecce6`, as a Rust `f32`
CPU twin and a WGSL library with the same functions in the same statement order. No GPU
dependency: the only dependency is the pure-Rust `libm` (platform-independent CPU bits).

Consumers: the render-rs raster pipeline (`standard-surface`), the PT megakernel
(`pt-megakernel`: eval/sample/pdf of every lobe and the counter RNG), the ofx-rs 3D fractal
(Direct and Path Tracing modes).

## Scope

`standard_surface` with `opacity = 1`, `thin_walled = false`, in which the graph reduces
exactly to

```text
coat_layer = coat over (clamp(mix(1, coat_color, coat)) *
             mix(specular over mix(sheen over mix(diffuse, subsurface, subsurface),
                                   transmission, transmission), metal, metalness))
```

with MaterialX's energy-compensated GGX lobes (dielectric specular with Airy thin film,
artist-friendly conductor, coat), Imageworks sheen, qualitative Oren-Nayar diffuse,
anisotropy and the coat-tinted emission EDF. Albedos use MaterialX's analytic fits (the
codegen default); Airy uses 2 iterations. With `transmission = 0` and `subsurface = 0` (the
defaults) every entry point returns the reflection-only model's bits, CPU and WGSL (the GPU
parity test hashes every output of its 142 592 subsurface-free cases, CPU `25d90b90737050a1`,
GPU `896a71eb21ec1c43`, unchanged by the subsurface closure).

**Subsurface** (`subsurface`, `subsurface_color`, `subsurface_radius`, `subsurface_scale`) is
MaterialX's own `subsurface_bsdf` (`pbrlib/genglsl/mx_subsurface_bsdf.glsl`): the Burley
diffusion profile integrated over a circle of the surface's curvature radius
(`lib/mx_microfacet_diffuse.glsl:158-199`), `response = color D(theta_l) / pi`, mixed with the
Oren-Nayar diffuse by `subsurface` (`subsurface_mix`, `standard_surface.mtlx:252-256`):
- MaterialX estimates the curvature in screen space, `length(fwidth(N)) / length(fwidth(P))`;
  here it is `ShadingFrame::curvature`, which a rasteriser fills with that expression and a path
  tracer with its geometric counterpart (0 = flat, radius 100);
- the hemisphere contract applies: MaterialX's approximation also responds to light below the
  horizon (the wrapped part of the integral; its full-sphere total exceeds 1, e.g. 1.10 at
  curvature 1, mfp 1); here only the upper hemisphere responds, with albedo exactly 1 on a flat
  surface and at most 1 elsewhere (0.936 at curvature 1, mfp 1);
- the environment (indirect) part is MaterialX's "simple indirect diffuse" `irradiance * color`;
- the diffuse lobe carries it (both are cosine-sampled), so `SS_LOBE_COUNT` is unchanged;
- `subsurface_anisotropy` is validated but, as in MaterialX's closure body, not read.

**Transmission** (`transmission`, `transmission_color`, `transmission_extra_roughness`) is a
path-traced rough dielectric BTDF (Walter et al. 2007; `src/transmission.rs`), because
MaterialX's own `T` mode is a prefiltered-environment lookup with no BTDF to port:
- outside, it is Fresnel-free under the dielectric specular's throughput (MaterialX's layering;
  the specular already removed the reflected energy); inside (`ShadingFrame::inside`, a back face
  of a transmissive surface) the surface is the bare interface with Walter's `1 - F(wo.m)`, so
  total internal reflection moves energy into the reflection;
- adjoint (flux) form: albedo at most 1, `f(wo, wi) / eta_wi^2 = f(wi, wo) / eta_wo^2`; the
  radiance factor `(eta_wo / eta_wi)^2` is omitted (it cancels on closed objects);
- no multiple-scattering compensation (MaterialX has none for transmission): rough glass loses
  the single-scatter share, measured white furnace 0.75 outside at grazing and 0.49 inside at
  roughness 1;
- `eval_environment` adds `Environment::transmission_radiance` times the layered tint, without
  MaterialX's extra `1 - FG` (`mx_environment_prefilter.glsl:15-16`), which would count the
  Fresnel twice;
- not inputs: `transmission_depth`, `_scatter`, `_scatter_anisotropy`, `_dispersion` (the interior
  medium belongs to the integrator) and `thin_walled`.

Lobe arrays have `SS_LOBE_COUNT` entries (`LobeWeights`, WGSL `array<f32, SS_LOBE_COUNT>`); callers
never hard-code the count.

## API

```rust
use standard_surface_bsdf::{SurfaceInputs, ShadingFrame, Environment, eval_light, eval_environment};

let inputs = SurfaceInputs::MATERIALX_DEFAULT;
inputs.validate()?; // the one fallible entry point; evaluation never clamps
let frame = ShadingFrame { n: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0], inside: false, curvature: 0.0 };
let lobes = eval_light(&inputs, &frame, wo, wi); // f(wo, wi) |cos|: base + specular + transmission
```

- `eval_light` / `eval_light_disc` (sun disc of angular radius `half_angle`, NDF widened),
  `eval_environment` (prefiltered radiance and irradiance supplied by the caller),
  `eval_emission`, `lobe_weights`, `dominant_dir`.
- `surface::Lobes { base, specular, transmission }` partitions the layered value.
- `sampling` / `sample`: counter RNG, samplers, `sample`/`pdf` (wave R2 of the plan).
- `transmission`: Snell `refract`, the BTDF `ggx_transmission` and its density
  `ggx_transmission_pdf`.
- `wgsl_source()`: the WGSL library (constants header generated from `consts.rs` + one file
  per module); prefixes `mx_`/`ss_`, `Mx`/`Ss`, `MX_`/`SS_`; no bindings, no entry point.

Contract beyond MaterialX: `eval_light` is exactly zero when `n.wo <= 0`, and when `n.wi <= 0`
except for the transmission lobe (so an opaque material is still zero below the horizon); a
GGX lobe with `max(alpha) < SS_ALPHA_MIN` (1e-3) is a delta lobe with zero response in
`eval_light` but unchanged throughput and selection weight.

Known MaterialX properties kept as-is: the layered model is not reciprocal (throughputs depend
on `NdotV` only); the dielectric throughput ignores the thin film, so a white furnace with a
film can exceed 1 (measured 1.33 at 400 nm); the GGX albedo fit overshoots 1 by up to 3% at
grazing smooth angles.

`SurfaceInputs::thin_film_energy` is an explicit option (never an environment variable):
`ThinFilmEnergy::MaterialX` (default, MaterialX bits) or `ThinFilmEnergy::Conserving`, which
takes the dielectric throughput from the film-aware albedo `E(fd)` and weights the Airy mirror
term of `E(fd)` by the GGX single-scatter energy (MaterialX blends the unshadowed mirror
reflectance into a rough lobe, which the energy compensation then amplifies). Dielectric and
conductor stacks with a film then keep the white furnace at 1 (MaterialX: 1.34 and 1.13).

## Verify

```
cargo test -p standard-surface-bsdf
cargo clippy -p standard-surface-bsdf --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p standard-surface-bsdf --no-deps
```

## Licence

Apache-2.0, a derivative of MaterialX (Copyright Contributors to the MaterialX Project); see
`LICENSE` and `NOTICE`.
