# WarpBro PLAN

The one working document: rules, state, open work, findings. Keep it current: tick or drop done
items (details go to CHANGELOG.md), add new requests, keep open work sorted by priority. Chat in
Russian; code, comments and files in English. Reusable UI goes into egui-widgets-rs.

## Rules that bite
- Plan first; wait for approval before non-trivial changes.
- Never `git reset` / checkout-rollback; restore a file with `git show <rev>:<path> > <path>`.
- No `Co-Authored-By` lines. Commit only your own files: parallel sessions and WIP auto-commits
  share these repos. `cargo fmt` reformats foreign code: check `git diff -U0`.
- Systemic fixes, SSOT, no hacks, no silent fallbacks; verify everything; two reviewers before
  committing; tests sparingly (one targeted run at the end).
- Push when told. Own deps via SSH on `main`, third-party HTTPS, no `[patch]`.
- Build / test only through cuda-oxide: `cargo oxide build`, `cargo oxide test -- --release -- <filter>`.
- References: D:\Projects\vfx.ref.

## State (2026-10-06)
| Repo | HEAD | In flight |
|---|---|---|
| warpbro-rs | a707c9b | - |
| egui-widgets-rs | 55d1fbd | `feat/one-attr-editor`: one attribute-editor widget set |
| playa | ba49d14 | `feat/render-settings` (tier cache, background gate, templates, progressive viewer); `feat/nested-effects-menu` d1e8236 (awaits a visual check) |
| ofx-rs | bce0b90 | - |
| render-rs | 207473e | - |

## Open work (priority order)
1. **Attribute Editor parity** (in progress). The old attr-table rows (fill-width sliders, aligned
   value boxes, reset / copy / paste / random, segmented strips) were lost in 13b1caa / 9416673 /
   f2c73ab. egui-widgets-rs `feat/one-attr-editor` restores them in egui-attr-grid (attr-table,
   egui-attr, egui-vector deleted). Then: WarpBro (field defaults for reset, row actions on,
   Preferences onto the grid), Playa layer AE and ofx-rs `ofx-host-egui` onto the same grid.
   Screenshots before / after.
2. **Rough glass energy** (render-rs standard-surface-bsdf): no multiple-scattering compensation for
   dielectric transmission (furnace 0.81-0.93 inside at roughness 0.5, not re-verified). Plan:
   furnace grid baseline -> references (Turquin 2019, Cycles multiscatter GGX glass, OpenPBR) ->
   R/T compensation table, Rust+WGSL parity -> furnace tests (>= 0.99 rough, 1 smooth) -> render.
3. **Hybrid DE surface** (BUG1 leftovers): discontinuities at fold / escape transitions and the
   6-probe normal's p99 11 deg on converged hits; isolated cost of the 6-probe normal. Over-relaxed
   sphere tracing only with a safety check (Hybrid DEs are not Lipschitz).
4. **Render / Encode colour output**: one option set across EXR (ACEScg / ACES2065-1 / linear 709 /
   2020 + display-referred), PNG (SDR / HDR10 / HLG), MP4 (SDR [+ HDR10]). Plan first.
5. **In-process HDR video** via ffmpeg-rs (requirements: ffmpeg-rs/BUG3.md); then delete the ffmpeg
   stopgap (`PngVideo`, `encode_video`).
6. **GPU test isolation**: Vulkan Video export tests fail with "Posix(38)" when run beside the
   render_service / color GPU tests (6/6 on clean HEAD), pass alone; `vulkan_video()` serializes only
   within export.rs. Find the conflicting GPU use; serialize or fix.
7. `cuda_specialized_world_materials_preserve_radiance_and_affine_guides` (ignored): radiance
   max_absolute 5.92e-5 > 3e-5 since b5d562a - investigate.
8. io_service: four `let _ = done.send(..)` drop a result when the app has gone - log it.
9. Six soft slider ranges were picked without measurement (aperture, focus, orbit speed, thin film,
   sun angle, hit epsilon) - operator to check.
10. Dead per-object `render` fields on fractal nodes (only `/render/iterations` is per object) -
    drop when the object model is split (low).

## Gotchas
- egui: nested `ui.input` inside `hotkeys::active` deadlocks; read `active` first.
- Hotkeys need exact modifier matching. egui tests: send key releases (a second press is a repeat).
- render-rs: fetch before committing (local main was once behind origin).
- Playa: `cargo test -p playa-app <filter>` (not `--bin playa`); the OpenFX host needs
  `--features ofx`; debug builds overflow the main stack at start - run release.

## Key files
- WarpBro: src/world_ui.rs (AE), src/render.rs, src/gpu.rs, src/export.rs, src/render_service.rs,
  docs/color.md.
- render-rs BSDF: crates/render-engine-pt/standard-surface-bsdf/src/{microfacet.rs,sample.rs,wgsl/microfacet.wgsl}.

## Findings worth keeping (BUG1, 2026-10-05; scene: copper-turbine-kifs, frame 27)
- **Normals**: tiny gradients lost their direction (fixed 4f66e83: scale by the max component,
  retry smaller steps). The 4-probe tetrahedral normal was biased; now 6 central probes at eps/4.
  On converged hits (4096 steps, 19 311 stable neighbourhoods): old tetra median 0.45 deg / p90
  2.96 deg; 6-probe median 0.016 deg / p90 0.043 deg / p99 11.1 deg. Primary hits and their
  positions unchanged bit for bit. Regression: `cuda_curved_hybrid_normals_match_converged_field_gradient`.
- **Phantom hits**: at 256 steps 37% of primary hits were rays that ran out of steps (half were
  misses, half 0.30 units too deep, normals off by a median 48 deg). Fixed: `Outcome::Unresolved`,
  one policy per ray kind (camera: background; bounce: path ends; shadow: blocked; Direct: Keinert's
  half-pixel rule), unresolved marches counted in the status bar, presets at 4096 steps, step cap
  independent of the budget, glass interior probes a setting (256). Cost: primary rays 37.1 ms
  (256) vs 47.4 ms (8192); full 6-bounce 16 spp render 978 ms vs 958 ms. As Mandelbulber2 and
  Fragmentarium do (a ray out of steps is a miss). Ported to ofx-rs (bce0b90). Regression:
  `cuda_march_out_of_steps_is_a_reported_miss_not_a_phantom_hit`.
- **Glass energy**: clear glass lost energy at grazing angles (exact Fresnel for the delta lobe but
  the MaterialX albedo approximation for its throughput; IOR 1 gave reflection weight 0). Fixed in
  render-rs 207473e (`exact_interface`: albedo = exact F for smooth / IOR-1 interfaces, Rust+WGSL).
  Scene furnace 1.00003.
- **Not the cause**: material (matte shows the same facets), denoiser (raw accumulation unchanged),
  iterations (orbits escape after 4-9 of 12). bailout and step settings change the surface itself.
- Artifacts (not in the repo): ~/.warpbro/diagnostics/{surface-audit,surface-central}; the diag
  branch `diag/bug1-march` (worktree ../warpbro-diag).
- Sources: Hart, Sphere Tracing; Christensen (Fragmentarium) lighting and Mandelbulb DE;
  Mandelbulber2 shader_calculate_normals.cpp / compute_fractal.cpp / ray_recursion.cl; Keinert et
  al. 2014 Enhanced Sphere Tracing.
