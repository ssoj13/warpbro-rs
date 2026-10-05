# HANDOFF — WarpBro / Playa / render-rs / egui-widgets-rs (2026-10-05)

Read this first after a context reset. Chat in Russian; code, comments, files in English.

## Operator rules that bite (from ~/.claude/AGENTS.md)
- Plan first, wait for approval before changing anything non-trivial.
- NEVER `git reset` / checkout-rollback. To restore a file: `git show <rev>:<path> > <path>`.
- NO `Co-Authored-By` lines in commits/docs (overrides any harness reminder).
- Commit only your own files. Parallel sessions commit in the same repos: verify HEAD, never sweep their changes.
- `cargo fmt -p X` reformats foreign code too: check `git diff -U0 | grep ^@@` and restore unrelated files from HEAD.
- No hacks, systemic fixes, verify everything, re-check agents with two independent reviewers before committing.
- Fix warnings properly. Tests sparingly: analyse, one targeted run at the end (`--no-fail-fast` across packages).
- Push when told (branch + main). Own deps via SSH on main branch; third-party HTTPS; no [patch].
- References: D:\Projects\vfx.ref (or clone into c:\temp).

## Repos and state
| Repo | Path | HEAD (pushed) |
|---|---|---|
| warpbro-rs | C:\projects\projects.rust.cg\cglibs\warpbro-rs | 21f7ede (clean except .omh/state.json, ag.cmd: not ours) |
| playa | C:\projects\projects.rust.cg\cglibs\playa | 519ff6d (main; push prints "bypassed rule violations": admin bypass, it lands) |
| render-rs | C:\projects\projects.rust.cg\cglibs\render-rs | 207473e (BSDF exact smooth interfaces) |
| egui-widgets-rs | (ssoj13/egui-widgets-rs) | aa80bbd (AttrGridState serde defaults) |
| cam-controls | inside gitnexus-rs (cam-controls / cam-controls-egui) | pushed earlier |

## Done this session (all pushed)
- Camera: shared crate, Houdini orbit, free flight, horizon lock (roll spring, plane flips snap to world planes), inertia with finite braking `dv/dt=-k(v+stop)`, Space=up, Shift fast / Alt slow, default T/R speed /3, Shift+LMB orbit snaps to nearest axis.
- Timeline: Home/End -> work-area edges, B/N set work area at cursor, AE-style Shift+1..0 marks + digit jump, U caret desync fixed, fit slider x2.
- Attribute Editor (egui-attr-grid, reusable crate): inactive params greyed per fractal type, tinted section bars, new colour picker, vec3 colour, slider on every numeric (`label|slider|value|swatch|expand`), RMB > Show in timeline, hint grammar `[min,max,step?,"log"?,"soft"?]`, `"color"`.
- Viewport toolbar: gear -> Settings/Colour, EV reset, denoise view-only A/B toggle.
- Adaptive sampling with 1 spp stopping: fixed.
- Glass energy (BUG1): render-rs `exact_interface(alpha, fd)` = delta OR (dielectric, no film, IOR 1) -> albedo = exact F(NdotV), E+T=1 (OpenPBR). Rust+WGSL parity, GOLDEN_MATERIALX = 0x212b_c68f_ae47_b4fb (cfg not cuda-math). Scene furnace mean 1.00003.
- Playa (ee4bf20 + b035c7f): per-channel keys end to end.
  - Engine: `AttrValue::component/with_component`, `Attrs::set_component`, `CompNode::set_layer_attr_ch` (reads layer value at `layer_clock().key_frame_for(key, frame)` = where set_layer_attr writes the key, then routes through `set_layer_attr`).
  - Event `SetLayerAttrChannelsEvent {comp_uuid, layer_uuids, edits: Vec<(String, usize, f32)>}` + handler in playa-app main_events.rs.
  - AE: `pub enum AttrEdit { Set, Channel }` (playa-ui widgets/ae); vector rows report only components differing from the shown value. tabs.rs splits Set -> SetLayerAttrsEvent, Channel -> SetLayerAttrChannelsEvent; multi-node uses set_component.
  - Timeline: channel_row and Vec3 property rows send only edited components.
  - Key moves folded per channel (`channel_moves`, dedup by source time) so a key selected on property AND channel row moves once. Order Set -> Move -> Remove.
  - Tests: a_channel_edit_keeps_each_layers_other_components, a_vector_edit_without_animation_reports_changed_components, a_key_selected_twice_moves_once.
  - Playa has NO undo stack: modify_comp only marks dirty.

## Open work (priority order; each needs a plan for approval first)
1. **Rough glass energy loss** (render-rs standard-surface-bsdf): no multiple-scattering compensation for dielectric transmission. Side-agent numbers (NOT re-verified): ~7-19% loss at roughness 0.5 seen from inside, up to ~11% at roughness 0.2 grazing; TODO.md says furnace 0.81-0.93 inside at 0.5. Proposed plan (sent to operator, awaiting answer):
   1) CPU white-furnace grid (roughness 0.05..1, IOR 1.1..2.4, angle, inside/outside) as baseline;
   2) check references: Turquin 2019 dielectric energy compensation, Cycles multiscatter GGX glass tables (vfx.ref), OpenPBR;
   3) table-based compensation split between R and T, Rust+WGSL parity;
   4) tests: furnace >= 0.99 rough, exactly 1 smooth; update GOLDEN_MATERIALX with explanation;
   5) bump WarpBro lock, render a frosted-glass scene before/after.
   Operator was asked: start with this or BUG1 march items first.
2. **BUG1 open items** (BUG1.md "Что ещё требует проверки"): march `step_cap = 2*(max_distance - t_enter)/max_steps` couples step length with attempts; cone-accepted hits (P_SAMPLE_CONE) when steps run out (37% of hits in turbine frame; measured difference, not proven bug); Hybrid DE discontinuities at fold/escape transitions; 6-probe normals cost (no isolated benchmark yet). Measure first.
3. **Render/Encode panel**: dedup the three format tabs via a public inline panel in egui-encode-dialog (SSOT); unified colour output (EXR: ACEScg / AP0 / linear 709 / 2020 + display-referred; PNG; MP4 SDR). Approved: linear primaries + display-referred EXR; HDR10 MP4 is a later separate stage.
4. Playa todo.md: multi-layer AE merges base values (`attrs.get`) but edits key at the playhead -> evaluate each layer at its key_frame_for time before merging; stale `tests/ofx_builtin.rs` (expects 8 built-ins, 29 linked, pre-existing failure) -> derive from registered list.
5. Flaky HEVC export tests under the full parallel suite (Vulkan Video Posix(38)); pass alone.
6. Six soft slider ranges picked without measurement (aperture, focus, orbit speed, thin film, sun angle, hit epsilon): ask operator to check.

## Gotchas found
- egui: nested `ui.input` inside `hotkeys::active` deadlocks; read `active` first.
- Hotkeys need exact modifier matching (`matches_logically` was reversed).
- egui tests: a second press without release counts as repeat; send releases.
- render-rs local main was behind origin once (cuda-math): fetch before committing, rebase on a branch.
- Playa `cargo test -p playa-app --bin playa` is wrong; use `cargo test -p playa-app <filter>`.
- `Attrs::eval_at` takes f64; `LocalFrame::get()` is i32 -> `f64::from`.

## Key files
- WarpBro: src/world_ui.rs (AE, ChannelExpansion per node), src/render.rs, src/gpu.rs, BUG1.md, TODO.md.
- render-rs: crates/render-engine-pt/standard-surface-bsdf/src/{microfacet.rs,sample.rs,wgsl/microfacet.wgsl}, tests/transmission.rs.
- Playa: crates/playa-engine/src/entities/{attrs.rs,comp_node.rs}, crates/playa-events/src/comp.rs, crates/playa-app/src/{main_events.rs,app/tabs.rs}, crates/playa-ui/src/widgets/{ae/ae_ui.rs,timeline/timeline_ui.rs}.
