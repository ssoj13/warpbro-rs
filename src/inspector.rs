//! Scene controls built from the shared egui-widgets-rs attribute editors.
use crate::palette::PaletteScheme;
use crate::scene::*;
use egui::{Color32, Ui};
use egui_attr_table::{AttrTable, attr_table};

pub fn section(ui: &mut Ui, label: &str, default_open: bool, body: impl FnOnce(&mut Ui)) {
    let id = ui.make_persistent_id(("scene-section", label));
    let open = ui
        .ctx()
        .data_mut(|d| d.get_persisted::<bool>(id))
        .unwrap_or(default_open);
    let response = egui_titlebar::CollapsingSection::new(label)
        .id_salt(id)
        .open(open)
        .tint(Color32::from_rgb(85, 140, 175), 0.12)
        .show(ui, body);
    ui.ctx()
        .data_mut(|d| d.insert_persisted(id, response.header.open));
}

/// Playa's vector grid inside the typed parameter sections; each grid has its own ID.
fn vector<const N: usize>(
    ui: &mut Ui,
    label: &str,
    value: &mut [f32; N],
    range: std::ops::RangeInclusive<f32>,
) {
    use egui_attr_grid::{AttrField, AttrGridState, AttrValue};
    ui.push_id(label, |ui| {
        let id = ui.make_persistent_id("vector-grid");
        let mut state = ui
            .ctx()
            .data_mut(|d| d.get_persisted::<AttrGridState>(id))
            .unwrap_or_default();
        if state.table.widths.is_empty() {
            state.table.widths = vec![egui_attr_table::label_width(ui.ctx())];
        }
        let av = if N == 3 {
            AttrValue::Vec3([value[0], value[1], value[2]])
        } else if N == 4 {
            AttrValue::Vec4([value[0], value[1], value[2], value[3]])
        } else {
            AttrValue::List(value.iter().copied().map(AttrValue::Float).collect())
        };
        let mut fields = [AttrField::new(label, av)];
        for (_, edited) in
            egui_attr_grid::render_grid(ui, &mut fields, &mut state, &Default::default())
        {
            let components: Vec<f32> = match edited {
                AttrValue::Vec3(v) => v.to_vec(),
                AttrValue::Vec4(v) => v.to_vec(),
                AttrValue::List(v) => v
                    .into_iter()
                    .filter_map(|v| {
                        if let AttrValue::Float(v) = v {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => Vec::new(),
            };
            for (slot, v) in value.iter_mut().zip(components) {
                *slot = v.clamp(*range.start(), *range.end());
            }
        }
        ui.ctx().data_mut(|d| d.insert_persisted(id, state));
    });
}

fn bulb(ui: &mut Ui, b: &mut Bulb) {
    attr_table(ui, |t| {
        t.row("Power").default(8.0).slider(&mut b.power, 2.0..=16.0);
        t.row("Bailout")
            .default(2.0)
            .slider(&mut b.bailout, 2.0..=16.0);
        t.row("θ scale")
            .default(1.0)
            .slider(&mut b.angle_scale[0], -4.0..=4.0);
        t.row("φ scale")
            .default(1.0)
            .slider(&mut b.angle_scale[1], -4.0..=4.0);
        t.row("θ phase °")
            .default(0.0)
            .slider(&mut b.angle_phase_degrees[0], -360.0..=360.0);
        t.row("φ phase °")
            .default(0.0)
            .slider(&mut b.angle_phase_degrees[1], -360.0..=360.0);
    });
    vector(
        ui,
        "Iteration rotation °",
        &mut b.rotation_degrees,
        -360.0..=360.0,
    );
}
fn mandelbox(ui: &mut Ui, b: &mut MandelBox) {
    attr_table(ui, |t| {
        t.row("Scale").default(2.0).slider(&mut b.scale, -4.0..=4.0);
        t.row("Min radius ratio")
            .default(0.5)
            .slider(&mut b.min_radius_ratio, 0.05..=1.0);
        t.row("Fixed radius")
            .default(1.0)
            .slider(&mut b.fixed_radius, 0.25..=2.0);
        t.row("Fold limit")
            .default(1.0)
            .slider(&mut b.fold_limit, 0.25..=2.0);
    });
    vector(
        ui,
        "Iteration rotation °",
        &mut b.rotation_degrees,
        -360.0..=360.0,
    );
}
fn kifs(ui: &mut Ui, k: &mut Kifs) {
    attr_table(ui, |t| {
        if t.row("Kind")
            .combo(
                &mut k.kind,
                &[
                    ("Tetrahedron", KifsKind::Tetrahedron),
                    ("Octahedron", KifsKind::Octahedron),
                    ("Menger", KifsKind::Menger),
                ],
            )
            .changed
        {
            k.scale = k.kind.preset_scale();
        }
        t.row("Scale")
            .default(k.kind.preset_scale())
            .slider(&mut k.scale, 1.2..=4.0);
    });
    vector(ui, "Centre offset", &mut k.offset, -2.0..=2.0);
    vector(
        ui,
        "Iteration rotation °",
        &mut k.rotation_degrees,
        -360.0..=360.0,
    );
}
pub fn formula(ui: &mut Ui, s: &mut Scene) {
    let mut code = s.formula.code();
    attr_table(ui, |t| {
        let items: Vec<_> = Formula::NAMES
            .iter()
            .enumerate()
            .map(|(i, n)| (*n, i as u32))
            .collect();
        t.row("Family").combo(&mut code, &items);
        t.row("Iterations")
            .default(12u32)
            .int_slider(&mut s.render.iterations, 1..=64);
    });
    if code != s.formula.code() {
        let fresh = Scene::preset(code);
        s.formula = fresh.formula;
        s.render.iterations = fresh.render.iterations;
        s.render.max_steps = fresh.render.max_steps;
        s.render.hit_epsilon = fresh.render.hit_epsilon;
        s.julia = None;
    }
    match &mut s.formula {
        Formula::Mandelbulb(b) => bulb(ui, b),
        Formula::Mandelbox(b) => mandelbox(ui, b),
        Formula::QuaternionJulia(q) => {
            vector(ui, "Constant", &mut q.constant, -1.5..=1.5);
            attr_table(ui, |t| {
                t.row("Slice W")
                    .default(0.0)
                    .slider(&mut q.slice_w, -1.5..=1.5);
                t.row("Bailout")
                    .default(4.0)
                    .slider(&mut q.bailout, 2.0..=16.0);
            });
            vector(ui, "4D rotation °", &mut q.rotation_degrees, -360.0..=360.0);
        }
        Formula::Kifs(k) => kifs(ui, k),
        Formula::Kleinian(k) => attr_table(ui, |t| {
            t.row("A")
                .default(Kleinian::PRESET.a)
                .slider(&mut k.a, 1.0..=2.2);
            t.row("B")
                .default(Kleinian::PRESET.b)
                .slider(&mut k.b, -1.0..=1.0);
            t.row("Ball bound")
                .tip("0 disables the ball bound")
                .default(2.0)
                .slider(&mut k.bound_radius, 0.0..=4.0);
        }),
        Formula::PseudoKleinian(k) => {
            vector(ui, "Box size", &mut k.box_size, 0.1..=2.0);
            vector(ui, "C", &mut k.c, -2.0..=2.0);
            vector(ui, "Offset", &mut k.offset, -2.0..=2.0);
            attr_table(ui, |t| {
                t.row("Size").default(1.0).slider(&mut k.size, 0.1..=2.0);
                t.row("Thickness")
                    .default(0.01)
                    .slider(&mut k.thickness, 0.0..=0.2);
                t.row("Ball bound")
                    .tip("0 disables the ball bound")
                    .default(0.0)
                    .slider(&mut k.bound_radius, 0.0..=4.0);
            });
        }
        Formula::Apollonian(a) => attr_table(ui, |t| {
            t.row("Scale").default(1.3).slider(&mut a.scale, 1.0..=2.0);
            t.row("Ball bound")
                .default(1.5)
                .slider(&mut a.bound_radius, 0.0..=4.0);
        }),
        Formula::Hybrid(h) => {
            attr_table(ui, |t| {
                let items = [
                    ("Off", HybridStep::Off),
                    ("Mandelbulb", HybridStep::Mandelbulb),
                    ("Mandelbox", HybridStep::Mandelbox),
                    ("KIFS fold", HybridStep::KifsFold),
                    ("Inversion", HybridStep::Inversion),
                ];
                for (i, step) in h.steps.iter_mut().enumerate() {
                    t.row(&format!("Step {}", i + 1)).combo(step, &items);
                }
                t.row("Bailout")
                    .default(8.0)
                    .slider(&mut h.bailout, 2.0..=16.0);
                t.row("Inversion scale")
                    .default(1.3)
                    .slider(&mut h.apollonian_scale, 1.0..=2.0);
            });
            section(ui, "Bulb step", false, |ui| bulb(ui, &mut h.bulb));
            section(ui, "Box step", false, |ui| mandelbox(ui, &mut h.mandelbox));
            section(ui, "KIFS step", false, |ui| kifs(ui, &mut h.kifs));
        }
    }
    if s.formula.supports_julia() {
        let mut on = s.julia.is_some();
        attr_table(ui, |t| {
            t.row("Julia mode").default(false).checkbox(&mut on);
        });
        if on {
            s.julia.get_or_insert([0.35, 0.35, -0.4]);
        } else {
            s.julia = None;
        }
        if let Some(c) = &mut s.julia {
            vector(ui, "Julia C", c, -2.0..=2.0);
        }
    }
    section(ui, "Object transform", false, |ui| {
        vector(ui, "Offset", &mut s.object.offset, -4.0..=4.0);
        vector(
            ui,
            "Rotation °",
            &mut s.object.rotation_degrees,
            -360.0..=360.0,
        );
        attr_table(ui, |t| {
            t.row("Scale")
                .default(1.0)
                .slider(&mut s.object.scale, 0.1..=4.0);
        });
    });
}
pub fn camera(ui: &mut Ui, s: &mut Scene) {
    let c = &mut s.camera;
    let defaults = Scene::preset(s.formula.code()).camera;
    attr_table(ui, |t| {
        t.row("Yaw °")
            .default(defaults.yaw_degrees)
            .slider(&mut c.yaw_degrees, -180.0..=180.0);
        t.row("Pitch °").default(defaults.pitch_degrees).slider(
            &mut c.pitch_degrees,
            if c.free_flight {
                -90.0..=90.0
            } else {
                -89.0..=89.0
            },
        );
        if c.free_flight {
            t.row("Roll °")
                .default(0.0)
                .slider(&mut c.roll_degrees, -180.0..=180.0);
        }
        t.row("Distance")
            .tip("Distance in formula framing radii")
            .default(defaults.distance)
            .slider_log(&mut c.distance, 0.05..=12.0);
        t.row("FOV °")
            .default(defaults.fov_y_degrees)
            .slider(&mut c.fov_y_degrees, 5.0..=120.0);
        t.row("Aperture")
            .default(0.0)
            .slider(&mut c.aperture, 0.0..=1.0);
        if c.aperture > 0.0 {
            t.row("Focus distance")
                .tip("0 focuses on the orbit target")
                .default(0.0)
                .slider(&mut c.focus_distance, 0.0..=12.0);
        }
    });
    vector(ui, "Target", &mut c.target, -10.0..=10.0);
    ui.small("LMB orbit · MMB pan · wheel zoom · RMB fly · ` / ~ switches flight mode");
    ui.small(if c.free_flight {
        "Free: WASD · R/C up/down · Q/E roll · Shift boost"
    } else {
        "Horizon: WASD · R/C up/down · Q/E enables roll · Shift boost"
    });
}
pub fn lighting(ui: &mut Ui, l: &mut Lighting) {
    let d = Lighting::default();
    attr_table(ui, |t| {
        t.row("Sun azimuth °")
            .default(d.sun_azimuth)
            .slider(&mut l.sun_azimuth, -180.0..=180.0);
        t.row("Sun elevation °")
            .default(d.sun_elevation)
            .slider(&mut l.sun_elevation, -30.0..=90.0);
        t.row("Sun colour")
            .default(d.sun_color)
            .color3(&mut l.sun_color);
        t.row("Sun intensity")
            .default(d.sun_intensity)
            .slider(&mut l.sun_intensity, 0.0..=16.0);
        t.row("Sun angle °")
            .default(d.sun_angle)
            .slider_log(&mut l.sun_angle, 0.05..=30.0);
        t.row("Sky intensity")
            .default(d.sky_intensity)
            .slider(&mut l.sky_intensity, 0.0..=16.0);
        t.row("Sky horizon")
            .default(d.sky_horizon)
            .color3(&mut l.sky_horizon);
        t.row("Sky zenith")
            .default(d.sky_zenith)
            .color3(&mut l.sky_zenith);
        t.row("Sky visible")
            .default(true)
            .checkbox(&mut l.background);
    });
}
fn surface_rows(t: &mut AttrTable<'_>, m: &mut Material) {
    let d = Material::default();
    t.row("Base colour")
        .default(d.base_color)
        .color3(&mut m.base_color);
    t.row("Base tint")
        .default(d.base_tint)
        .color3(&mut m.base_tint);
    t.row("Base weight")
        .default(d.base)
        .slider(&mut m.base, 0.0..=1.0);
    t.row("Metalness")
        .default(d.metalness)
        .slider(&mut m.metalness, 0.0..=1.0);
    t.row("Specular colour")
        .default(d.specular_color)
        .color3(&mut m.specular_color);
    t.row("Specular weight")
        .default(d.specular)
        .slider(&mut m.specular, 0.0..=1.0);
    t.row("Roughness")
        .default(d.specular_roughness)
        .slider(&mut m.specular_roughness, 0.0..=1.0);
    t.row("IOR")
        .default(d.specular_ior)
        .slider(&mut m.specular_ior, 1.0..=3.0);
    t.row("Emission colour")
        .default(d.emission_color)
        .color3(&mut m.emission_color);
    t.row("Emission weight")
        .default(d.emission)
        .slider(&mut m.emission, 0.0..=4.0);
    if m.model == MaterialModel::StandardSurface {
        t.row("Diffuse roughness")
            .default(d.diffuse_roughness)
            .slider(&mut m.diffuse_roughness, 0.0..=1.0);
        t.row("Anisotropy")
            .default(d.specular_anisotropy)
            .slider(&mut m.specular_anisotropy, 0.0..=1.0);
        t.row("Aniso rotation")
            .default(d.specular_rotation)
            .slider(&mut m.specular_rotation, 0.0..=1.0);
        t.row("Coat colour")
            .default(d.coat_color)
            .color3(&mut m.coat_color);
        t.row("Coat weight")
            .default(d.coat)
            .slider(&mut m.coat, 0.0..=1.0);
        t.row("Coat roughness")
            .default(d.coat_roughness)
            .slider(&mut m.coat_roughness, 0.0..=1.0);
        t.row("Coat IOR")
            .default(d.coat_ior)
            .slider(&mut m.coat_ior, 1.0..=3.0);
        t.row("Coat affects colour")
            .default(d.coat_affect_color)
            .slider(&mut m.coat_affect_color, 0.0..=1.0);
        t.row("Coat affects roughness")
            .default(d.coat_affect_roughness)
            .slider(&mut m.coat_affect_roughness, 0.0..=1.0);
        t.row("Sheen colour")
            .default(d.sheen_color)
            .color3(&mut m.sheen_color);
        t.row("Sheen weight")
            .default(d.sheen)
            .slider(&mut m.sheen, 0.0..=1.0);
        t.row("Sheen roughness")
            .default(d.sheen_roughness)
            .slider(&mut m.sheen_roughness, 0.0..=1.0);
        t.row("Thin film nm")
            .default(d.thin_film_thickness)
            .slider(&mut m.thin_film_thickness, 0.0..=2000.0);
        t.row("Thin film IOR")
            .default(d.thin_film_ior)
            .slider(&mut m.thin_film_ior, 1.0..=3.0);
    }
}
pub fn material(ui: &mut Ui, m: &mut Material) {
    attr_table(ui, |t| {
        t.row("Model").combo(
            &mut m.model,
            &[
                ("Fast", MaterialModel::Fast),
                ("Standard Surface", MaterialModel::StandardSurface),
            ],
        );
        let mut selected = m.preset.clone().unwrap_or_else(|| "Custom".into());
        let mut items = vec![("Custom", "Custom".to_string())];
        items.extend(
            crate::materials::PRESETS
                .iter()
                .map(|p| (p.name(), p.name().to_string())),
        );
        if t.row("Library").combo(&mut selected, &items).changed {
            if let Some(p) = crate::materials::PRESETS
                .iter()
                .find(|p| p.name() == selected)
            {
                p.apply(m);
            } else {
                m.preset = None;
            }
        }
        t.row("Colour source").combo(
            &mut m.color_source,
            &[
                ("Palette", ColorSource::Palette),
                ("Material", ColorSource::Material),
            ],
        );
        surface_rows(t, m);
        let mut facing = m.facing.is_some();
        if t.row("Facing mix")
            .default(false)
            .checkbox(&mut facing)
            .changed
        {
            m.facing = facing.then_some(Facing {
                color: [0.18, 0.10, 0.65],
                roughness: 0.22,
                metallic: 0.0,
                exponent: 3.0,
            });
        }
        if let Some(f) = &mut m.facing {
            t.row("Grazing colour")
                .default([0.18, 0.10, 0.65])
                .color3(&mut f.color);
            t.row("Facing exponent")
                .default(3.0)
                .slider(&mut f.exponent, 0.5..=8.0);
            t.row("Grazing roughness")
                .default(0.22)
                .slider(&mut f.roughness, 0.0..=1.0);
            t.row("Grazing metallic")
                .default(0.0)
                .slider(&mut f.metallic, 0.0..=1.0);
        }
    });
}
pub fn palette(ui: &mut Ui, s: &mut Scene) {
    attr_table(ui, |t| {
        let items: Vec<_> = PaletteScheme::ALL.iter().map(|p| (p.label(), *p)).collect();
        t.row("Palette").combo(&mut s.palette, &items);
        t.row("Colouring").combo(
            &mut s.coloring,
            &[
                ("Radius", Coloring::Radius),
                ("Trap origin", Coloring::TrapOrigin),
                ("Trap plane", Coloring::TrapPlane),
                ("Trap point", Coloring::TrapPoint),
            ],
        );
        if s.coloring != Coloring::Radius {
            t.row("Trap scale")
                .default(1.0)
                .slider_log(&mut s.trap_scale, 0.05..=20.0);
            if s.coloring == Coloring::TrapPlane {
                t.row("Plane normal")
                    .combo(&mut s.trap_axis, &[("X", 0), ("Y", 1), ("Z", 2)]);
            }
        }
    });
    palette_strip(ui, s.palette);
    if matches!(s.coloring, Coloring::TrapPlane | Coloring::TrapPoint) {
        vector(ui, "Trap point", &mut s.trap_point, -4.0..=4.0);
    }
}
fn palette_strip(ui: &mut Ui, scheme: PaletteScheme) {
    let lut = crate::palette::build_lut(scheme);
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 12.0), egui::Sense::hover());
    for i in 0..64 {
        let c = lut[i * (lut.len() - 2) / 63];
        let to8 = |v: f32| (crate::color::oetf(v.clamp(0.0, 1.0)) * 255.0) as u8;
        let x0 = rect.left() + rect.width() * i as f32 / 64.0;
        let x1 = rect.left() + rect.width() * (i + 1) as f32 / 64.0;
        ui.painter().rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom())),
            0.0,
            Color32::from_rgb(to8(c[0]), to8(c[1]), to8(c[2])),
        );
    }
}
pub fn render(ui: &mut Ui, r: &mut Render, target: &mut u32, resolution: &mut f32) {
    attr_table(ui, |t| {
        t.row("Bounces")
            .default(6u32)
            .int_slider(&mut r.max_bounces, 0..=16);
        t.row("March steps")
            .default(256u32)
            .int_slider(&mut r.max_steps, 32..=2048);
        t.row("Hit epsilon")
            .default(0.001)
            .slider_log(&mut r.hit_epsilon, 0.0001..=0.01);
        t.row("Step factor")
            .default(0.85)
            .slider(&mut r.step_factor, 0.3..=1.0);
        t.row("Exposure EV")
            .default(0.0)
            .slider(&mut r.exposure_stops, -6.0..=6.0);
        t.row("Saturation")
            .default(1.0)
            .slider(&mut r.saturation, 0.0..=2.0);
        t.row("Legacy Reinhard")
            .default(false)
            .checkbox(&mut r.reinhard);
        t.row("OIDN denoise")
            .default(true)
            .checkbox(&mut r.denoise.enabled);
        if r.denoise.enabled {
            t.row("Denoise every N samples")
                .tip("0 runs only the final pass; changing this preserves accumulated samples")
                .default(128u32)
                .int_slider(&mut r.denoise.interval, 0..=65536);
            let modes: Vec<_> = crate::denoise::Mode::ALL
                .iter()
                .map(|v| (v.label(), *v))
                .collect();
            t.row("Denoise guides").combo(&mut r.denoise.mode, &modes);
            let qualities: Vec<_> = crate::denoise::Quality::ALL
                .iter()
                .map(|v| (v.label(), *v))
                .collect();
            t.row("Denoise quality")
                .combo(&mut r.denoise.quality, &qualities);
        }
        t.row("Target samples")
            .default(1024u32)
            .int_slider(target, 1..=65536);
        t.row("Viewport scale")
            .default(1.0)
            .slider(resolution, 0.25..=2.0);
    });
}
