//! Global render controls built from the shared attribute editor.
use crate::scene::Render;
use egui::Ui;
use egui_attr_table::attr_table;

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
