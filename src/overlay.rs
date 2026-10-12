//! Viewport extras: wire shapes of what the viewport shows but does not trace (masks, group
//! locators, the cameras other than the active one, the sun) and the cubes of a Bounds display,
//! in world space. `App::draw_overlays` projects them all through the viewport camera; which
//! nodes appear is `WorldDocument::shows` under the ViewportSettings kind filter, as in the
//! render.
use crate::render_profiles::ViewportPolicy;
use crate::scene::{Bounds, MaskShape};
use crate::world::{NodeId, WorldDocument, WorldKind};
use glam::{Mat4, Vec3};

/// How a wire is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    Plain,
    /// Its node is selected.
    Selected,
    /// An inverted mask: what it cuts is inside.
    Cut,
}

/// A polyline in world space.
#[derive(Clone, Debug, PartialEq)]
pub struct Wire {
    pub points: Vec<Vec3>,
    pub style: Style,
}

/// Segments of a drawn circle.
const CIRCLE: usize = 48;

fn segment(m: Mat4, a: Vec3, b: Vec3, style: Style) -> Wire {
    Wire {
        points: vec![m.transform_point3(a), m.transform_point3(b)],
        style,
    }
}

fn circle(m: Mat4, u: Vec3, v: Vec3, radius: f32, style: Style) -> Wire {
    Wire {
        points: (0..=CIRCLE)
            .map(|i| {
                let t = i as f32 / CIRCLE as f32 * std::f32::consts::TAU;
                m.transform_point3((u * t.cos() + v * t.sin()) * radius)
            })
            .collect(),
        style,
    }
}

/// The 12 edges of the box of half extents `half` around the local origin.
fn cube(m: Mat4, half: Vec3, style: Style) -> impl Iterator<Item = Wire> {
    let corner = move |i: usize| {
        Vec3::new(
            if i & 1 != 0 { half.x } else { -half.x },
            if i & 2 != 0 { half.y } else { -half.y },
            if i & 4 != 0 { half.z } else { -half.z },
        )
    };
    (0..8).flat_map(move |i| {
        [1, 2, 4]
            .into_iter()
            .filter(move |bit| i & bit == 0)
            .map(move |bit| segment(m, corner(i), corner(i | bit), style))
    })
}

/// A mask's shape: a sphere's three great circles, a box's edges, or the grid of a Cells mask's
/// 3x3x3 cells around its origin (which cells it removes is its seed's, not drawn).
fn mask(m: Mat4, mask: &crate::scene::Mask, style: Style) -> Vec<Wire> {
    let s = Vec3::from_array(mask.size);
    match mask.shape {
        MaskShape::Sphere => [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)]
            .into_iter()
            .map(|(u, v)| circle(m, u, v, s.x, style))
            .collect(),
        MaskShape::Box => cube(m, s, style).collect(),
        MaskShape::Cells => {
            let edge = s.x;
            let at = |i: usize| (i as f32 - 1.5) * edge;
            let mut wires = Vec::new();
            for (a, b) in (0..4).flat_map(|a| (0..4).map(move |b| (a, b))) {
                let (p, q) = (at(a), at(b));
                let (lo, hi) = (at(0), at(3));
                wires.push(segment(m, Vec3::new(lo, p, q), Vec3::new(hi, p, q), style));
                wires.push(segment(m, Vec3::new(p, lo, q), Vec3::new(p, hi, q), style));
                wires.push(segment(m, Vec3::new(p, q, lo), Vec3::new(p, q, hi), style));
            }
            wires
        }
    }
}

/// A group's locator: its three axes.
fn locator(m: Mat4, style: Style) -> Vec<Wire> {
    const SIZE: f32 = 0.5;
    [Vec3::X, Vec3::Y, Vec3::Z]
        .into_iter()
        .map(|axis| segment(m, -axis * SIZE, axis * SIZE, style))
        .collect()
}

/// A camera's view pyramid out to a fifth of its distance, framed at `aspect`, with a triangle
/// marking up.
fn camera(camera: &crate::scene::Camera, reference: f32, aspect: f32, style: Style) -> Vec<Wire> {
    let (eye, orientation, distance) = camera.eye(reference);
    let depth = distance * 0.2;
    let h = depth * (camera.fov_y_degrees.to_radians() * 0.5).tan();
    let w = h * aspect;
    let m = Mat4::from_rotation_translation(orientation, eye);
    let c = [
        Vec3::new(-w, -h, -depth),
        Vec3::new(w, -h, -depth),
        Vec3::new(w, h, -depth),
        Vec3::new(-w, h, -depth),
    ];
    let mut wires: Vec<Wire> = c
        .iter()
        .map(|corner| segment(m, Vec3::ZERO, *corner, style))
        .collect();
    wires.push(Wire {
        points: [c[0], c[1], c[2], c[3], c[0]]
            .iter()
            .map(|p| m.transform_point3(*p))
            .collect(),
        style,
    });
    wires.push(Wire {
        points: [
            Vec3::new(-w * 0.4, h * 1.1, -depth),
            Vec3::new(0.0, h * 1.5, -depth),
            Vec3::new(w * 0.4, h * 1.1, -depth),
        ]
        .iter()
        .map(|p| m.transform_point3(*p))
        .collect(),
        style,
    });
    wires
}

/// The sun at its node's place: an arrow along its light, a unit long.
fn sun(origin: Vec3, toward_sun: Vec3, style: Style) -> Vec<Wire> {
    let tail = origin + toward_sun;
    let along = -toward_sun;
    let side = along.any_orthonormal_vector() * 0.12;
    let back = origin - along * 0.25;
    vec![
        Wire {
            points: vec![tail, origin],
            style,
        },
        Wire {
            points: vec![back + side, origin, back - side],
            style,
        },
    ]
}

/// The extras of the nodes the viewport shows at `frame` under `policy`; none when its extras
/// are off. `selected` draws brighter, `aspect` frames the cameras.
pub fn extras(
    document: &WorldDocument,
    frame: f64,
    policy: &ViewportPolicy,
    selected: &[NodeId],
    aspect: f32,
) -> Result<Vec<Wire>, String> {
    let mut wires = Vec::new();
    if !policy.extras {
        return Ok(wires);
    }
    for node in document.nodes() {
        if !document.shows(node.id, node.kind, frame, policy.shown)? {
            continue;
        }
        let style = if selected.contains(&node.id) {
            Style::Selected
        } else {
            Style::Plain
        };
        match node.kind {
            WorldKind::Mask => {
                let settings = document.mask(node.id, frame)?;
                let style = if settings.invert && style == Style::Plain {
                    Style::Cut
                } else {
                    style
                };
                wires.extend(mask(document.node_world(node.id, frame)?, &settings, style));
            }
            WorldKind::Group => {
                wires.extend(locator(document.node_world(node.id, frame)?, style));
            }
            WorldKind::Camera if document.active_camera != Some(node.id) => {
                let (pose, reference) = document.camera_pose(node.id, frame)?;
                wires.extend(camera(&pose, reference, aspect, style));
            }
            WorldKind::DirectionalLight => {
                let light = document.sun_light(node.id, frame)?;
                let origin = document
                    .node_world(node.id, frame)?
                    .transform_point3(Vec3::ZERO);
                wires.extend(sun(origin, light.direction(), style));
            }
            _ => {}
        }
    }
    Ok(wires)
}

/// The cubes of a Bounds display (`EffectiveRender::show`).
pub fn bounds(bounds: &[Bounds]) -> impl Iterator<Item = Wire> + '_ {
    bounds.iter().flat_map(|b| {
        cube(
            Mat4::from_cols_array_2d(&b.world),
            Vec3::splat(b.half),
            Style::Plain,
        )
    })
}

/// `wire`'s screen segments through `clip` (projection * view) into `rect`, each cut at the
/// camera plane (only what is in front of the eye is drawn).
pub fn project(wire: &Wire, clip: Mat4, rect: egui::Rect) -> Vec<[egui::Pos2; 2]> {
    const NEAR: f32 = 1.0e-6;
    let screen = |p: glam::Vec4| {
        let ndc = p.truncate() / p.w;
        egui::pos2(
            rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(),
            rect.top() + (0.5 - ndc.y * 0.5) * rect.height(),
        )
    };
    wire.points
        .windows(2)
        .filter_map(|pair| {
            let (mut a, mut c) = (clip * pair[0].extend(1.0), clip * pair[1].extend(1.0));
            if a.w <= NEAR && c.w <= NEAR {
                return None;
            }
            if a.w <= NEAR {
                a = c + (a - c) * ((c.w - NEAR) / (c.w - a.w));
            } else if c.w <= NEAR {
                c = a + (c - a) * ((a.w - NEAR) / (a.w - c.w));
            }
            Some([screen(a), screen(c)])
        })
        .collect()
}
