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

/// A polyline in world space, of a node (what a viewport click on it picks).
#[derive(Clone, Debug, PartialEq)]
pub struct Wire {
    pub points: Vec<Vec3>,
    pub style: Style,
    pub node: Option<NodeId>,
}

/// Segments of a drawn circle.
const CIRCLE: usize = 48;

fn segment(m: Mat4, a: Vec3, b: Vec3, style: Style) -> Wire {
    Wire {
        points: vec![m.transform_point3(a), m.transform_point3(b)],
        style,
        node: None,
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
        node: None,
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
/// 3x3x3 cells around its origin (which cells it removes is its seed's, not drawn). The kernels'
/// cell `(i, j, k)` spans `[i, i + 1) * edge` per axis (`floor(p / edge)`, fractal-kernels
/// `mask_distance`), so the drawn cells are -1..=1, their borders at -1, 0, 1 and 2 edges.
fn mask(m: Mat4, settings: &crate::scene::Mask, style: Style) -> Vec<Wire> {
    let mut wires = Vec::new();
    for frame in settings.copy_frames() {
        wires.extend(mask_shape(m * frame, settings, style));
    }
    wires
}

/// One copy of a mask's shape, drawn around the local origin of `m`.
fn mask_shape(m: Mat4, mask: &crate::scene::Mask, style: Style) -> Vec<Wire> {
    let s = Vec3::from_array(mask.size);
    match mask.shape {
        MaskShape::Sphere => [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)]
            .into_iter()
            .map(|(u, v)| circle(m, u, v, s.x, style))
            .collect(),
        MaskShape::Box => cube(m, s, style).collect(),
        MaskShape::Cells => {
            let edge = s.x;
            let at = |i: usize| (i as f32 - 1.0) * edge;
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
        node: None,
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
        node: None,
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
            node: None,
        },
        Wire {
            points: vec![back + side, origin, back - side],
            style,
            node: None,
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
        let first = wires.len();
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
        for wire in &mut wires[first..] {
            wire.node = Some(node.id);
        }
    }
    Ok(wires)
}

/// The cubes of a Bounds display (`EffectiveRender::show`), a selected fractal's highlighted.
pub fn bounds<'a>(bounds: &'a [Bounds], selected: &'a [NodeId]) -> impl Iterator<Item = Wire> + 'a {
    bounds.iter().flat_map(move |b| {
        let style = if b.node.is_some_and(|n| selected.contains(&n)) {
            Style::Selected
        } else {
            Style::Plain
        };
        cube(
            Mat4::from_cols_array_2d(&b.world),
            Vec3::splat(b.half),
            style,
        )
        .map(move |mut w| {
            w.node = b.node;
            w
        })
    })
}

/// The wireframe bounding boxes of selected fractal objects (`objects` is empty in
/// `Display::Bounds`, where [`bounds`] already draws all fractals). Drawn with [`Style::Selected`]
/// and `node: None` so the bounding box does not intercept viewport clicks onto objects behind its
/// empty corners. Also highlights fractals descending from a selected group.
pub fn selected_object_bounds<'a>(
    objects: &'a [crate::scene::Scene],
    document: &'a WorldDocument,
    selected: &'a [NodeId],
) -> impl Iterator<Item = Wire> + 'a {
    objects
        .iter()
        .filter(move |object| {
            object.node.is_some_and(|n| {
                selected
                    .iter()
                    .any(|&sel| sel == n || document.descends(n, sel))
            })
        })
        .flat_map(|object| {
            let world = object
                .object_world
                .unwrap_or(Mat4::IDENTITY.to_cols_array_2d());
            cube(
                Mat4::from_cols_array_2d(&world),
                Vec3::splat(object.formula.framing_radius()),
                Style::Selected,
            )
        })
}

/// The node of the wire nearest `pos` within `reach` points, on screen through `clip` into
/// `rect` (`project`), among the nodes `accept` takes: what a viewport click on the extras
/// picks. Wires are drawn over the image, so they win over the fractal behind them.
pub fn wire_at<'a>(
    wires: impl Iterator<Item = &'a Wire>,
    clip: Mat4,
    rect: egui::Rect,
    pos: egui::Pos2,
    reach: f32,
    accept: impl Fn(NodeId) -> bool,
) -> Option<NodeId> {
    let mut best: Option<(f32, NodeId)> = None;
    for wire in wires {
        let Some(node) = wire.node.filter(|node| accept(*node)) else {
            continue;
        };
        for [a, b] in project(wire, clip, rect) {
            let ab = b - a;
            let t = ((pos - a).dot(ab) / ab.length_sq().max(1e-12)).clamp(0.0, 1.0);
            let distance = (a + ab * t).distance(pos);
            if distance <= reach && best.is_none_or(|(d, _)| distance < d) {
                best = Some((distance, node));
            }
        }
    }
    best.map(|(_, node)| node)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Bulb, Formula};

    #[test]
    fn a_click_picks_the_nearest_wire_within_reach() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 100.0));
        // Identity clip: x, y in -1..1 map across the rect, y up.
        let (one, two) = (NodeId::new(), NodeId::new());
        let wire = |x: f32, node: NodeId| Wire {
            points: vec![Vec3::new(x, -1.0, 0.5), Vec3::new(x, 1.0, 0.5)],
            style: Style::Plain,
            node: Some(node),
        };
        let wires = [
            wire(0.0, one),
            wire(0.1, two),
            Wire {
                node: None,
                ..wire(0.5, one)
            },
        ];
        let at = |x: f32| {
            wire_at(
                wires.iter(),
                Mat4::IDENTITY,
                rect,
                egui::pos2(x, 50.0),
                6.0,
                |_| true,
            )
        };
        // The wires stand at x = 100 and 110 points.
        assert_eq!(at(103.0), Some(one));
        assert_eq!(at(108.0), Some(two));
        assert_eq!(at(130.0), None, "out of reach");
        assert_eq!(at(150.0), None, "a wire of no node is not picked");
        // A nearer wire the caller does not accept leaves the farther one picked.
        let only_two = wire_at(
            wires.iter(),
            Mat4::IDENTITY,
            rect,
            egui::pos2(104.0, 50.0),
            6.0,
            |node| node == two,
        );
        assert_eq!(only_two, Some(two));
    }

    #[test]
    fn the_cells_grid_lies_on_the_kernel_cell_borders() {
        let mask = crate::scene::Mask {
            shape: MaskShape::Cells,
            size: [2.0, 2.0, 2.0],
            ..Default::default()
        };
        let wires = mask_shape(Mat4::IDENTITY, &mask, Style::Plain);
        assert_eq!(wires.len(), 48, "16 lines along each of the three axes");
        // Cells -1..=1 of edge 2: borders at integer multiples of the edge, -2..=4.
        for p in wires.iter().flat_map(|w| w.points.iter()) {
            for v in p.to_array() {
                assert!(
                    [-2.0, 0.0, 2.0, 4.0].contains(&v),
                    "{v} is not a cell border"
                );
            }
        }
    }

    #[test]
    fn selected_object_bounds_draws_selected_fractal_and_group_children_without_pick_nodes() {
        let doc = WorldDocument::from_scene(&crate::scene::Scene::preset(Formula::Mandelbulb(
            Bulb::PRESET,
        )));
        let mut e = crate::world::WorldEditor::new(doc);
        let fractal = e
            .document
            .nodes()
            .into_iter()
            .find(|n| n.kind == WorldKind::Fractal)
            .unwrap()
            .id;
        e.execute(crate::world::WorldCommand::Create {
            kind: WorldKind::Group,
            name: "Grp".into(),
            parent: None,
        })
        .unwrap();
        let group = e.active().unwrap();
        e.execute(crate::world::WorldCommand::Reparent {
            id: fractal,
            parent: Some(group),
        })
        .unwrap();
        let evaluated = e.document.snapshot(0.0).unwrap();

        assert_eq!(
            selected_object_bounds(&evaluated.objects, &e.document, &[]).count(),
            0
        );

        let direct: Vec<_> =
            selected_object_bounds(&evaluated.objects, &e.document, &[fractal]).collect();
        assert_eq!(direct.len(), 12, "one cube has 12 wireframe edges");
        assert!(
            direct
                .iter()
                .all(|w| w.style == Style::Selected && w.node.is_none())
        );

        let via_group: Vec<_> =
            selected_object_bounds(&evaluated.objects, &e.document, &[group]).collect();
        assert_eq!(
            via_group.len(),
            12,
            "selecting a group draws its child fractal's box"
        );
    }
}
