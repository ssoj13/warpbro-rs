use glam::Vec2;

/// Viewport extent in pixels (minimum 1×1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewportSize {
    pub width: u32,
    pub height: u32,
}

impl ViewportSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
        }
    }

    pub fn width_f(self) -> f32 {
        self.width as f32
    }

    pub fn height_f(self) -> f32 {
        self.height as f32
    }

    pub fn aspect(self) -> f32 {
        self.width_f() / self.height_f()
    }
}

/// Pointer position inside the viewport widget (top-left origin, Y down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportPoint {
    pub x: f32,
    pub y: f32,
}

impl ViewportPoint {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn new_clamped(x: f32, y: f32, size: ViewportSize) -> Self {
        Self {
            x: x.clamp(0.0, size.width_f()),
            y: y.clamp(0.0, size.height_f()),
        }
    }

    pub fn from_screen(
        screen_x: f32,
        screen_y: f32,
        rect_min_x: f32,
        rect_min_y: f32,
        size: ViewportSize,
    ) -> Self {
        Self::new_clamped(
            screen_x - rect_min_x,
            screen_y - rect_min_y,
            size,
        )
    }

    pub fn needed_clamp(raw_x: f32, raw_y: f32, size: ViewportSize) -> bool {
        raw_x < 0.0
            || raw_y < 0.0
            || raw_x > size.width_f()
            || raw_y > size.height_f()
    }

    pub fn to_ndc(self, size: ViewportSize) -> NdcPoint {
        NdcPoint::from_viewport(self, size)
    }

    pub fn to_texture_uv(self, size: ViewportSize) -> TextureUv {
        TextureUv::from_viewport(self, size)
    }
}

/// Normalized device coordinates for WebGPU (`clip_space`, Y up).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NdcPoint {
    pub x: f32,
    pub y: f32,
}

impl NdcPoint {
    pub fn from_viewport(p: ViewportPoint, size: ViewportSize) -> Self {
        let w = size.width_f();
        let h = size.height_f();
        Self {
            x: (p.x / w) * 2.0 - 1.0,
            y: 1.0 - (p.y / h) * 2.0,
        }
    }

    pub fn as_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    pub fn to_viewport(self, size: ViewportSize) -> ViewportPoint {
        let w = size.width_f();
        let h = size.height_f();
        ViewportPoint {
            x: ((self.x + 1.0) * 0.5) * w,
            y: (1.0 - self.y) * 0.5 * h,
        }
    }
}

/// Texture / framebuffer UV (0..1), top-left origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureUv {
    pub u: f32,
    pub v: f32,
}

impl TextureUv {
    pub fn from_viewport(p: ViewportPoint, size: ViewportSize) -> Self {
        Self {
            u: p.x / size.width_f(),
            v: p.y / size.height_f(),
        }
    }
}

/// Screen-space axis-aligned rectangle (egui `Rect` fields).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenRect {
    pub min_x: f32,
    pub min_y: f32,
    pub width: f32,
    pub height: f32,
}

impl ScreenRect {
    pub fn viewport_size(self) -> ViewportSize {
        ViewportSize::new(self.width.max(1.0) as u32, self.height.max(1.0) as u32)
    }

    pub fn contains_screen(self, screen_x: f32, screen_y: f32) -> bool {
        screen_x >= self.min_x
            && screen_y >= self.min_y
            && screen_x <= self.min_x + self.width
            && screen_y <= self.min_y + self.height
    }

    pub fn screen_to_viewport(self, screen_x: f32, screen_y: f32) -> ViewportPoint {
        ViewportPoint::from_screen(screen_x, screen_y, self.min_x, self.min_y, self.viewport_size())
    }
}

/// World-space ray from a viewport pixel and inverse view-projection (RH, WebGPU depth).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportRay {
    pub origin: [f32; 3],
    pub dir: [f32; 3],
}

impl ViewportRay {
    /// Build a world-space ray from a viewport pixel and the camera's inverse
    /// view-projection.
    ///
    /// **Why `project_point3` and not `transform_point3`:** `inv_view_proj` is
    /// *not* affine — the inverse of a perspective matrix has a non-trivial
    /// bottom row, so the implicit `w = 1` shortcut in `transform_point3`
    /// produces a constant-direction ray (all NDCs collapse to camera
    /// `forward`). `project_point3` performs the perspective divide that the
    /// inverse projection needs, so off-centre pixels tilt the ray correctly.
    /// See glam docs for `Mat4::transform_point3` (“assumes affine”) vs
    /// `Mat4::project_point3` (“applies perspective correction”).
    pub fn from_inv_view_proj(
        point: ViewportPoint,
        size: ViewportSize,
        inv_view_proj_cols: [[f32; 4]; 4],
        eye: [f32; 3],
    ) -> Self {
        use glam::{Mat4, Vec3};
        let ndc = point.to_ndc(size);
        let inv = Mat4::from_cols_array_2d(&inv_view_proj_cols);
        // Reverse-Z: the near plane is at NDC depth 1, the far plane at 0 (the
        // projection swaps near/far). Unproject both so `dir` still points INTO
        // the scene (far − near); getting these backwards reverses the pick ray.
        let near = inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
        let far = inv.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
        let dir = (far - near).normalize();
        Self {
            origin: eye,
            dir: dir.to_array(),
        }
    }
}

/// Pan deltas are already in viewport pixels (no conversion).
pub type PanDelta = Vec2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_corners_to_ndc() {
        let size = ViewportSize::new(800, 600);
        let tl = ViewportPoint { x: 0.0, y: 0.0 }.to_ndc(size);
        assert!((tl.x + 1.0).abs() < 1e-5);
        assert!((tl.y - 1.0).abs() < 1e-5);

        let br = ViewportPoint { x: 800.0, y: 600.0 }.to_ndc(size);
        assert!((br.x - 1.0).abs() < 1e-5);
        assert!((br.y + 1.0).abs() < 1e-5);

        let c = ViewportPoint { x: 400.0, y: 300.0 }.to_ndc(size);
        assert!(c.x.abs() < 1e-5);
        assert!(c.y.abs() < 1e-5);
    }

    #[test]
    fn ndc_round_trip() {
        let size = ViewportSize::new(756, 680);
        let p = ViewportPoint { x: 320.5, y: 210.0 };
        let back = p.to_ndc(size).to_viewport(size);
        assert!((back.x - p.x).abs() < 0.01);
        assert!((back.y - p.y).abs() < 0.01);
    }

    #[test]
    fn clamp_rejects_oob_screen_style() {
        let size = ViewportSize::new(756, 680);
        let p = ViewportPoint::new_clamped(840.0, 318.0, size);
        assert!((p.x - 756.0).abs() < 1e-3);
        assert!((p.y - 318.0).abs() < 1e-3);
        assert!(ViewportPoint::needed_clamp(840.0, 318.0, size));
    }
}
