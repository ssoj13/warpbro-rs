//! Viewport coordinate spaces and pick rays (single source of truth).
//!
//! | Space | Origin | Y axis | Typical use |
//! |-------|--------|--------|-------------|
//! | **Screen** (egui) | top-left of the window | down | `interact_pointer_pos` |
//! | **Viewport pixels** | top-left of the 3D widget | down | UI events, pick input |
//! | **NDC** (WebGPU clip) | center of the viewport | **up** | unproject, clip-space rays |
//!
//! # Pitfalls when unprojecting
//!
//! `inv_view_proj = (P * V).inverse()` is **not affine** — the bottom row of the
//! inverse of a perspective matrix has non-trivial entries, so the implicit
//! `w = 1` shortcut in [`glam::Mat4::transform_point3`] gives wrong results
//! (every pixel collapses to a ray along camera `forward`, dragging stops
//! tracking the cursor, picks become rank-deficient).
//!
//! When unprojecting an NDC point through `inv_view_proj`:
//! - **CPU (Rust):** use [`glam::Mat4::project_point3`] — it performs the
//!   perspective divide. [`ViewportRay::from_inv_view_proj`] is the canonical
//!   path; route every CPU-side viewport raycast through it instead of
//!   re-implementing the matrix math at the call-site.
//! - **GPU (WGSL):** apply `inv_view_proj` to `vec4f(ndc, depth, 1.0)` and
//!   then divide `xyz` by `w` manually (see
//!   `render-pick/shaders/trace_spheres.wgsl::build_ray` for the worked
//!   example).
//!
//! Anything that needs world-space camera rays must take this route; any new
//! `transform_point3` call on a non-affine matrix is almost certainly a bug.

mod coords;

pub use coords::{
    NdcPoint, PanDelta, ScreenRect, TextureUv, ViewportPoint, ViewportRay, ViewportSize,
};
