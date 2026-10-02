//! Scalar and `[f32; 3]` helpers of the Rust twin.
//!
//! Two kinds of items live here:
//! - **Twins** (`mx_square`, `mx_pow5`, `mx_pow6`): MaterialX helpers with a same-named WGSL
//!   function in `wgsl/math.wgsl`, listed in [`crate::wgsl::TWIN_FUNCTIONS`].
//! - **Builtin stand-ins** (`add3`, `dot3`, `normalize3`, `mixf`, `clampf`, ...): the Rust
//!   spelling of WGSL builtins and vector operators. They follow the WGSL definitions
//!   (`mix(a, b, t) = a * (1 - t) + b * t`, `clamp(x, lo, hi) = min(max(x, lo), hi)`,
//!   `reflect(e1, e2) = e1 - 2 dot(e2, e1) e2`) so that both twins evaluate the same f32
//!   expression, and they never use `mul_add`. Components are reached by destructuring, never
//!   by indexing. Transcendentals come from the pure-Rust `libm` crate, so CPU bits do not
//!   depend on the platform C runtime.
//!
//! [`MxBsdf`] is the closure value shared by every closure module.

/// WGSL functions of `wgsl/math.wgsl`, sorted: the twins of this module's functions
/// (`mx_` names kept, `ss_` names are the Rust names with the prefix). Checked by the
/// `twin_registry_matches_wgsl_and_rust` test; collected by [`crate::wgsl::TWIN_FUNCTIONS`].
pub const WGSL_TWINS: &[&str] = &["mx_pow5", "mx_pow6", "mx_square"];

/// A BSDF closure value: MaterialX `struct BSDF { vec3 response; vec3 throughput; }`
/// (`pbrlib/genglsl/lib/mx_closure_type.glsl`). `response` is the reflected radiance (or its
/// `f * cos` factor for direct light), `throughput` the fraction passed to the layer beneath.
/// The MaterialX default value is `BSDF(vec3(0), vec3(1))`
/// (`source/MaterialXGenGlsl/GlslSyntax.cpp:283`), [`MxBsdf::EMPTY`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MxBsdf {
    /// Reflected contribution of this closure.
    pub response: [f32; 3],
    /// Fraction of energy passed to the layer beneath (MaterialX `mx_layer_bsdf`).
    pub throughput: [f32; 3],
}

impl MxBsdf {
    /// The MaterialX default closure value: no response, full throughput. A closure whose weight
    /// is below `M_FLOAT_EPS` returns this unchanged (`mx_dielectric_bsdf.glsl:6-9`).
    pub const EMPTY: Self = Self {
        response: [0.0; 3],
        throughput: [1.0; 3],
    };
}

/// `mx_square(x) = x * x`: `stdlib/genglsl/lib/mx_math.glsl:26-29`.
pub fn mx_square(x: f32) -> f32 {
    x * x
}

/// `mx_pow5(x) = square(square(x)) * x`: `pbrlib/genglsl/lib/mx_microfacet.glsl:4-7`.
pub fn mx_pow5(x: f32) -> f32 {
    mx_square(mx_square(x)) * x
}

/// `mx_pow6(x) = square(square(x)) * square(x)`: `pbrlib/genglsl/lib/mx_microfacet.glsl:9-13`.
pub fn mx_pow6(x: f32) -> f32 {
    let x2 = mx_square(x);
    mx_square(x2) * x2
}

/// WGSL `vec3<f32>(s)`.
pub const fn splat3(s: f32) -> [f32; 3] {
    [s, s, s]
}

/// WGSL `a + b` on `vec3<f32>`.
pub fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    [ax + bx, ay + by, az + bz]
}

/// WGSL `a - b` on `vec3<f32>`.
pub fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    [ax - bx, ay - by, az - bz]
}

/// WGSL `a * b` on `vec3<f32>` (component-wise).
pub fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    [ax * bx, ay * by, az * bz]
}

/// WGSL `a / b` on `vec3<f32>` (component-wise).
pub fn div3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    [ax / bx, ay / by, az / bz]
}

/// WGSL `a / s` for `vec3<f32>` and `f32` (a true division, not a multiply by `1 / s`).
pub fn div3s(a: [f32; 3], s: f32) -> [f32; 3] {
    let [ax, ay, az] = a;
    [ax / s, ay / s, az / s]
}

/// WGSL `a * s` for `vec3<f32>` and `f32`.
pub fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    let [ax, ay, az] = a;
    [ax * s, ay * s, az * s]
}

/// WGSL `dot(a, b)` on `vec3<f32>`, evaluated as `(ax*bx + ay*by) + az*bz`.
pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    ax * bx + ay * by + az * bz
}

/// WGSL `cross(a, b)`.
pub fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
}

/// WGSL `length(a)`.
pub fn length3(a: [f32; 3]) -> f32 {
    crate::fm::sqrtf(dot3(a, a))
}

/// WGSL `normalize(a) = a / length(a)`. A zero vector yields NaN, as in WGSL; callers that can
/// see a degenerate vector test its length first ([`crate::surface::shading_tangent`]).
pub fn normalize3(a: [f32; 3]) -> [f32; 3] {
    let len = length3(a);
    let [ax, ay, az] = a;
    [ax / len, ay / len, az / len]
}

/// WGSL `reflect(e1, e2) = e1 - 2 * dot(e2, e1) * e2`.
pub fn reflect3(e1: [f32; 3], e2: [f32; 3]) -> [f32; 3] {
    sub3(e1, scale3(e2, 2.0 * dot3(e2, e1)))
}

/// WGSL `mix(a, b, t) = a * (1 - t) + b * t` for `f32`.
pub fn mixf(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// WGSL `mix(a, b, t)` for `vec3<f32>` with a scalar `t`.
pub fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    add3(scale3(a, 1.0 - t), scale3(b, t))
}

/// WGSL `mix(a, b, t)` for `vec3<f32>` with a vector `t`.
pub fn mix3v(a: [f32; 3], b: [f32; 3], t: [f32; 3]) -> [f32; 3] {
    add3(mul3(a, sub3(splat3(1.0), t)), mul3(b, t))
}

/// WGSL `clamp(x, lo, hi) = min(max(x, lo), hi)` for `f32`.
pub fn clampf(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

/// WGSL `clamp(v, vec3(lo), vec3(hi))`.
pub fn clamp3(v: [f32; 3], lo: f32, hi: f32) -> [f32; 3] {
    let [x, y, z] = v;
    [clampf(x, lo, hi), clampf(y, lo, hi), clampf(z, lo, hi)]
}

/// WGSL `max(v, vec3(s))`.
pub fn max3s(v: [f32; 3], s: f32) -> [f32; 3] {
    let [x, y, z] = v;
    [x.max(s), y.max(s), z.max(s)]
}

/// WGSL `sqrt(v)` on `vec3<f32>`.
pub fn sqrt3(v: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = v;
    [crate::fm::sqrtf(x), crate::fm::sqrtf(y), crate::fm::sqrtf(z)]
}

/// WGSL `pow(v, vec3(e))`.
pub fn pow3s(v: [f32; 3], e: f32) -> [f32; 3] {
    let [x, y, z] = v;
    [crate::fm::powf(x, e), crate::fm::powf(y, e), crate::fm::powf(z, e)]
}

/// WGSL `cos(v)` on `vec3<f32>`.
pub fn cos3(v: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = v;
    [crate::fm::cosf(x), crate::fm::cosf(y), crate::fm::cosf(z)]
}

/// WGSL `exp(v)` on `vec3<f32>`.
pub fn exp3(v: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = v;
    [crate::fm::expf(x), crate::fm::expf(y), crate::fm::expf(z)]
}

/// WGSL `atan2(y, x)` on `vec3<f32>`.
pub fn atan2_3(y: [f32; 3], x: [f32; 3]) -> [f32; 3] {
    let [yx, yy, yz] = y;
    let [xx, xy, xz] = x;
    [
        crate::fm::atan2f(yx, xx),
        crate::fm::atan2f(yy, xy),
        crate::fm::atan2f(yz, xz),
    ]
}

/// WGSL `max(a, b)` component-wise on `vec3<f32>`.
pub fn max3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    [ax.max(bx), ay.max(by), az.max(bz)]
}

/// WGSL `a + b` on `vec2<f32>`.
pub fn add2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let [ax, ay] = a;
    let [bx, by] = b;
    [ax + bx, ay + by]
}

/// WGSL `a * s` for `vec2<f32>` and `f32`.
pub fn scale2(a: [f32; 2], s: f32) -> [f32; 2] {
    let [ax, ay] = a;
    [ax * s, ay * s]
}

/// WGSL `a + b` on `vec4<f32>`.
pub fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [ax + bx, ay + by, az + bz, aw + bw]
}

/// WGSL `a * s` for `vec4<f32>` and `f32`.
pub fn scale4(a: [f32; 4], s: f32) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    [ax * s, ay * s, az * s, aw * s]
}

/// `true` when every component is finite (Rust-only test and validation helper).
pub fn is_finite3(v: [f32; 3]) -> bool {
    let [x, y, z] = v;
    x.is_finite() && y.is_finite() && z.is_finite()
}
