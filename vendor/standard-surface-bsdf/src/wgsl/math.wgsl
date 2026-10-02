// standard-surface-bsdf: WGSL twin of src/math.rs (MaterialX v1.39.5-22-g47cecce6 port).
// Vector helpers of the Rust twin are WGSL builtins/operators here; only the named MaterialX
// helpers are functions.

// A BSDF closure value: MaterialX `struct BSDF` (pbrlib/genglsl/lib/mx_closure_type.glsl).
struct MxBsdf {
    response: vec3<f32>,
    throughput: vec3<f32>,
}

// stdlib/genglsl/lib/mx_math.glsl:26-29
fn mx_square(x: f32) -> f32 {
    return x * x;
}

// pbrlib/genglsl/lib/mx_microfacet.glsl:4-7
fn mx_pow5(x: f32) -> f32 {
    return mx_square(mx_square(x)) * x;
}

// pbrlib/genglsl/lib/mx_microfacet.glsl:9-13
fn mx_pow6(x: f32) -> f32 {
    let x2 = mx_square(x);
    return mx_square(x2) * x2;
}
