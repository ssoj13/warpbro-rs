// standard-surface-bsdf: WGSL twin of src/sampling.rs, statement for statement.
// MaterialX v1.39.5-22-g47cecce6: lib/mx_microfacet.glsl:85-100,
// lib/mx_microfacet_specular.glsl:41-68 (names lower-cased); pcg4d (Jarzynski & Olano 2020).

// pcg4d counter hash, wrapping u32 arithmetic (bit-identical to the Rust twin).
fn ss_pcg4d(v_in: vec4<u32>) -> vec4<u32> {
    var v = v_in * SS_PCG_MUL + vec4<u32>(SS_PCG_INC);
    v.x = v.x + v.y * v.w;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v.w = v.w + v.y * v.z;
    v = v ^ (v >> vec4<u32>(SS_PCG_SHIFT));
    v.x = v.x + v.y * v.w;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v.w = v.w + v.y * v.z;
    return v;
}

// Top 24 bits as a float in [0, 1), exact.
fn ss_u32_to_unit(x: u32) -> f32 {
    return f32(x >> 8u) * SS_U32_TO_UNIT;
}

// One uniform for dimension `dim` of sample `sample` of pixel (px, py).
fn ss_rng(px: u32, py: u32, sample: u32, dim: u32, seed: u32) -> f32 {
    return ss_u32_to_unit(ss_pcg4d(vec4<u32>(px, py, sample, dim ^ seed)).x);
}

// pbrlib/genglsl/lib/mx_microfacet.glsl:85-94
fn mx_cosine_sample_hemisphere(xi: vec2<f32>) -> vec3<f32> {
    let phi = MX_TWO_PI * xi.x;
    let cos_theta = sqrt(xi.y);
    let sin_theta = sqrt(1.0 - xi.y);
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// pbrlib/genglsl/lib/mx_microfacet.glsl:96-100
fn mx_cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    return max(cos_theta, 0.0) * MX_PI_INV;
}

// Uniform direction in the cone about +z with 1 - cos(half_angle) = one_minus_cos_max.
fn ss_sample_cone(xi: vec2<f32>, one_minus_cos_max: f32) -> vec3<f32> {
    let phi = MX_TWO_PI * xi.x;
    let one_minus_cos = xi.y * one_minus_cos_max;
    let cos_theta = 1.0 - one_minus_cos;
    let sin_theta = sqrt(max(one_minus_cos * (2.0 - one_minus_cos), 0.0));
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

// Density of ss_sample_cone.
fn ss_cone_pdf(one_minus_cos_max: f32) -> f32 {
    return 1.0 / (MX_TWO_PI * one_minus_cos_max);
}

// lib/mx_microfacet_specular.glsl:41-62 (Dupuy & Benyoub 2023 spherical caps).
fn mx_ggx_importance_sample_vndf(xi: vec2<f32>, v: vec3<f32>, alpha: vec2<f32>) -> vec3<f32> {
    // Transform the view direction to the hemisphere configuration.
    let hv = normalize(vec3<f32>(v.x * alpha.x, v.y * alpha.y, v.z));

    // Sample a spherical cap in (-V.z, 1].
    let phi = MX_TWO_PI * xi.x;
    let z = (1.0 - xi.y) * (1.0 + hv.z) - hv.z;
    let sin_theta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    let x = sin_theta * cos(phi);
    let y = sin_theta * sin(phi);

    // Compute the microfacet normal.
    let hx = x + hv.x;
    let hy = y + hv.y;
    let hz = z + hv.z;

    // Transform the microfacet normal back to the ellipsoid configuration.
    return normalize(vec3<f32>(hx * alpha.x, hy * alpha.y, max(hz, 0.0)));
}

// lib/mx_microfacet_specular.glsl:64-68; pass the anisotropic G1 (plan §2, C1).
fn mx_ggx_vndf_reflection_pdf(h: vec3<f32>, alpha: vec2<f32>, g1v: f32, ndotv: f32) -> f32 {
    return mx_ggx_ndf(h, alpha) * g1v / (4.0 * ndotv);
}

// Anisotropic GGX Smith Lambda(V) (Heitz 2014, JCGT 3(2), §5); V.z enters squared.
fn ss_ggx_lambda_aniso(v: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let a2 = (alpha.x * alpha.x * v.x * v.x + alpha.y * alpha.y * v.y * v.y) / (v.z * v.z);
    return (-1.0 + sqrt(1.0 + a2)) * 0.5;
}

// Anisotropic Smith G1 = 1 / (1 + Lambda) (Heitz 2014).
fn ss_ggx_smith_g1_aniso(v: vec3<f32>, alpha: vec2<f32>) -> f32 {
    return 1.0 / (1.0 + ss_ggx_lambda_aniso(v, alpha));
}

// World to tangent frame.
fn ss_to_local(v: vec3<f32>, x: vec3<f32>, y: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(v, x), dot(v, y), dot(v, n));
}

// Tangent frame to world.
fn ss_to_world(l: vec3<f32>, x: vec3<f32>, y: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return x * l.x + y * l.y + n * l.z;
}

// Power heuristic (beta = 2), 0 when both densities are 0.
fn ss_power_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    let sum = a2 + b2;
    if (sum > 0.0) {
        return a2 / sum;
    }
    return 0.0;
}

// Balance heuristic, 0 when both densities are 0.
fn ss_balance_heuristic(pdf_a: f32, pdf_b: f32) -> f32 {
    let sum = pdf_a + pdf_b;
    if (sum > 0.0) {
        return pdf_a / sum;
    }
    return 0.0;
}
