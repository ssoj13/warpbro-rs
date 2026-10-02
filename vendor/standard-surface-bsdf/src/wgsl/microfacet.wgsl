// standard-surface-bsdf: WGSL twin of src/microfacet.rs, statement for statement.
// MaterialX v1.39.5-22-g47cecce6: lib/mx_microfacet_specular.glsl, lib/mx_microfacet.glsl,
// mx_roughness_anisotropy.glsl, mx_dielectric_bsdf.glsl, mx_conductor_bsdf.glsl.

// dielectric_bsdf inputs (pbrlib_defs.mtlx:62-73) in reflection mode.
struct MxDielectric {
    weight: f32,
    tint: vec3<f32>,
    ior: f32,
    roughness: vec2<f32>,
    thinfilm_thickness: f32,
    thinfilm_ior: f32,
    thin_film_energy: u32,
}

// conductor_bsdf inputs (mx_conductor_bsdf.glsl:4).
struct MxConductor {
    weight: f32,
    ior_n: vec3<f32>,
    ior_k: vec3<f32>,
    roughness: vec2<f32>,
    thinfilm_thickness: f32,
    thinfilm_ior: f32,
    thin_film_energy: u32,
}

// lib/mx_microfacet_specular.glsl:32-39
fn mx_ggx_ndf(h: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let hex = h.x / alpha.x;
    let hey = h.y / alpha.y;
    let denom = hex * hex + hey * hey + mx_square(h.z);
    return 1.0 / (MX_PI * alpha.x * alpha.y * mx_square(denom));
}

// lib/mx_microfacet_specular.glsl:70-77
fn mx_ggx_smith_g1(cos_theta: f32, alpha: f32) -> f32 {
    let cos_theta2 = mx_square(cos_theta);
    let tan_theta2 = (1.0 - cos_theta2) / cos_theta2;
    return 2.0 / (1.0 + sqrt(1.0 + mx_square(alpha) * tan_theta2));
}

// lib/mx_microfacet_specular.glsl:79-88
fn mx_ggx_smith_g2(ndotl: f32, ndotv: f32, alpha: f32) -> f32 {
    let alpha2 = mx_square(alpha);
    let lambda_l = sqrt(alpha2 + (1.0 - alpha2) * mx_square(ndotl));
    let lambda_v = sqrt(alpha2 + (1.0 - alpha2) * mx_square(ndotv));
    return 2.0 * ndotl * ndotv / (lambda_l * ndotv + lambda_v * ndotl);
}

// lib/mx_microfacet_specular.glsl:90-108
fn mx_ggx_dir_albedo_analytic(ndotv: f32, alpha: f32, f0: vec3<f32>, f90: vec3<f32>) -> vec3<f32> {
    let x = ndotv;
    let y = alpha;
    let x2 = mx_square(x);
    let y2 = mx_square(y);
    let r = MX_GGX_ALBEDO_C0 + MX_GGX_ALBEDO_C1 * x + MX_GGX_ALBEDO_C2 * y
        + MX_GGX_ALBEDO_C3 * x * y + MX_GGX_ALBEDO_C4 * x2 + MX_GGX_ALBEDO_C5 * y2
        + MX_GGX_ALBEDO_C6 * x2 * y + MX_GGX_ALBEDO_C7 * x * y2 + MX_GGX_ALBEDO_C8 * x2 * y2;
    let ab_x = clamp(r.x / r.z, 0.0, 1.0);
    let ab_y = clamp(r.y / r.w, 0.0, 1.0);
    return f0 * ab_x + f90 * ab_y;
}

// lib/mx_microfacet_specular.glsl:171-174
fn mx_ggx_dir_albedo_scalar(ndotv: f32, alpha: f32, f0: f32, f90: f32) -> f32 {
    return mx_ggx_dir_albedo_analytic(ndotv, alpha, vec3<f32>(f0), vec3<f32>(f90)).x;
}

// lib/mx_microfacet_specular.glsl:176-180
fn mx_average_alpha(alpha: vec2<f32>) -> f32 {
    return sqrt(alpha.x * alpha.y);
}

// pbrlib/genglsl/mx_roughness_anisotropy.glsl:1-15
fn mx_roughness_anisotropy(roughness: f32, anisotropy: f32) -> vec2<f32> {
    let roughness_sqr = clamp(roughness * roughness, MX_FLOAT_EPS, 1.0);
    if (anisotropy > 0.0) {
        let aspect = sqrt(1.0 - clamp(anisotropy, 0.0, MX_ANISOTROPY_MAX));
        return vec2<f32>(min(roughness_sqr / aspect, 1.0), roughness_sqr * aspect);
    }
    return vec2<f32>(roughness_sqr, roughness_sqr);
}

// pbrlib/genglsl/lib/mx_microfacet.glsl:55-59
fn mx_forward_facing_normal(n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    if (dot(n, v) < 0.0) {
        return n * -1.0;
    }
    return n;
}

// pbrlib/genglsl/lib/mx_microfacet.glsl:108-118 (Duff et al. 2017); columns X, Y, N.
fn mx_orthonormal_basis(n: vec3<f32>) -> mat3x3<f32> {
    let sign_ = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign_ + n.z);
    let b = n.x * n.y * a;
    let x = vec3<f32>(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
    let y = vec3<f32>(b, sign_ + n.y * n.y * a, -n.y);
    return mat3x3<f32>(x, y, n);
}

// Plan §4 disc widening: per axis min(1, sqrt(alpha^2 + widen^2)); widen <= 0 keeps alpha.
fn ss_ndf_alpha(alpha: vec2<f32>, widen: f32) -> vec2<f32> {
    if (widen > 0.0) {
        let w2 = widen * widen;
        return vec2<f32>(
            min(sqrt(alpha.x * alpha.x + w2), 1.0),
            min(sqrt(alpha.y * alpha.y + w2), 1.0),
        );
    }
    return alpha;
}

// Plan §2 (C2): delta lobe when max(alpha_x, alpha_y) < SS_ALPHA_MIN.
fn ss_is_delta(alpha: vec2<f32>) -> bool {
    return max(alpha.x, alpha.y) < SS_ALPHA_MIN;
}

// Directional albedo behind a dielectric throughput: MaterialX scalar E(F0) comp, or the
// film-aware E(fd) comp for SS_THIN_FILM_ENERGY_CONSERVING (task R2b).
fn ss_dielectric_dir_albedo(ndotv: f32, avg_alpha: f32, f0: f32, fd: MxFresnelData, comp: vec3<f32>, energy: u32) -> vec3<f32> {
    if (energy == SS_THIN_FILM_ENERGY_CONSERVING) {
        return ss_film_dir_albedo(ndotv, avg_alpha, fd, energy) * comp;
    }
    return comp * mx_ggx_dir_albedo_scalar(ndotv, avg_alpha, f0, 1.0);
}

// pbrlib/genglsl/mx_dielectric_bsdf.glsl:4-52 (CLOSURE_TYPE_REFLECTION).
fn mx_dielectric_bsdf_reflection(v: vec3<f32>, l: vec3<f32>, n_in: vec3<f32>, x_in: vec3<f32>, p: MxDielectric, ndf_widen: f32) -> MxBsdf {
    if (p.weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);

    let fd = mx_init_fresnel_dielectric(p.ior, p.thinfilm_thickness, p.thinfilm_ior);
    let f0 = mx_ior_to_f0(p.ior);

    let safe_alpha = vec2<f32>(
        clamp(p.roughness.x, MX_FLOAT_EPS, 1.0),
        clamp(p.roughness.y, MX_FLOAT_EPS, 1.0),
    );
    let avg_alpha = mx_average_alpha(safe_alpha);
    let safe_tint = max(p.tint, vec3<f32>(0.0));

    let x = normalize(x_in - n * dot(x_in, n));
    let y = cross(n, x);
    let h = normalize(l + v);

    let ndotl = clamp(dot(n, l), MX_FLOAT_EPS, 1.0);
    let vdoth = clamp(dot(v, h), MX_FLOAT_EPS, 1.0);

    let ht = vec3<f32>(dot(h, x), dot(h, y), dot(h, n));

    let f = mx_compute_fresnel(vdoth, fd);
    let d = mx_ggx_ndf(ht, ss_ndf_alpha(safe_alpha, ndf_widen));
    let g = mx_ggx_smith_g2(ndotl, ndotv, avg_alpha);

    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, fd);
    let dir_albedo = ss_dielectric_dir_albedo(ndotv, avg_alpha, f0, fd, comp, p.thin_film_energy);
    return MxBsdf(
        f * d * g * comp * safe_tint * p.weight / (4.0 * ndotv),
        vec3<f32>(1.0) - dir_albedo * p.weight,
    );
}

// pbrlib/genglsl/mx_dielectric_bsdf.glsl:64-72 with lib/mx_environment_prefilter.glsl:10-23.
fn mx_dielectric_bsdf_indirect(v: vec3<f32>, n_in: vec3<f32>, p: MxDielectric, radiance: vec3<f32>) -> MxBsdf {
    if (p.weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);

    let fd = mx_init_fresnel_dielectric(p.ior, p.thinfilm_thickness, p.thinfilm_ior);
    let f0 = mx_ior_to_f0(p.ior);

    let safe_alpha = vec2<f32>(
        clamp(p.roughness.x, MX_FLOAT_EPS, 1.0),
        clamp(p.roughness.y, MX_FLOAT_EPS, 1.0),
    );
    let avg_alpha = mx_average_alpha(safe_alpha);
    let safe_tint = max(p.tint, vec3<f32>(0.0));

    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, fd);
    let dir_albedo = ss_dielectric_dir_albedo(ndotv, avg_alpha, f0, fd, comp, p.thin_film_energy);

    let li = radiance * ss_film_dir_albedo(ndotv, avg_alpha, fd, p.thin_film_energy);
    return MxBsdf(
        li * safe_tint * comp * p.weight,
        vec3<f32>(1.0) - dir_albedo * p.weight,
    );
}

// pbrlib/genglsl/mx_conductor_bsdf.glsl:4-44 (CLOSURE_TYPE_REFLECTION).
fn mx_conductor_bsdf_reflection(v: vec3<f32>, l: vec3<f32>, n_in: vec3<f32>, x_in: vec3<f32>, p: MxConductor, ndf_widen: f32) -> MxBsdf {
    if (p.weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);

    let fd = mx_init_fresnel_conductor(p.ior_n, p.ior_k, p.thinfilm_thickness, p.thinfilm_ior);

    let safe_alpha = vec2<f32>(
        clamp(p.roughness.x, MX_FLOAT_EPS, 1.0),
        clamp(p.roughness.y, MX_FLOAT_EPS, 1.0),
    );
    let avg_alpha = mx_average_alpha(safe_alpha);

    let x = normalize(x_in - n * dot(x_in, n));
    let y = cross(n, x);
    let h = normalize(l + v);

    let ndotl = clamp(dot(n, l), MX_FLOAT_EPS, 1.0);
    let vdoth = clamp(dot(v, h), MX_FLOAT_EPS, 1.0);

    let ht = vec3<f32>(dot(h, x), dot(h, y), dot(h, n));

    let f = mx_compute_fresnel(vdoth, fd);
    let d = mx_ggx_ndf(ht, ss_ndf_alpha(safe_alpha, ndf_widen));
    let g = mx_ggx_smith_g2(ndotl, ndotv, avg_alpha);

    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, fd);
    return MxBsdf(
        f * d * g * comp * p.weight / (4.0 * ndotv),
        vec3<f32>(0.0),
    );
}

// pbrlib/genglsl/mx_conductor_bsdf.glsl:45-50 (CLOSURE_TYPE_INDIRECT).
fn mx_conductor_bsdf_indirect(v: vec3<f32>, n_in: vec3<f32>, p: MxConductor, radiance: vec3<f32>) -> MxBsdf {
    if (p.weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);

    let fd = mx_init_fresnel_conductor(p.ior_n, p.ior_k, p.thinfilm_thickness, p.thinfilm_ior);

    let safe_alpha = vec2<f32>(
        clamp(p.roughness.x, MX_FLOAT_EPS, 1.0),
        clamp(p.roughness.y, MX_FLOAT_EPS, 1.0),
    );
    let avg_alpha = mx_average_alpha(safe_alpha);

    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, fd);
    let li = radiance * ss_film_dir_albedo(ndotv, avg_alpha, fd, p.thin_film_energy);
    return MxBsdf(li * comp * p.weight, vec3<f32>(0.0));
}
