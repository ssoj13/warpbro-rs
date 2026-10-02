// standard-surface-bsdf: WGSL twin of src/diffuse.rs, statement for statement.
// MaterialX v1.39.5-22-g47cecce6: lib/mx_microfacet_diffuse.glsl, mx_oren_nayar_diffuse_bsdf.glsl.

// lib/mx_microfacet_diffuse.glsl:6-18
fn mx_oren_nayar_diffuse(ndotv: f32, ndotl: f32, ldotv: f32, roughness: f32) -> f32 {
    let s = ldotv - ndotl * ndotv;
    var stinv = 0.0;
    if (s > 0.0) {
        stinv = s / max(ndotl, ndotv);
    }

    let sigma2 = mx_square(roughness);
    let a = 1.0 - 0.5 * (sigma2 / (sigma2 + MX_ON_A_OFFSET));
    let b = MX_ON_B_SCALE * sigma2 / (sigma2 + MX_ON_B_OFFSET);

    return a + b * stinv;
}

// lib/mx_microfacet_diffuse.glsl:20-28
fn mx_oren_nayar_diffuse_dir_albedo_analytic(ndotv: f32, roughness: f32) -> f32 {
    let r = MX_ON_ALBEDO_C0 + MX_ON_ALBEDO_C1 * roughness + MX_ON_ALBEDO_C2 * ndotv * roughness
        + MX_ON_ALBEDO_C3 * mx_square(roughness);
    return r.x / r.y;
}

// lib/mx_microfacet_diffuse.glsl:75-83
fn mx_oren_nayar_diffuse_dir_albedo(ndotv: f32, roughness: f32) -> f32 {
    let dir_albedo = mx_oren_nayar_diffuse_dir_albedo_analytic(ndotv, roughness);
    return clamp(dir_albedo, 0.0, 1.0);
}

// lib/mx_microfacet_diffuse.glsl:85-95
fn mx_oren_nayar_fujii_diffuse_dir_albedo(cos_theta: f32, roughness: f32) -> f32 {
    let a = 1.0 / (1.0 + MX_FUJII_CONSTANT_1 * roughness);
    let b = roughness * a;
    let si = sqrt(max(1.0 - mx_square(cos_theta), 0.0));
    let g = si * (acos(clamp(cos_theta, -1.0, 1.0)) - si * cos_theta)
        + 2.0 * ((si / cos_theta) * (1.0 - si * si * si) - si) / 3.0;
    return a + (b * g * MX_PI_INV);
}

// lib/mx_microfacet_diffuse.glsl:97-101
fn mx_oren_nayar_fujii_diffuse_avg_albedo(roughness: f32) -> f32 {
    let a = 1.0 / (1.0 + MX_FUJII_CONSTANT_1 * roughness);
    return a * (1.0 + MX_FUJII_CONSTANT_2 * roughness);
}

// lib/mx_microfacet_diffuse.glsl:103-127
fn mx_oren_nayar_compensated_diffuse(ndotv: f32, ndotl: f32, ldotv: f32, roughness: f32, color: vec3<f32>) -> vec3<f32> {
    let s = ldotv - ndotl * ndotv;
    var stinv = s;
    if (s > 0.0) {
        stinv = s / max(ndotl, ndotv);
    }

    // Single-scatter lobe.
    let a = 1.0 / (1.0 + MX_FUJII_CONSTANT_1 * roughness);
    let lobe_single_scatter = color * a * (1.0 + roughness * stinv);

    // Multi-scatter lobe.
    let dir_albedo_v = mx_oren_nayar_fujii_diffuse_dir_albedo(ndotv, roughness);
    let dir_albedo_l = mx_oren_nayar_fujii_diffuse_dir_albedo(ndotl, roughness);
    let avg_albedo = mx_oren_nayar_fujii_diffuse_avg_albedo(roughness);
    let color_multi_scatter = color * color * avg_albedo
        / (vec3<f32>(1.0) - color * max(1.0 - avg_albedo, 0.0));
    let lobe_multi_scatter = color_multi_scatter
        * max(1.0 - dir_albedo_v, MX_FLOAT_EPS)
        * max(1.0 - dir_albedo_l, MX_FLOAT_EPS)
        / max(1.0 - avg_albedo, MX_FLOAT_EPS);

    return lobe_single_scatter + lobe_multi_scatter;
}

// lib/mx_microfacet_diffuse.glsl:129-136
fn mx_oren_nayar_compensated_diffuse_dir_albedo(cos_theta: f32, roughness: f32, color: vec3<f32>) -> vec3<f32> {
    let dir_albedo = mx_oren_nayar_fujii_diffuse_dir_albedo(cos_theta, roughness);
    let avg_albedo = mx_oren_nayar_fujii_diffuse_avg_albedo(roughness);
    let color_multi_scatter = color * color * avg_albedo
        / (vec3<f32>(1.0) - color * max(1.0 - avg_albedo, 0.0));
    return mix(color_multi_scatter, color, vec3<f32>(dir_albedo));
}

// pbrlib/genglsl/mx_oren_nayar_diffuse_bsdf.glsl:4-28
fn mx_oren_nayar_diffuse_bsdf_reflection(v: vec3<f32>, l: vec3<f32>, n_in: vec3<f32>, weight: f32, color: vec3<f32>, roughness: f32, energy_compensation: bool) -> MxBsdf {
    if (weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);
    let ndotl = clamp(dot(n, l), MX_FLOAT_EPS, 1.0);
    let ldotv = clamp(dot(l, v), MX_FLOAT_EPS, 1.0);

    var diffuse: vec3<f32>;
    if (energy_compensation) {
        diffuse = mx_oren_nayar_compensated_diffuse(ndotv, ndotl, ldotv, roughness, color);
    } else {
        diffuse = color * mx_oren_nayar_diffuse(ndotv, ndotl, ldotv, roughness);
    }
    return MxBsdf(diffuse * weight * ndotl * MX_PI_INV, vec3<f32>(0.0));
}

// pbrlib/genglsl/mx_oren_nayar_diffuse_bsdf.glsl:29-36
fn mx_oren_nayar_diffuse_bsdf_indirect(v: vec3<f32>, n_in: vec3<f32>, weight: f32, color: vec3<f32>, roughness: f32, energy_compensation: bool, irradiance: vec3<f32>) -> MxBsdf {
    if (weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);

    var diffuse: vec3<f32>;
    if (energy_compensation) {
        diffuse = mx_oren_nayar_compensated_diffuse_dir_albedo(ndotv, roughness, color);
    } else {
        diffuse = color * mx_oren_nayar_diffuse_dir_albedo(ndotv, roughness);
    }
    return MxBsdf(irradiance * diffuse * weight, vec3<f32>(0.0));
}

// lib/mx_microfacet_diffuse.glsl:158-166
fn mx_burley_diffusion_profile(dist: f32, shape: vec3<f32>) -> vec3<f32> {
    let num1 = exp(shape * -dist);
    let num2 = exp(shape * -dist / 3.0);
    let denom = max(dist, MX_FLOAT_EPS);
    return (num1 + num2) / denom;
}

// lib/mx_microfacet_diffuse.glsl:168-192; dot(n, l) clamped to [-1, 1] before acos.
fn mx_integrate_burley_diffusion(n: vec3<f32>, l: vec3<f32>, radius: f32, mfp: vec3<f32>) -> vec3<f32> {
    let theta = acos(clamp(dot(n, l), -1.0, 1.0));

    // Estimate the Burley diffusion shape from mean free path.
    let shape = vec3<f32>(1.0) / max(mfp, vec3<f32>(MX_BURLEY_MFP_MIN));

    // Integrate the profile over the sphere.
    var sum_d = vec3<f32>(0.0);
    var sum_r = vec3<f32>(0.0);
    for (var i: i32 = 0; i < MX_BURLEY_SAMPLE_COUNT; i = i + 1) {
        let x = -MX_PI + (f32(i) + 0.5) * MX_BURLEY_SAMPLE_WIDTH;
        let dist = radius * abs(2.0 * sin(x * 0.5));
        let r = mx_burley_diffusion_profile(dist, shape);
        sum_d = sum_d + r * max(cos(theta + x), 0.0);
        sum_r = sum_r + r;
    }

    return sum_d / sum_r;
}

// lib/mx_microfacet_diffuse.glsl:194-199 with the caller's curvature instead of fwidth(N)/fwidth(P).
fn ss_subsurface_scattering_approx(n: vec3<f32>, l: vec3<f32>, curvature: f32, albedo: vec3<f32>, mfp: vec3<f32>) -> vec3<f32> {
    let radius = 1.0 / max(curvature, MX_SUBSURFACE_CURVATURE_MIN);
    return albedo * mx_integrate_burley_diffusion(n, l, radius, mfp) / MX_PI;
}

// pbrlib/genglsl/mx_subsurface_bsdf.glsl:4-26 (occlusion 1; anisotropy unread by MaterialX).
fn mx_subsurface_bsdf_reflection(v: vec3<f32>, l: vec3<f32>, n_in: vec3<f32>, curvature: f32, weight: f32, color: vec3<f32>, radius: vec3<f32>) -> MxBsdf {
    if (weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    let n = mx_forward_facing_normal(n_in, v);
    let sss = ss_subsurface_scattering_approx(n, l, curvature, color, radius);
    return MxBsdf(sss * weight, vec3<f32>(0.0));
}

// pbrlib/genglsl/mx_subsurface_bsdf.glsl:27-31
fn mx_subsurface_bsdf_indirect(weight: f32, color: vec3<f32>, irradiance: vec3<f32>) -> MxBsdf {
    if (weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(0.0));
    }
    return MxBsdf(irradiance * color * weight, vec3<f32>(0.0));
}
