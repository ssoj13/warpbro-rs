// standard-surface-bsdf: WGSL twin of src/sheen.rs, statement for statement.
// MaterialX v1.39.5-22-g47cecce6: lib/mx_microfacet_sheen.glsl, mx_sheen_bsdf.glsl (conty_kulla).

// lib/mx_microfacet_sheen.glsl:3-11
fn mx_imageworks_sheen_ndf(ndoth: f32, roughness: f32) -> f32 {
    let inv_roughness = 1.0 / max(roughness, MX_SHEEN_ROUGHNESS_MIN);
    let cos2 = ndoth * ndoth;
    let sin2 = 1.0 - cos2;
    return (2.0 + inv_roughness) * pow(sin2, inv_roughness * 0.5) / MX_TWO_PI;
}

// lib/mx_microfacet_sheen.glsl:13-25
fn mx_imageworks_sheen_brdf(ndotl: f32, ndotv: f32, ndoth: f32, roughness: f32) -> f32 {
    let d = mx_imageworks_sheen_ndf(ndoth, roughness);
    return d / (4.0 * (ndotl + ndotv - ndotl * ndotv));
}

// lib/mx_microfacet_sheen.glsl:27-37
fn mx_imageworks_sheen_dir_albedo_analytic(ndotv: f32, roughness: f32) -> f32 {
    let r = MX_SHEEN_ALBEDO_C0 + MX_SHEEN_ALBEDO_C1 * ndotv + MX_SHEEN_ALBEDO_C2 * roughness
        + MX_SHEEN_ALBEDO_C3 * ndotv * roughness + MX_SHEEN_ALBEDO_C4 * mx_square(ndotv)
        + MX_SHEEN_ALBEDO_C5 * mx_square(roughness);
    return r.x / r.y;
}

// lib/mx_microfacet_sheen.glsl:81-91
fn mx_imageworks_sheen_dir_albedo(ndotv: f32, roughness: f32) -> f32 {
    let dir_albedo = mx_imageworks_sheen_dir_albedo_analytic(ndotv, roughness);
    return clamp(dir_albedo, 0.0, 1.0);
}

// pbrlib/genglsl/mx_sheen_bsdf.glsl:4-33,42
fn mx_sheen_bsdf_reflection(v: vec3<f32>, l: vec3<f32>, n_in: vec3<f32>, weight: f32, color: vec3<f32>, roughness: f32) -> MxBsdf {
    if (weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);

    let h = normalize(l + v);
    let ndotl = clamp(dot(n, l), MX_FLOAT_EPS, 1.0);
    let ndoth = clamp(dot(n, h), MX_FLOAT_EPS, 1.0);

    let fr = color * mx_imageworks_sheen_brdf(ndotl, ndotv, ndoth, roughness);
    let dir_albedo = mx_imageworks_sheen_dir_albedo(ndotv, roughness);
    return MxBsdf(fr * ndotl * weight, vec3<f32>(1.0 - dir_albedo * weight));
}

// pbrlib/genglsl/mx_sheen_bsdf.glsl:44-60
fn mx_sheen_bsdf_indirect(v: vec3<f32>, n_in: vec3<f32>, weight: f32, color: vec3<f32>, roughness: f32, irradiance: vec3<f32>) -> MxBsdf {
    if (weight < MX_FLOAT_EPS) {
        return MxBsdf(vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);

    let dir_albedo = mx_imageworks_sheen_dir_albedo(ndotv, roughness);
    return MxBsdf(irradiance * color * dir_albedo * weight, vec3<f32>(1.0 - dir_albedo * weight));
}
