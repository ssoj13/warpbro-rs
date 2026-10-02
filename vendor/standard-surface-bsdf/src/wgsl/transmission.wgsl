// standard-surface-bsdf: WGSL twin of src/transmission.rs, statement for statement.
// Rough dielectric BTDF (Walter et al. 2007, eqs. 16-17, 21), adjoint (flux) form; Fresnel-free
// under the layered specular, with 1 - F(wo.m) for the bare interface; eta = eta_wi / eta_wo.

// Snell refraction of wo through m into relative IOR eta; zero vector under total internal reflection.
fn ss_refract(wo: vec3<f32>, m: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_o = dot(wo, m);
    let inv_eta = 1.0 / eta;
    let sin2_i = inv_eta * inv_eta * max(1.0 - cos_o * cos_o, 0.0);
    if (sin2_i >= 1.0) {
        return vec3<f32>(0.0);
    }
    let cos_i = sqrt(1.0 - sin2_i);
    return wo * -inv_eta + m * (inv_eta * cos_o - cos_i);
}

// Generalized half vector normalize(wo + eta wi), turned to the side of n (Walter 2007, eq. 16).
fn ss_transmission_half(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let h = normalize(wo + wi * eta);
    if (dot(h, n) < 0.0) {
        return h * -1.0;
    }
    return h;
}

// Smooth refraction: delta alpha, or eta = 1 (straight through).
fn ss_transmission_is_delta(alpha: vec2<f32>, eta: f32) -> bool {
    return ss_is_delta(alpha) || eta == 1.0;
}

// f_t(wo, wi) |n.wi| of closure p (alpha, ior = eta), NDF widened by ndf_widen, G2 on the
// material alpha, 1 - F when fresnel.
fn ss_ggx_transmission(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, t: vec3<f32>, p: MxDielectric, ndf_widen: f32, fresnel: bool) -> f32 {
    let alpha = p.roughness;
    let eta = p.ior;
    let x = normalize(t - n * dot(t, n));
    let y = cross(n, x);
    let v = ss_to_local(wo, x, y, n);
    let l = ss_to_local(wi, x, y, n);
    let sides = v.z > 0.0 && l.z < 0.0;
    if (!sides) {
        return 0.0;
    }
    let h = ss_transmission_half(v, l, vec3<f32>(0.0, 0.0, 1.0), eta);
    let vdoth = dot(v, h);
    let ldoth = dot(l, h);
    let facing = vdoth > 0.0 && ldoth < 0.0;
    if (!facing) {
        return 0.0;
    }
    let d = mx_ggx_ndf(h, ss_ndf_alpha(alpha, ndf_widen));
    let g = 1.0 / (1.0 + ss_ggx_lambda_aniso(v, alpha) + ss_ggx_lambda_aniso(l, alpha));
    let denom = eta * ldoth + vdoth;
    var transmittance = 1.0;
    if (fresnel) {
        transmittance = 1.0 - mx_fresnel_dielectric(vdoth, eta);
    }
    return transmittance * d * g * vdoth * -ldoth * eta * eta / (v.z * denom * denom);
}

// Density of refract(wo, VNDF half vector): D_V(m) |dm/dwi| (Walter 2007, eq. 17).
fn ss_ggx_transmission_pdf(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, t: vec3<f32>, p: MxDielectric) -> f32 {
    let alpha = p.roughness;
    let eta = p.ior;
    let x = normalize(t - n * dot(t, n));
    let y = cross(n, x);
    let v = ss_to_local(wo, x, y, n);
    let l = ss_to_local(wi, x, y, n);
    let sides = v.z > 0.0 && l.z < 0.0;
    if (!sides) {
        return 0.0;
    }
    let h = ss_transmission_half(v, l, vec3<f32>(0.0, 0.0, 1.0), eta);
    let vdoth = dot(v, h);
    let ldoth = dot(l, h);
    let facing = vdoth > 0.0 && ldoth < 0.0;
    if (!facing) {
        return 0.0;
    }
    let visible = ss_ggx_smith_g1_aniso(v, alpha) * vdoth * mx_ggx_ndf(h, alpha) / v.z;
    let denom = eta * ldoth + vdoth;
    return visible * eta * eta * -ldoth / (denom * denom);
}
