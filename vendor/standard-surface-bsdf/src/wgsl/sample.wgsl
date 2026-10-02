// standard-surface-bsdf: WGSL twin of src/sample.rs, statement for statement.
// One-sample lobe selection: VNDF (anisotropic G1 density) for coat/specular/metal, cosine for
// sheen/diffuse, delta lobes below SS_ALPHA_MIN (plan §2).

// Rust `Sample`; lobe is an SS_LOBE_* index.
struct SsSample {
    wi: vec3<f32>,
    weight: vec3<f32>,
    pdf: f32,
    lobe: u32,
    valid: bool,
    delta: bool,
}

// Lobe index for u in [0, 1) from the normalised weights; -1 when every weight is 0.
fn ss_select_lobe(weights: array<f32, SS_LOBE_COUNT>, u: f32) -> i32 {
    var w = weights;
    var chosen = -1;
    var acc = 0.0;
    for (var k: i32 = 0; k < i32(SS_LOBE_COUNT); k = k + 1) {
        let p = w[k];
        if (p > 0.0) {
            chosen = k;
            if (u < acc + p) {
                return chosen;
            }
        }
        acc = acc + p;
    }
    return chosen;
}

// Full-sphere density of one GGX lobe's VNDF reflection: D G1_aniso(V) / (4 V.z).
fn ss_ggx_lobe_pdf(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, t: vec3<f32>, alpha: vec2<f32>) -> f32 {
    let x = normalize(t - n * dot(t, n));
    let y = cross(n, x);
    let v = ss_to_local(wo, x, y, n);
    let l = ss_to_local(wi, x, y, n);
    let h = normalize(v + l);
    let visible = v.z > 0.0 && dot(v, h) > 0.0;
    if (!visible) {
        return 0.0;
    }
    let g1 = ss_ggx_smith_g1_aniso(v, alpha);
    return mx_ggx_vndf_reflection_pdf(h, alpha, g1, v.z);
}

// Mixture density of ss_sample_prepared at wi.
fn ss_pdf_prepared(s: SsLayers, wo: vec3<f32>, wi: vec3<f32>, weights: array<f32, SS_LOBE_COUNT>) -> f32 {
    let n = s.n;
    let above = dot(n, wo) > 0.0 && dot(n, wi) > 0.0;
    if (!above) {
        return ss_pdf_transmission(s, wo, wi, weights);
    }
    let p_coat = weights[SS_LOBE_COAT];
    let p_specular = weights[SS_LOBE_SPECULAR];
    let p_metal = weights[SS_LOBE_METAL];
    let p_sheen = weights[SS_LOBE_SHEEN];
    let p_diffuse = weights[SS_LOBE_DIFFUSE];
    let cos_l = dot(n, wi);
    var pdf = 0.0;
    if (p_coat > 0.0 && !ss_is_delta(s.coat.roughness)) {
        pdf = pdf + p_coat * ss_ggx_lobe_pdf(wo, wi, n, s.coat_tangent, s.coat.roughness);
    }
    if (p_specular > 0.0 && !ss_is_delta(s.specular.roughness)) {
        pdf = pdf + p_specular * ss_ggx_lobe_pdf(wo, wi, n, s.main_tangent, s.specular.roughness);
    }
    if (p_metal > 0.0 && !ss_is_delta(s.metal.roughness)) {
        pdf = pdf + p_metal * ss_ggx_lobe_pdf(wo, wi, n, s.main_tangent, s.metal.roughness);
    }
    if (p_sheen > 0.0) {
        pdf = pdf + p_sheen * mx_cosine_hemisphere_pdf(cos_l);
    }
    if (p_diffuse > 0.0) {
        pdf = pdf + p_diffuse * mx_cosine_hemisphere_pdf(cos_l);
    }
    return pdf;
}

// Density of the transmission lobe for wo above and wi below the horizon; 0 otherwise.
fn ss_pdf_transmission(s: SsLayers, wo: vec3<f32>, wi: vec3<f32>, weights: array<f32, SS_LOBE_COUNT>) -> f32 {
    let n = s.n;
    let p_transmission = weights[SS_LOBE_TRANSMISSION];
    let eta = s.transmission.ior;
    let continuous = dot(n, wo) > 0.0 && dot(n, wi) < 0.0 && p_transmission > 0.0 && !ss_transmission_is_delta(s.transmission.roughness, eta);
    if (!continuous) {
        return 0.0;
    }
    return p_transmission * ss_ggx_transmission_pdf(wo, wi, n, s.main_tangent, s.transmission);
}

// ss_pdf with caller-supplied weights.
fn ss_pdf_with(i: SsInputs, f: SsFrame, wo: vec3<f32>, wi: vec3<f32>, weights: array<f32, SS_LOBE_COUNT>) -> f32 {
    return ss_pdf_prepared(ss_layers(i, f, wo), wo, wi, weights);
}

// Solid-angle density of the continuous directions of ss_sample.
fn ss_pdf(i: SsInputs, f: SsFrame, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
    return ss_pdf_with(i, f, wo, wi, ss_lobe_weights(i, f, wo));
}

// Delta limit of the dielectric reflection closure: F(NdotV) comp tint w.
fn ss_dielectric_mirror_response(v: vec3<f32>, n_in: vec3<f32>, p: MxDielectric) -> vec3<f32> {
    if (p.weight < MX_FLOAT_EPS) {
        return vec3<f32>(0.0);
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);
    let fd = mx_init_fresnel_dielectric(p.ior, p.thinfilm_thickness, p.thinfilm_ior);
    let safe_alpha = vec2<f32>(
        clamp(p.roughness.x, MX_FLOAT_EPS, 1.0),
        clamp(p.roughness.y, MX_FLOAT_EPS, 1.0),
    );
    let avg_alpha = mx_average_alpha(safe_alpha);
    let f = mx_compute_fresnel(ndotv, fd);
    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, fd);
    return f * comp * max(p.tint, vec3<f32>(0.0)) * p.weight;
}

// Delta limit of the conductor reflection closure: F(NdotV) comp w.
fn ss_conductor_mirror_response(v: vec3<f32>, n_in: vec3<f32>, p: MxConductor) -> vec3<f32> {
    if (p.weight < MX_FLOAT_EPS) {
        return vec3<f32>(0.0);
    }
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);
    let fd = mx_init_fresnel_conductor(p.ior_n, p.ior_k, p.thinfilm_thickness, p.thinfilm_ior);
    let safe_alpha = vec2<f32>(
        clamp(p.roughness.x, MX_FLOAT_EPS, 1.0),
        clamp(p.roughness.y, MX_FLOAT_EPS, 1.0),
    );
    let avg_alpha = mx_average_alpha(safe_alpha);
    let f = mx_compute_fresnel(ndotv, fd);
    let comp = mx_ggx_energy_compensation(ndotv, avg_alpha, fd);
    return f * comp * p.weight;
}

// Reflected energy of a delta GGX lobe with every layer factor above it.
fn ss_delta_response(s: SsLayers, wo: vec3<f32>, lobe: u32) -> vec3<f32> {
    let coat_throughput = mx_dielectric_bsdf_indirect(wo, s.n, s.coat, vec3<f32>(1.0)).throughput;
    let under_coat = coat_throughput * s.attenuation;
    let m = s.metalness;
    if (lobe == SS_LOBE_COAT) {
        return ss_dielectric_mirror_response(wo, s.n, s.coat);
    } else if (lobe == SS_LOBE_SPECULAR) {
        return under_coat * (1.0 - m) * ss_dielectric_mirror_response(wo, s.n, s.specular);
    } else if (lobe == SS_LOBE_METAL) {
        return under_coat * m * ss_conductor_mirror_response(wo, s.n, s.metal);
    } else if (lobe == SS_LOBE_TRANSMISSION) {
        return ss_transmission_albedo(s, wo);
    }
    return vec3<f32>(0.0);
}

// The transmission branch of ss_sample_prepared: smooth refraction (delta) or a VNDF half
// vector refracted below the horizon; p_k is the lobe's selection probability.
fn ss_sample_transmission(i: SsInputs, f: SsFrame, s: SsLayers, wo: vec3<f32>, xi: vec2<f32>, weights: array<f32, SS_LOBE_COUNT>, p_k: f32) -> SsSample {
    let n = s.n;
    let lobe = SS_LOBE_TRANSMISSION;
    let alpha = s.transmission.roughness;
    let eta = s.transmission.ior;
    if (ss_transmission_is_delta(alpha, eta)) {
        let wi = ss_refract(wo, n, eta);
        return SsSample(wi, ss_delta_response(s, wo, lobe) / p_k, p_k, lobe, dot(n, wi) < 0.0, true);
    }
    let x = normalize(s.main_tangent - n * dot(s.main_tangent, n));
    let y = cross(n, x);
    let v = ss_to_local(wo, x, y, n);
    let h = mx_ggx_importance_sample_vndf(xi, v, alpha);
    let wi = ss_to_world(ss_refract(v, h, eta), x, y, n);
    // Total internal reflection gives the zero vector, which is not below the horizon.
    let below = dot(n, wi) < 0.0;
    if (!below) {
        return SsSample(wi, vec3<f32>(0.0), 0.0, lobe, false, false);
    }
    let density = ss_pdf_prepared(s, wo, wi, weights);
    let positive = density > 0.0;
    if (!positive) {
        return SsSample(wi, vec3<f32>(0.0), 0.0, lobe, false, false);
    }
    let e = ss_eval_light(i, f, wo, wi);
    return SsSample(wi, e.transmission / density, density, lobe, true, false);
}

// One BSDF sample: u.x selects the lobe, u.yz sample it.
fn ss_sample_prepared(i: SsInputs, f: SsFrame, s: SsLayers, wo: vec3<f32>, u: vec3<f32>, weights: array<f32, SS_LOBE_COUNT>) -> SsSample {
    let invalid = SsSample(vec3<f32>(0.0), vec3<f32>(0.0), 0.0, SS_LOBE_DIFFUSE, false, false);
    let n = s.n;
    let facing = dot(n, wo) > 0.0;
    if (!facing) {
        return invalid;
    }
    let xi = vec2<f32>(u.y, u.z);
    let chosen = ss_select_lobe(weights, u.x);
    if (chosen < 0) {
        return invalid;
    }
    let lobe = u32(chosen);
    var w = weights;
    let p_k = w[lobe];
    if (lobe == SS_LOBE_TRANSMISSION) {
        return ss_sample_transmission(i, f, s, wo, xi, weights, p_k);
    }
    var alpha = vec2<f32>(1.0, 1.0);
    var tangent = n;
    var ggx = false;
    if (lobe == SS_LOBE_COAT) {
        alpha = s.coat.roughness;
        tangent = s.coat_tangent;
        ggx = true;
    } else if (lobe == SS_LOBE_SPECULAR) {
        alpha = s.specular.roughness;
        tangent = s.main_tangent;
        ggx = true;
    } else if (lobe == SS_LOBE_METAL) {
        alpha = s.metal.roughness;
        tangent = s.main_tangent;
        ggx = true;
    }

    if (ggx && ss_is_delta(alpha)) {
        let wi = reflect(wo * -1.0, n);
        return SsSample(wi, ss_delta_response(s, wo, lobe) / p_k, p_k, lobe, dot(n, wi) > 0.0, true);
    }

    var wi: vec3<f32>;
    if (ggx) {
        let x = normalize(tangent - n * dot(tangent, n));
        let y = cross(n, x);
        let v = ss_to_local(wo, x, y, n);
        let h = mx_ggx_importance_sample_vndf(xi, v, alpha);
        let l = reflect(v * -1.0, h);
        wi = ss_to_world(l, x, y, n);
    } else {
        let basis = mx_orthonormal_basis(n);
        wi = ss_to_world(mx_cosine_sample_hemisphere(xi), basis[0], basis[1], n);
    }
    let above = dot(n, wi) > 0.0;
    if (!above) {
        return SsSample(wi, vec3<f32>(0.0), 0.0, lobe, false, false);
    }
    let density = ss_pdf_prepared(s, wo, wi, weights);
    let positive = density > 0.0;
    if (!positive) {
        return SsSample(wi, vec3<f32>(0.0), 0.0, lobe, false, false);
    }
    let e = ss_eval_light(i, f, wo, wi);
    return SsSample(wi, (e.base + e.specular) / density, density, lobe, true, false);
}

// ss_sample with caller-supplied weights.
fn ss_sample_with(i: SsInputs, f: SsFrame, wo: vec3<f32>, u: vec3<f32>, weights: array<f32, SS_LOBE_COUNT>) -> SsSample {
    return ss_sample_prepared(i, f, ss_layers(i, f, wo), wo, u, weights);
}

// One BSDF sample for view direction wo and uniforms u in [0, 1).
fn ss_sample(i: SsInputs, f: SsFrame, wo: vec3<f32>, u: vec3<f32>) -> SsSample {
    return ss_sample_with(i, f, wo, u, ss_lobe_weights(i, f, wo));
}
