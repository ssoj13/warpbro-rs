// standard-surface-bsdf: WGSL twin of src/surface.rs, statement for statement.
// MaterialX v1.39.5-22-g47cecce6: bxdf/standard_surface.mtlx:113-429 (reflection-only subset),
// mx_layer_bsdf.glsl, mx_mix_bsdf.glsl, mx_multiply_bsdf_color3.glsl,
// mx_generalized_schlick_edf.glsl, stdlib mx_rotate_vector3.glsl.

// Rust `SurfaceInputs`: MaterialX standard_surface input names (bxdf/standard_surface.mtlx:14-100).
struct SsInputs {
    base: f32,
    base_color: vec3<f32>,
    diffuse_roughness: f32,
    metalness: f32,
    specular: f32,
    specular_color: vec3<f32>,
    specular_roughness: f32,
    specular_ior: f32,
    specular_anisotropy: f32,
    specular_rotation: f32,
    transmission: f32,
    transmission_color: vec3<f32>,
    transmission_extra_roughness: f32,
    subsurface: f32,
    subsurface_color: vec3<f32>,
    subsurface_radius: vec3<f32>,
    subsurface_scale: f32,
    subsurface_anisotropy: f32,
    sheen: f32,
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    coat: f32,
    coat_color: vec3<f32>,
    coat_roughness: f32,
    coat_anisotropy: f32,
    coat_rotation: f32,
    coat_ior: f32,
    coat_affect_color: f32,
    coat_affect_roughness: f32,
    thin_film_thickness: f32,
    thin_film_ior: f32,
    emission: f32,
    emission_color: vec3<f32>,
    thin_film_energy: u32,
}

// Rust `ShadingFrame`: unit forward-facing normal, a tangent hint, whether wo is inside, and
// the surface curvature (1 / radius) read by the subsurface closure.
struct SsFrame {
    n: vec3<f32>,
    tangent: vec3<f32>,
    inside: bool,
    curvature: f32,
}

// Rust `Lobes`: base (diffuse + sheen), specular (coat + specular + metal) and transmission.
struct SsLobes {
    base: vec3<f32>,
    specular: vec3<f32>,
    transmission: vec3<f32>,
}

// Rust `Environment`: irradiance / pi about n, and prefiltered lobe radiances.
struct SsEnvironment {
    irradiance: vec3<f32>,
    specular_radiance: vec3<f32>,
    coat_radiance: vec3<f32>,
    transmission_radiance: vec3<f32>,
}

// Rust `Layers`: derived closure parameters and graph intermediates for one view direction.
struct SsLayers {
    n: vec3<f32>,
    v: vec3<f32>,
    main_tangent: vec3<f32>,
    coat_tangent: vec3<f32>,
    coat: MxDielectric,
    specular: MxDielectric,
    metal: MxConductor,
    attenuation: vec3<f32>,
    metalness: f32,
    diffuse_color: vec3<f32>,
    transmission: MxDielectric,
    transmission_mix: f32,
    inside: bool,
    subsurface_color: vec3<f32>,
    subsurface_radius: vec3<f32>,
    subsurface_mix: f32,
}

// pbrlib/genglsl/mx_layer_bsdf.glsl:3-7
fn mx_layer_bsdf(top: MxBsdf, base: MxBsdf) -> MxBsdf {
    return MxBsdf(top.response + base.response * top.throughput, top.throughput * base.throughput);
}

// pbrlib/genglsl/mx_mix_bsdf.glsl:3-7
fn mx_mix_bsdf(fg: MxBsdf, bg: MxBsdf, mix_value: f32) -> MxBsdf {
    return MxBsdf(
        mix(bg.response, fg.response, vec3<f32>(mix_value)),
        mix(bg.throughput, fg.throughput, vec3<f32>(mix_value)),
    );
}

// pbrlib/genglsl/mx_multiply_bsdf_color3.glsl:3-8
fn mx_multiply_bsdf_color3(in1: MxBsdf, in2: vec3<f32>) -> MxBsdf {
    return MxBsdf(in1.response * clamp(in2, vec3<f32>(0.0), vec3<f32>(1.0)), in1.throughput);
}

// pbrlib/genglsl/mx_generalized_schlick_edf.glsl:4-13
fn mx_generalized_schlick_edf(v: vec3<f32>, n_in: vec3<f32>, color0: vec3<f32>, color90: vec3<f32>, exponent: f32, base: vec3<f32>) -> vec3<f32> {
    let n = mx_forward_facing_normal(n_in, v);
    let ndotv = clamp(dot(n, v), MX_FLOAT_EPS, 1.0);
    let f = mx_fresnel_schlick_exp(ndotv, color0, color90, exponent);
    return base * f;
}

// stdlib/genglsl/mx_rotate_vector3.glsl:1-13
fn mx_rotate_vector3(v: vec3<f32>, amount: f32, axis_in: vec3<f32>) -> vec3<f32> {
    let axis = normalize(axis_in);
    let rotation_radians = amount * MX_DEG_TO_RAD;
    let s = sin(rotation_radians);
    let c = cos(rotation_radians);
    let oc = 1.0 - c;
    return v * c + cross(v, axis) * s + axis * dot(axis, v) * oc;
}

// Anisotropy axis: tangent projected onto the plane of n (Duff basis if degenerate), then
// rotated by rotation * 360 degrees when anisotropy > 0 (standard_surface.mtlx:152-189).
fn ss_shading_tangent(n: vec3<f32>, t: vec3<f32>, rotation: f32, anisotropy: f32) -> vec3<f32> {
    let p = t - n * dot(t, n);
    let len2 = dot(p, p);
    var x: vec3<f32>;
    if (len2 < SS_TANGENT_MIN_LEN2) {
        x = mx_orthonormal_basis(n)[0];
    } else {
        x = normalize(p);
    }
    if (anisotropy > 0.0) {
        return normalize(mx_rotate_vector3(x, rotation * 360.0, n));
    }
    return x;
}

// Graph node main_roughness (standard_surface.mtlx:117-133).
fn ss_main_alpha(i: SsInputs) -> vec2<f32> {
    let coat_affect = i.coat_affect_roughness * i.coat * i.coat_roughness;
    let coat_affected_roughness = mix(i.specular_roughness, 1.0, coat_affect);
    return mx_roughness_anisotropy(coat_affected_roughness, i.specular_anisotropy);
}

// Graph node coat_roughness_vector (standard_surface.mtlx:345-348).
fn ss_coat_alpha(i: SsInputs) -> vec2<f32> {
    return mx_roughness_anisotropy(i.coat_roughness, i.coat_anisotropy);
}

// Graph node transmission_roughness (standard_surface.mtlx:135-150).
fn ss_transmission_alpha(i: SsInputs) -> vec2<f32> {
    let coat_affect = i.coat_affect_roughness * i.coat * i.coat_roughness;
    let roughness = clamp(i.specular_roughness + i.transmission_extra_roughness, 0.0, 1.0);
    let coat_affected_roughness = mix(roughness, 1.0, coat_affect);
    return mx_roughness_anisotropy(coat_affected_roughness, i.specular_anisotropy);
}

// Closure parameters and graph intermediates for one view direction; on the inside of a
// transmissive surface, the bare dielectric interface with relative IOR 1 / specular_IOR.
fn ss_layers(i: SsInputs, f: SsFrame, wo: vec3<f32>) -> SsLayers {
    let main = ss_main_alpha(i);
    let artistic = mx_artistic_ior(i.base_color * i.base, i.specular_color * i.specular);
    let coat_gamma = clamp(i.coat, 0.0, 1.0) * i.coat_affect_color + 1.0;
    var s = SsLayers(
        f.n,
        wo,
        ss_shading_tangent(f.n, f.tangent, i.specular_rotation, i.specular_anisotropy),
        ss_shading_tangent(f.n, f.tangent, i.coat_rotation, i.coat_anisotropy),
        MxDielectric(i.coat, vec3<f32>(1.0), i.coat_ior, ss_coat_alpha(i), 0.0, MX_THINFILM_IOR_DEFAULT, i.thin_film_energy),
        MxDielectric(
            i.specular,
            i.specular_color,
            i.specular_ior,
            main,
            i.thin_film_thickness,
            i.thin_film_ior,
            i.thin_film_energy,
        ),
        MxConductor(1.0, artistic.ior, artistic.extinction, main, i.thin_film_thickness, i.thin_film_ior, i.thin_film_energy),
        clamp(mix(vec3<f32>(1.0), i.coat_color, vec3<f32>(i.coat)), vec3<f32>(0.0), vec3<f32>(1.0)),
        i.metalness,
        pow(max(i.base_color, vec3<f32>(0.0)), vec3<f32>(coat_gamma)),
        MxDielectric(1.0, i.transmission_color, i.specular_ior, ss_transmission_alpha(i), 0.0, MX_THINFILM_IOR_DEFAULT, i.thin_film_energy),
        i.transmission,
        false,
        pow(max(i.subsurface_color, vec3<f32>(0.0)), vec3<f32>(coat_gamma)),
        i.subsurface_radius * i.subsurface_scale,
        i.subsurface,
    );
    let inside = f.inside && i.transmission > 0.0;
    if (inside) {
        let inverse_ior = 1.0 / i.specular_ior;
        s.coat.weight = 0.0;
        s.attenuation = vec3<f32>(1.0);
        s.metalness = 0.0;
        s.specular.weight = 1.0;
        s.specular.tint = vec3<f32>(1.0);
        s.specular.ior = inverse_ior;
        s.specular.roughness = s.transmission.roughness;
        s.specular.thinfilm_thickness = 0.0;
        s.transmission.ior = inverse_ior;
        s.transmission_mix = 1.0;
        s.inside = true;
    }
    return s;
}

// Graph node subsurface_mix for direct light (standard_surface.mtlx:252-256); exactly the
// Oren-Nayar closure when subsurface = 0 (the subsurface closure is then not evaluated).
fn ss_subsurface_mix_reflection(i: SsInputs, f: SsFrame, s: SsLayers, wo: vec3<f32>, wi: vec3<f32>) -> MxBsdf {
    let diffuse = mx_oren_nayar_diffuse_bsdf_reflection(
        wo,
        wi,
        s.n,
        i.base,
        s.diffuse_color,
        i.diffuse_roughness,
        false,
    );
    if (s.subsurface_mix <= 0.0) {
        return diffuse;
    }
    let subsurface = mx_subsurface_bsdf_reflection(
        wo,
        wi,
        s.n,
        f.curvature,
        1.0,
        s.subsurface_color,
        s.subsurface_radius,
    );
    return mx_mix_bsdf(subsurface, diffuse, s.subsurface_mix);
}

// Graph node subsurface_mix for environment light.
fn ss_subsurface_mix_indirect(i: SsInputs, s: SsLayers, wo: vec3<f32>, irradiance: vec3<f32>) -> MxBsdf {
    let diffuse = mx_oren_nayar_diffuse_bsdf_indirect(
        wo,
        s.n,
        i.base,
        s.diffuse_color,
        i.diffuse_roughness,
        false,
        irradiance,
    );
    if (s.subsurface_mix <= 0.0) {
        return diffuse;
    }
    let subsurface = mx_subsurface_bsdf_indirect(1.0, s.subsurface_color, irradiance);
    return mx_mix_bsdf(subsurface, diffuse, s.subsurface_mix);
}

// The Lobes partition from the five closure values (plan §2 "eval").
fn ss_combine(s: SsLayers, coat: MxBsdf, specular: MxBsdf, metal: MxBsdf, sheen: MxBsdf, diffuse: MxBsdf) -> SsLobes {
    let m = s.metalness;
    let under_coat = coat.throughput * s.attenuation;
    let dielectric_under = under_coat * (1.0 - m);
    return SsLobes(
        dielectric_under * specular.throughput * ((sheen.response + sheen.throughput * diffuse.response) * (1.0 - s.transmission_mix)),
        coat.response + under_coat * (specular.response * (1.0 - m) + metal.response * m),
        vec3<f32>(0.0),
    );
}

// Layer factors and tint above the transmission BTDF (inside: the bare tint); zero when
// transmission_mix = 0.
fn ss_transmission_tint(s: SsLayers, wo: vec3<f32>) -> vec3<f32> {
    if (s.transmission_mix <= 0.0) {
        return vec3<f32>(0.0);
    }
    if (s.inside) {
        return max(s.transmission.tint, vec3<f32>(0.0)) * s.transmission_mix;
    }
    let unit = vec3<f32>(1.0);
    let coat = mx_dielectric_bsdf_indirect(wo, s.n, s.coat, unit);
    let specular = mx_dielectric_bsdf_indirect(wo, s.n, s.specular, unit);
    let under_coat = coat.throughput * s.attenuation;
    let dielectric_under = under_coat * (1.0 - s.metalness);
    return dielectric_under * specular.throughput * max(s.transmission.tint, vec3<f32>(0.0)) * s.transmission_mix;
}

// Direct light with GGX NDFs widened by ndf_widen; behind ss_eval_light and ss_eval_light_disc.
fn ss_eval_light_widened(i: SsInputs, f: SsFrame, wo: vec3<f32>, wi: vec3<f32>, ndf_widen: f32, split_delta: bool) -> SsLobes {
    // Hemisphere rule (plan §2, I1): the reflection lobes need both directions above the
    // horizon; otherwise only the transmission lobe can respond.
    let above = dot(f.n, wo) > 0.0 && dot(f.n, wi) > 0.0;
    if (!above) {
        return ss_eval_transmission(i, f, wo, wi, ndf_widen, split_delta);
    }
    let s = ss_layers(i, f, wo);
    var coat = mx_dielectric_bsdf_reflection(wo, wi, s.n, s.coat_tangent, s.coat, ndf_widen);
    var specular = mx_dielectric_bsdf_reflection(wo, wi, s.n, s.main_tangent, s.specular, ndf_widen);
    var metal = mx_conductor_bsdf_reflection(wo, wi, s.n, s.main_tangent, s.metal, ndf_widen);
    let sheen = mx_sheen_bsdf_reflection(wo, wi, s.n, i.sheen, i.sheen_color, i.sheen_roughness);
    let diffuse = ss_subsurface_mix_reflection(i, f, s, wo, wi);
    // Delta lobes (plan §2, C2): zero response, throughput kept.
    if (split_delta && ss_is_delta(ss_ndf_alpha(s.coat.roughness, ndf_widen))) {
        coat.response = vec3<f32>(0.0);
    }
    if (split_delta && ss_is_delta(ss_ndf_alpha(s.specular.roughness, ndf_widen))) {
        specular.response = vec3<f32>(0.0);
    }
    if (split_delta && ss_is_delta(ss_ndf_alpha(s.metal.roughness, ndf_widen))) {
        metal.response = vec3<f32>(0.0);
    }
    return ss_combine(s, coat, specular, metal, sheen, diffuse);
}

// Dielectric Fresnel F(n.wo) of the specular closure's IOR (the bare interface inside).
fn ss_interface_fresnel(s: SsLayers, wo: vec3<f32>) -> f32 {
    let ndotv = clamp(dot(s.n, wo), MX_FLOAT_EPS, 1.0);
    return mx_fresnel_dielectric(ndotv, s.specular.ior);
}

// Selection share of the inside interface's reflection: mix(F(n.wo), 1/2, sqrt(avg alpha)).
fn ss_interface_reflect_share(s: SsLayers, wo: vec3<f32>) -> f32 {
    let roughness = sqrt(mx_average_alpha(s.transmission.roughness));
    return mix(ss_interface_fresnel(s, wo), 0.5, roughness);
}

// Energy the transmission lobe passes at wo with the BTDF albedo taken as 1.
fn ss_transmission_albedo(s: SsLayers, wo: vec3<f32>) -> vec3<f32> {
    let tint = ss_transmission_tint(s, wo);
    if (s.inside) {
        return tint * (1.0 - ss_interface_fresnel(s, wo));
    }
    return tint;
}

// The transmission part of direct light: wo above, wi below the horizon, transmission > 0.
fn ss_eval_transmission(i: SsInputs, f: SsFrame, wo: vec3<f32>, wi: vec3<f32>, ndf_widen: f32, split_delta: bool) -> SsLobes {
    let zero = SsLobes(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    let transmits = dot(f.n, wo) > 0.0 && dot(f.n, wi) < 0.0 && i.transmission > 0.0;
    if (!transmits) {
        return zero;
    }
    let s = ss_layers(i, f, wo);
    let eta = s.transmission.ior;
    let delta = ss_transmission_is_delta(ss_ndf_alpha(s.transmission.roughness, ndf_widen), eta);
    if ((split_delta && delta) || eta == 1.0) {
        return zero;
    }
    let t = ss_ggx_transmission(wo, wi, s.n, s.main_tangent, s.transmission, ndf_widen, s.inside);
    return SsLobes(vec3<f32>(0.0), vec3<f32>(0.0), ss_transmission_tint(s, wo) * t);
}

// f(wo, wi) cos(n, wi) for a point or directional light.
fn ss_eval_light(i: SsInputs, f: SsFrame, wo: vec3<f32>, wi: vec3<f32>) -> SsLobes {
    return ss_eval_light_widened(i, f, wo, wi, 0.0, true);
}

// MaterialX's raster behaviour: clamped alpha, no delta split (Rust `eval_light_materialx`).
fn ss_eval_light_materialx(i: SsInputs, f: SsFrame, wo: vec3<f32>, wi: vec3<f32>) -> SsLobes {
    return ss_eval_light_widened(i, f, wo, wi, 0.0, false);
}

// Disc light of angular radius half_angle evaluated at the disc centre (plan §4).
fn ss_eval_light_disc(i: SsInputs, f: SsFrame, wo: vec3<f32>, wi: vec3<f32>, half_angle: f32) -> SsLobes {
    return ss_eval_light_widened(i, f, wo, wi, tan(half_angle) * 0.5, true);
}

// Pre-integrated environment light (CLOSURE_TYPE_INDIRECT path).
fn ss_eval_environment(i: SsInputs, f: SsFrame, wo: vec3<f32>, env: SsEnvironment) -> SsLobes {
    let s = ss_layers(i, f, wo);
    let coat = mx_dielectric_bsdf_indirect(wo, s.n, s.coat, env.coat_radiance);
    let specular = mx_dielectric_bsdf_indirect(wo, s.n, s.specular, env.specular_radiance);
    let metal = mx_conductor_bsdf_indirect(wo, s.n, s.metal, env.specular_radiance);
    let sheen = mx_sheen_bsdf_indirect(wo, s.n, i.sheen, i.sheen_color, i.sheen_roughness, env.irradiance);
    let diffuse = ss_subsurface_mix_indirect(i, s, wo, env.irradiance);
    var lobes = ss_combine(s, coat, specular, metal, sheen, diffuse);
    lobes.transmission = ss_transmission_albedo(s, wo) * env.transmission_radiance;
    return lobes;
}

// Emission EDF: mix(E, coat-tinted generalized Schlick EDF, coat) (standard_surface.mtlx:365-409).
fn ss_eval_emission(i: SsInputs, f: SsFrame, wo: vec3<f32>) -> vec3<f32> {
    let emission = i.emission_color * i.emission;
    let coat_f0 = mx_ior_to_f0(i.coat_ior);
    let coat_emission = mx_generalized_schlick_edf(
        wo,
        f.n,
        vec3<f32>(1.0 - coat_f0),
        vec3<f32>(0.0),
        MX_COAT_EMISSION_EXPONENT,
        emission * i.coat_color,
    );
    return mix(emission, coat_emission, vec3<f32>(i.coat));
}

// Luminance with the MaterialX default (ACEScg) coefficients.
fn ss_luminance(c: vec3<f32>) -> f32 {
    return dot(c, SS_LUMA);
}

// Normalised lobe-selection probabilities at wo, indexed by SS_LOBE_*.
fn ss_lobe_weights(i: SsInputs, f: SsFrame, wo: vec3<f32>) -> array<f32, SS_LOBE_COUNT> {
    let s = ss_layers(i, f, wo);
    let unit = vec3<f32>(1.0);
    let coat = mx_dielectric_bsdf_indirect(wo, s.n, s.coat, unit);
    let specular = mx_dielectric_bsdf_indirect(wo, s.n, s.specular, unit);
    let metal = mx_conductor_bsdf_indirect(wo, s.n, s.metal, unit);
    let sheen = mx_sheen_bsdf_indirect(wo, s.n, i.sheen, i.sheen_color, i.sheen_roughness, unit);
    let diffuse = ss_subsurface_mix_indirect(i, s, wo, unit);
    let m = s.metalness;
    let under_coat = coat.throughput * s.attenuation;
    let dielectric_under = under_coat * (1.0 - m);
    let under_specular = dielectric_under * specular.throughput * (1.0 - s.transmission_mix);

    let p_coat = max(ss_luminance(unit - coat.throughput), 0.0);
    var p_specular = max(ss_luminance(dielectric_under * (unit - specular.throughput)), 0.0);
    if (s.inside) {
        p_specular = ss_interface_reflect_share(s, wo);
    }
    let p_metal = max(ss_luminance(under_coat * m * metal.response), 0.0);
    let p_sheen = max(ss_luminance(under_specular * (unit - sheen.throughput)), 0.0);
    let p_diffuse = max(ss_luminance(under_specular * sheen.throughput * diffuse.response), 0.0);
    var p_transmission = max(ss_luminance(ss_transmission_albedo(s, wo)), 0.0);
    if (s.inside) {
        p_transmission = max(ss_luminance(max(s.transmission.tint, vec3<f32>(0.0))), 0.0) * (1.0 - ss_interface_reflect_share(s, wo));
    }

    let total = p_coat + p_specular + p_metal + p_sheen + p_diffuse + p_transmission;
    if (total > 0.0) {
        return array<f32, SS_LOBE_COUNT>(
            p_coat / total,
            p_specular / total,
            p_metal / total,
            p_sheen / total,
            p_diffuse / total,
            p_transmission / total,
        );
    }
    return array<f32, SS_LOBE_COUNT>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

// Dominant direction of a GGX lobe (Lagarde & de Rousiers 2014, Frostbite §4.9.3).
fn ss_dominant_dir(f: SsFrame, wo: vec3<f32>, alpha: f32) -> vec3<f32> {
    let r = reflect(wo * -1.0, f.n);
    let smoothness = 1.0 - alpha;
    let lerp_factor = smoothness * (sqrt(smoothness) + alpha);
    return normalize(mix(f.n, r, vec3<f32>(lerp_factor)));
}
