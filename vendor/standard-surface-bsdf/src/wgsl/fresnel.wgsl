// standard-surface-bsdf: WGSL twin of src/fresnel.rs, statement for statement.
// MaterialX v1.39.5-22-g47cecce6: pbrlib/genglsl/lib/mx_microfacet_specular.glsl,
// pbrlib/genglsl/mx_artistic_ior.glsl.

// lib/mx_microfacet_specular.glsl:8-30 (without `refraction`).
struct MxFresnelData {
    model: u32,
    airy: bool,
    ior: vec3<f32>,
    extinction: vec3<f32>,
    f0: vec3<f32>,
    f82: vec3<f32>,
    f90: vec3<f32>,
    exponent: f32,
    tf_thickness: f32,
    tf_ior: f32,
}

// Parallel (p) and perpendicular (s) pair: the `out vec3` parameters of
// lib/mx_microfacet_specular.glsl:246,273.
struct MxPolarized {
    p: vec3<f32>,
    s: vec3<f32>,
}

// Output of mx_artistic_ior.
struct MxArtisticIor {
    ior: vec3<f32>,
    extinction: vec3<f32>,
}

// lib/mx_microfacet_specular.glsl:183-186
fn mx_ior_to_f0(ior: f32) -> f32 {
    return mx_square((ior - 1.0) / (ior + 1.0));
}

// lib/mx_microfacet_specular.glsl:194-198
fn mx_f0_to_ior_vec3(f0: vec3<f32>) -> vec3<f32> {
    let sqrt_f0 = sqrt(clamp(f0, vec3<f32>(MX_F0_TO_IOR_MIN), vec3<f32>(MX_F0_TO_IOR_MAX)));
    return (vec3<f32>(1.0) + sqrt_f0) / (vec3<f32>(1.0) - sqrt_f0);
}

// pbrlib/genglsl/lib/mx_microfacet.glsl:49-53
fn mx_fresnel_schlick_exp(cos_theta: f32, f0: vec3<f32>, f90: vec3<f32>, exponent: f32) -> vec3<f32> {
    let x = clamp(1.0 - cos_theta, 0.0, 1.0);
    return mix(f0, f90, vec3<f32>(pow(x, exponent)));
}

// lib/mx_microfacet_specular.glsl:200-209
fn mx_fresnel_hoffman_schlick(cos_theta: f32, fd: MxFresnelData) -> vec3<f32> {
    let x = clamp(cos_theta, 0.0, 1.0);
    let a = mix(fd.f0, fd.f90, vec3<f32>(pow(1.0 - MX_COS_THETA_MAX, fd.exponent)))
        * (vec3<f32>(1.0) - fd.f82) * MX_COS_THETA_FACTOR;
    return mix(fd.f0, fd.f90, vec3<f32>(pow(1.0 - x, fd.exponent))) - a * x * mx_pow6(1.0 - x);
}

// lib/mx_microfacet_specular.glsl:211-225
fn mx_fresnel_dielectric(cos_theta: f32, ior: f32) -> f32 {
    let c = cos_theta;
    let g2 = ior * ior + c * c - 1.0;
    if (g2 < 0.0) {
        // Total internal reflection.
        return 1.0;
    }
    let g = sqrt(g2);
    return 0.5 * mx_square((g - c) / (g + c))
        * (1.0 + mx_square(((g + c) * c - 1.0) / ((g - c) * c + 1.0)));
}

// lib/mx_microfacet_specular.glsl:227-243; returns (Rp, Rs).
fn mx_fresnel_dielectric_polarized(cos_theta: f32, ior: f32) -> vec2<f32> {
    let cos_theta2 = mx_square(clamp(cos_theta, 0.0, 1.0));
    let sin_theta2 = 1.0 - cos_theta2;

    let t0 = max(ior * ior - sin_theta2, 0.0);
    let t1 = t0 + cos_theta2;
    let t2 = 2.0 * sqrt(t0) * cos_theta;
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = cos_theta2 * t0 + sin_theta2 * sin_theta2;
    let t4 = t2 * sin_theta2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    return vec2<f32>(rp, rs);
}

// lib/mx_microfacet_specular.glsl:245-263
fn mx_fresnel_conductor_polarized(cos_theta: f32, n: vec3<f32>, k: vec3<f32>) -> MxPolarized {
    let cos_theta2 = mx_square(clamp(cos_theta, 0.0, 1.0));
    let sin_theta2 = 1.0 - cos_theta2;
    let n2 = n * n;
    let k2 = k * k;

    let t0 = n2 - k2 - vec3<f32>(sin_theta2);
    let a2plusb2 = sqrt(t0 * t0 + n2 * 4.0 * k2);
    let t1 = a2plusb2 + vec3<f32>(cos_theta2);
    let a = sqrt(max((a2plusb2 + t0) * 0.5, vec3<f32>(0.0)));
    let t2 = a * 2.0 * cos_theta;
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = a2plusb2 * cos_theta2 + vec3<f32>(sin_theta2 * sin_theta2);
    let t4 = t2 * sin_theta2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    return MxPolarized(rp, rs);
}

// lib/mx_microfacet_specular.glsl:265-270
fn mx_fresnel_conductor(cos_theta: f32, n: vec3<f32>, k: vec3<f32>) -> vec3<f32> {
    let r = mx_fresnel_conductor_polarized(cos_theta, n, k);
    return (r.p + r.s) * 0.5;
}

// lib/mx_microfacet_specular.glsl:272-285
fn mx_fresnel_conductor_phase_polarized(cos_theta: f32, eta1: f32, eta2: vec3<f32>, kappa2: vec3<f32>) -> MxPolarized {
    let k2 = kappa2 / eta2;
    let sin_theta_sqr = vec3<f32>(1.0) - vec3<f32>(cos_theta * cos_theta);
    let one_minus_k2k2 = vec3<f32>(1.0) - k2 * k2;
    let a = eta2 * eta2 * one_minus_k2k2 - sin_theta_sqr * (eta1 * eta1);
    let two_eta2_eta2_k2 = eta2 * 2.0 * eta2 * k2;
    let b = sqrt(a * a + two_eta2_eta2_k2 * two_eta2_eta2_k2);
    let u = sqrt((a + b) / 2.0);
    let v = max(sqrt((b - a) / 2.0), vec3<f32>(0.0));

    let uu_vv = u * u + v * v;
    let phi_s = atan2(v * (2.0 * eta1) * cos_theta, uu_vv - vec3<f32>(mx_square(eta1 * cos_theta)));
    let one_plus_k2k2 = vec3<f32>(1.0) + k2 * k2;
    let phi_p_y = eta2 * (2.0 * eta1) * eta2 * cos_theta * (k2 * 2.0 * u - one_minus_k2k2 * v);
    let phi_p_x_base = eta2 * eta2 * one_plus_k2k2 * cos_theta;
    let phi_p = atan2(phi_p_y, phi_p_x_base * phi_p_x_base - uu_vv * (eta1 * eta1));
    return MxPolarized(phi_p, phi_s);
}

// Plan §2 "Airy phase": x - 2 pi floor(x / (2 pi) + 0.5); floor(x + 0.5) on both twins.
fn ss_reduce_phase(x: f32) -> f32 {
    return x - MX_TWO_PI * floor(x * SS_INV_TWO_PI + 0.5);
}

// lib/mx_microfacet_specular.glsl:287-298, cosine arguments reduced by ss_reduce_phase.
fn ss_airy_gaussian(val: f32, pos: f32, var_: f32, phase: f32, shift: f32) -> f32 {
    return val * sqrt(MX_TWO_PI * var_)
        * cos(ss_reduce_phase(pos * phase + shift))
        * exp(-var_ * phase * phase);
}

// lib/mx_microfacet_specular.glsl:287-298
fn mx_eval_sensitivity(opd: f32, shift: vec3<f32>) -> vec3<f32> {
    // Gaussian fits, given by 3 parameters: val, pos and var.
    let phase = MX_TWO_PI * opd;
    var xyz = vec3<f32>(
        ss_airy_gaussian(MX_AIRY_VAL.x, MX_AIRY_POS.x, MX_AIRY_VAR.x, phase, shift.x),
        ss_airy_gaussian(MX_AIRY_VAL.y, MX_AIRY_POS.y, MX_AIRY_VAR.y, phase, shift.y),
        ss_airy_gaussian(MX_AIRY_VAL.z, MX_AIRY_POS.z, MX_AIRY_VAR.z, phase, shift.z),
    );
    xyz = vec3<f32>(
        xyz.x + ss_airy_gaussian(MX_AIRY_X2_VAL, MX_AIRY_X2_POS, MX_AIRY_X2_VAR, phase, shift.x),
        xyz.y,
        xyz.z,
    );
    return xyz / MX_AIRY_NORM;
}

// lib/mx_microfacet_specular.glsl:300-399
fn mx_fresnel_airy(cos_theta: f32, fd: MxFresnelData) -> vec3<f32> {
    // Assume vacuum on the outside.
    let eta1 = 1.0;
    let eta2 = max(fd.tf_ior, eta1);
    let schlick = fd.model == MX_FRESNEL_MODEL_SCHLICK;
    var eta3 = fd.ior;
    var kappa3 = fd.extinction;
    if (schlick) {
        eta3 = mx_f0_to_ior_vec3(fd.f0);
        kappa3 = vec3<f32>(0.0);
    }
    let cos_theta_t = sqrt(1.0 - (1.0 - mx_square(cos_theta)) * mx_square(eta1 / eta2));

    // First interface.
    var r12 = mx_fresnel_dielectric_polarized(cos_theta, eta2 / eta1);
    if (cos_theta_t <= 0.0) {
        // Total internal reflection.
        r12 = vec2<f32>(1.0, 1.0);
    }
    let r12_p = r12.x;
    let r12_s = r12.y;
    let t121_p = 1.0 - r12_p;
    let t121_s = 1.0 - r12_s;

    // Second interface.
    var r23: MxPolarized;
    if (schlick) {
        let f = mx_fresnel_hoffman_schlick(cos_theta_t, fd);
        r23 = MxPolarized(f * 0.5, f * 0.5);
    } else {
        r23 = mx_fresnel_conductor_polarized(cos_theta_t, eta3 / eta2, kappa3 / eta2);
    }

    // Phase shift.
    let cos_b = cos(atan(eta2 / eta1));
    let phi21_p = select(MX_PI, 0.0, cos_theta < cos_b);
    let phi21_s = MX_PI;
    var phi23: MxPolarized;
    if (schlick) {
        let p = vec3<f32>(
            select(0.0, MX_PI, eta3.x < eta2),
            select(0.0, MX_PI, eta3.y < eta2),
            select(0.0, MX_PI, eta3.z < eta2),
        );
        phi23 = MxPolarized(p, p);
    } else {
        phi23 = mx_fresnel_conductor_phase_polarized(cos_theta_t, eta2, eta3, kappa3);
    }
    let r123p = max(sqrt(r23.p * r12_p), vec3<f32>(0.0));
    let r123s = max(sqrt(r23.s * r12_s), vec3<f32>(0.0));

    // Iridescence term.
    var i = vec3<f32>(0.0);

    // Optical path difference.
    let dist_meters = fd.tf_thickness * MX_AIRY_NM_TO_M;
    let opd = 2.0 * eta2 * cos_theta_t * dist_meters;

    // Parallel polarisation: reflectance term for m = 0 (DC term amplitude).
    let rs = (r23.p * mx_square(t121_p)) / (vec3<f32>(1.0) - r23.p * r12_p);
    i = i + (vec3<f32>(r12_p) + rs);

    // Reflectance terms for m > 0 (pairs of diracs).
    var cm = rs - vec3<f32>(t121_p);
    for (var m: i32 = 1; m <= MX_AIRY_FRESNEL_ITERATIONS; m = m + 1) {
        let mf = f32(m);
        cm = cm * r123p;
        let sm = mx_eval_sensitivity(mf * opd, (phi23.p + vec3<f32>(phi21_p)) * mf) * 2.0;
        i = i + cm * sm;
    }

    // Perpendicular polarisation: reflectance term for m = 0 (DC term amplitude).
    let rp = (r23.s * mx_square(t121_s)) / (vec3<f32>(1.0) - r23.s * r12_s);
    i = i + (vec3<f32>(r12_s) + rp);

    // Reflectance terms for m > 0 (pairs of diracs).
    cm = rp - vec3<f32>(t121_s);
    for (var m: i32 = 1; m <= MX_AIRY_FRESNEL_ITERATIONS; m = m + 1) {
        let mf = f32(m);
        cm = cm * r123s;
        let sm = mx_eval_sensitivity(mf * opd, (phi23.s + vec3<f32>(phi21_s)) * mf) * 2.0;
        i = i + cm * sm;
    }

    // Average parallel and perpendicular polarisation.
    i = i * 0.5;

    // Convert back to RGB reflectance.
    return clamp(
        vec3<f32>(dot(MX_XYZ_TO_RGB_R0, i), dot(MX_XYZ_TO_RGB_R1, i), dot(MX_XYZ_TO_RGB_R2, i)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

// lib/mx_microfacet_specular.glsl:401-416
fn mx_init_fresnel_dielectric(ior: f32, tf_thickness: f32, tf_ior: f32) -> MxFresnelData {
    return MxFresnelData(
        MX_FRESNEL_MODEL_DIELECTRIC,
        tf_thickness > 0.0,
        vec3<f32>(ior),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        0.0,
        tf_thickness,
        tf_ior,
    );
}

// lib/mx_microfacet_specular.glsl:418-433
fn mx_init_fresnel_conductor(ior: vec3<f32>, extinction: vec3<f32>, tf_thickness: f32, tf_ior: f32) -> MxFresnelData {
    return MxFresnelData(
        MX_FRESNEL_MODEL_CONDUCTOR,
        tf_thickness > 0.0,
        ior,
        extinction,
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        0.0,
        tf_thickness,
        tf_ior,
    );
}

// lib/mx_microfacet_specular.glsl:435-450
fn mx_init_fresnel_schlick(f0: vec3<f32>, f82: vec3<f32>, f90: vec3<f32>, exponent: f32, tf_thickness: f32, tf_ior: f32) -> MxFresnelData {
    return MxFresnelData(
        MX_FRESNEL_MODEL_SCHLICK,
        tf_thickness > 0.0,
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        f0,
        f82,
        f90,
        exponent,
        tf_thickness,
        tf_ior,
    );
}

// lib/mx_microfacet_specular.glsl:452-470
fn mx_compute_fresnel(cos_theta: f32, fd: MxFresnelData) -> vec3<f32> {
    if (fd.airy) {
        return mx_fresnel_airy(cos_theta, fd);
    } else if (fd.model == MX_FRESNEL_MODEL_DIELECTRIC) {
        return vec3<f32>(mx_fresnel_dielectric(cos_theta, fd.ior.x));
    } else if (fd.model == MX_FRESNEL_MODEL_CONDUCTOR) {
        return mx_fresnel_conductor(cos_theta, fd.ior, fd.extinction);
    }
    return mx_fresnel_hoffman_schlick(cos_theta, fd);
}

// lib/mx_microfacet_specular.glsl:472-499
fn mx_ggx_dir_albedo_fd(ndotv: f32, alpha: f32, fd: MxFresnelData) -> vec3<f32> {
    if (fd.airy) {
        // Approximation using a blend between mirror (alpha = 0) and rougher cases.
        let mirror_dir_albedo = mx_compute_fresnel(ndotv, fd);
        let f0 = mx_fresnel_airy(1.0, fd);
        let rough_dir_albedo = mx_ggx_dir_albedo_analytic(ndotv, alpha, f0, vec3<f32>(1.0));
        return mix(mirror_dir_albedo, rough_dir_albedo, vec3<f32>(sqrt(alpha)));
    } else if (fd.model == MX_FRESNEL_MODEL_DIELECTRIC) {
        let f0 = mx_ior_to_f0(fd.ior.x);
        return mx_ggx_dir_albedo_analytic(ndotv, alpha, vec3<f32>(f0), vec3<f32>(1.0));
    } else if (fd.model == MX_FRESNEL_MODEL_CONDUCTOR) {
        let f0 = mx_fresnel_conductor(1.0, fd.ior, fd.extinction);
        return mx_ggx_dir_albedo_analytic(ndotv, alpha, f0, vec3<f32>(1.0));
    }
    return mx_ggx_dir_albedo_analytic(ndotv, alpha, fd.f0, fd.f90);
}

// FG of a GGX lobe per thin-film energy model (Rust `film_dir_albedo`, tasks R2b/R2c).
// Conserving with a film weights the mirror term by the single-scatter energy Ess.
fn ss_film_dir_albedo(ndotv: f32, alpha: f32, fd: MxFresnelData, energy: u32) -> vec3<f32> {
    if (energy == SS_THIN_FILM_ENERGY_CONSERVING && fd.airy) {
        let ess = mx_ggx_dir_albedo_scalar(ndotv, alpha, 1.0, 1.0);
        let mirror_dir_albedo = mx_compute_fresnel(ndotv, fd) * ess;
        let f0 = mx_fresnel_airy(1.0, fd);
        let rough_dir_albedo = mx_ggx_dir_albedo_analytic(ndotv, alpha, f0, vec3<f32>(1.0));
        return mix(mirror_dir_albedo, rough_dir_albedo, vec3<f32>(sqrt(alpha)));
    }
    return mx_ggx_dir_albedo_fd(ndotv, alpha, fd);
}

// lib/mx_microfacet_specular.glsl:501-511
fn mx_fresnel_average(fd: MxFresnelData) -> vec3<f32> {
    let f0 = mx_compute_fresnel(1.0, fd);
    var f90 = vec3<f32>(1.0);
    if (fd.model == MX_FRESNEL_MODEL_SCHLICK && !fd.airy) {
        f90 = fd.f90;
    }
    return f0 + (f90 - f0) * MX_FRESNEL_AVERAGE_FACTOR;
}

// lib/mx_microfacet_specular.glsl:513-521
fn mx_ggx_energy_compensation(ndotv: f32, alpha: f32, fd: MxFresnelData) -> vec3<f32> {
    let fss = mx_fresnel_average(fd);
    let ess = mx_ggx_dir_albedo_scalar(ndotv, alpha, 1.0, 1.0);
    return vec3<f32>(1.0) + fss * (1.0 - ess) / ess;
}

// pbrlib/genglsl/mx_artistic_ior.glsl:1-17
fn mx_artistic_ior(reflectivity: vec3<f32>, edge_color: vec3<f32>) -> MxArtisticIor {
    let r = clamp(reflectivity, vec3<f32>(0.0), vec3<f32>(MX_F0_TO_IOR_MAX));
    let r_sqrt = sqrt(r);
    let n_min = (vec3<f32>(1.0) - r) / (vec3<f32>(1.0) + r);
    let n_max = (vec3<f32>(1.0) + r_sqrt) / (vec3<f32>(1.0) - r_sqrt);
    let ior = mix(n_max, n_min, edge_color);

    let np1 = ior + vec3<f32>(1.0);
    let nm1 = ior - vec3<f32>(1.0);
    var k2 = (np1 * np1 * r - nm1 * nm1) / (vec3<f32>(1.0) - r);
    k2 = max(k2, vec3<f32>(0.0));
    return MxArtisticIor(ior, sqrt(k2));
}
