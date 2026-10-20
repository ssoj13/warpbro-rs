// =============================================================================
// Extended 3D Fractals & Procedural Distance Estimators (Families 8..27)
// All parameters read from the per-object formula slot block starting at P_BULB_POWER (39).
// =============================================================================

const FAMILY_AMAZING_SURFACE: u32 = 8u;
const FAMILY_BRISTORBROT: u32 = 9u;
const FAMILY_BURNING_SHIP: u32 = 10u;
const FAMILY_KALEIDO_TWIST: u32 = 11u;
const FAMILY_POLYHEDRAL_KIFS: u32 = 12u;
const FAMILY_HARMONIC_BULB: u32 = 13u;
const FAMILY_MANDELBAR: u32 = 14u;
const FAMILY_MOBIUS_IFS: u32 = 15u;
const FAMILY_GYROID_TPMS: u32 = 16u;
const FAMILY_LIDINOID_TPMS: u32 = 17u;
const FAMILY_ATTRACTOR_TUBE: u32 = 18u;
const FAMILY_TORUS_KNOT: u32 = 19u;
const FAMILY_HOPF_FIBRATION: u32 = 20u;
const FAMILY_DUPIN_CYCLIDE: u32 = 21u;
const FAMILY_SUPERFORMULA: u32 = 22u;
const FAMILY_PHYLLOTAXIS: u32 = 23u;
const FAMILY_NAUTILUS: u32 = 24u;
const FAMILY_VORONOI_FOAM: u32 = 25u;
const FAMILY_CRYSTAL_GEODE: u32 = 26u;
const FAMILY_ASTROLABE: u32 = 27u;

fn ext_slot(is_world: bool, obj: u32, offset: u32) -> f32 {
    return pr(is_world, obj, P_BULB_POWER + offset);
}

fn ext_slot3(is_world: bool, obj: u32, offset: u32) -> vec3<f32> {
    return pr3(is_world, obj, P_BULB_POWER + offset);
}

fn ext_signed_family(family: u32) -> bool {
    return family == FAMILY_KIFS
        || family == FAMILY_KLEINIAN
        || family == FAMILY_PSEUDO_KLEINIAN
        || family == FAMILY_AMAZING_SURFACE
        || family == FAMILY_KALEIDO_TWIST
        || family == FAMILY_POLYHEDRAL_KIFS
        || family >= FAMILY_MOBIUS_IFS;
}

// 1. Amazing Surface (KaliBox)
fn amazing_surface_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let oc = orbit_constant(is_world, obj, q);
    let c = oc.xyz;
    let dr_c = oc.w;
    let scale = ext_slot(is_world, obj, 0u);
    let min_r2 = max(ext_slot(is_world, obj, 1u), 1.0e-6);
    let fixed_r2 = max(ext_slot(is_world, obj, 2u), min_r2);
    let fold_lim = ext_slot3(is_world, obj, 3u);
    let shift = ext_slot3(is_world, obj, 6u);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);

    var z = q;
    var dr = 1.0;
    var trap = TRAP_START;

    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, (length(z) - 0.35) / max(dr, 1.0e-30), trap);
        }
        z = apply_iter_rot(rot, z);
        if fold_lim.x > 0.0 {
            z.x = clamp(z.x, -fold_lim.x, fold_lim.x) * 2.0 - z.x;
        }
        if fold_lim.y > 0.0 {
            z.y = clamp(z.y, -fold_lim.y, fold_lim.y) * 2.0 - z.y;
        }
        if fold_lim.z > 0.0 {
            z.z = clamp(z.z, -fold_lim.z, fold_lim.z) * 2.0 - z.z;
        }
        z = z + shift;
        let r2 = dot(z, z);
        var factor = 1.0;
        if r2 < min_r2 {
            factor = fixed_r2 / min_r2;
        } else if r2 < fixed_r2 {
            factor = fixed_r2 / r2;
        }
        let step_s = scale * factor;
        let abs_s = max(abs(step_s), 1.0e-6);
        if dr > DERIVATIVE_LIMIT / abs_s {
            break;
        }
        z = z * step_s + c;
        dr = dr * abs_s + dr_c;
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    let est = (length(z) - 0.35) / max(dr, 1.0e-30);
    return fraction_blend(frac, est, trap);
}

// 2. Bristorbrot
fn bristorbrot_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let oc = orbit_constant(is_world, obj, q);
    let c = oc.xyz;
    let dr_c = oc.w;
    let scale = ext_slot(is_world, obj, 0u);
    let y_sign = ext_slot(is_world, obj, 1u);
    let z_sign = ext_slot(is_world, obj, 2u);
    let twist = ext_slot(is_world, obj, 3u);
    let bailout = pr(is_world, obj, P_BAILOUT);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);

    var z = q;
    var dr = 1.0;
    var r = length(z);
    var trap = TRAP_START;

    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, bulb_estimate(r, dr, bailout), trap);
        }
        if r > bailout {
            break;
        }
        if r < 1.0e-8 {
            return vec2<f32>(0.0, trap);
        }
        let growth = max(2.0 * abs(scale) * r, 1.0e-6);
        if dr > DERIVATIVE_LIMIT / growth {
            break;
        }
        var t = apply_iter_rot(rot, z);
        if twist != 0.0 {
            let ang = twist * t.x;
            let sa = sin(ang);
            let ca = cos(ang);
            t = vec3<f32>(t.x, t.y * ca - t.z * sa, t.y * sa + t.z * ca);
        }
        let nx = t.x * t.x - t.y * t.y - t.z * t.z;
        let ny = t.y * (2.0 * t.x + y_sign * t.z);
        let nz = t.z * (2.0 * t.x + z_sign * t.y);
        dr = growth * dr + dr_c;
        z = vec3<f32>(nx, ny, nz) * scale + c;
        r = length(z);
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, bulb_estimate(r, dr, bailout), trap);
}

// 3. Burning Ship 3D
fn burning_ship_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let oc = orbit_constant(is_world, obj, q);
    let c = oc.xyz;
    let dr_c = oc.w;
    let power = max(ext_slot(is_world, obj, 0u), 1.2);
    let abs_axes = clamp(ext_slot3(is_world, obj, 1u), vec3<f32>(0.0), vec3<f32>(1.0));
    let shift = ext_slot3(is_world, obj, 4u);
    let bailout = pr(is_world, obj, P_BAILOUT);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);

    var z = q;
    var dr = 1.0;
    var r = length(z);
    var trap = TRAP_START;

    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, bulb_estimate(r, dr, bailout), trap);
        }
        if r > bailout {
            break;
        }
        if r < 1.0e-8 {
            return vec2<f32>(0.0, trap);
        }
        var t = mix(z, abs(z), abs_axes) + shift;
        t = apply_iter_rot(rot, t);
        let rt = max(length(t), 1.0e-8);
        let r_pow_m1 = pow(rt, power - 1.0);
        let growth = power * r_pow_m1;
        if dr > DERIVATIVE_LIMIT / growth {
            break;
        }
        let theta = acos(clamp(t.z / rt, -1.0, 1.0)) * power;
        let phi = atan2(t.y, t.x) * power;
        let mag = r_pow_m1 * rt;
        dr = growth * dr + dr_c;
        let st = sin(theta);
        z = vec3<f32>(st * cos(phi), st * sin(phi), cos(theta)) * mag + c;
        r = length(z);
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, bulb_estimate(r, dr, bailout), trap);
}

// 4. Kaleido Twist IFS (Marble Marcher)
fn kaleido_twist_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let scale = max(ext_slot(is_world, obj, 0u), 1.05);
    let shift = ext_slot3(is_world, obj, 1u);
    let clamp_z = ext_slot(is_world, obj, 4u);
    let post_en = ext_slot(is_world, obj, 5u) != 0.0;
    let post_r0 = ext_slot3(is_world, obj, 6u);
    let post_r1 = ext_slot3(is_world, obj, 9u);
    let post_r2 = ext_slot3(is_world, obj, 12u);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);

    var z = q;
    var dr = 1.0;
    var trap = TRAP_START;

    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, unit_box_distance(z) / max(dr, 1.0e-30), trap);
        }
        if dr > DERIVATIVE_LIMIT / scale {
            break;
        }
        z = abs(z);
        z = apply_iter_rot(rot, z);
        if z.x < z.y {
            z = vec3<f32>(z.y, z.x, z.z);
        }
        if z.x < z.z {
            z = vec3<f32>(z.z, z.y, z.x);
        }
        if z.y < z.z {
            z = vec3<f32>(z.x, z.z, z.y);
        }
        z.x = scale * z.x - shift.x;
        z.y = scale * z.y - shift.y;
        z.z = scale * z.z;
        if z.z > 0.5 * shift.z * clamp_z {
            z.z = z.z - shift.z;
        }
        if post_en {
            z = vec3<f32>(dot(post_r0, z), dot(post_r1, z), dot(post_r2, z));
        }
        dr = dr * scale;
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, unit_box_distance(z) / max(dr, 1.0e-30), trap);
}

// 5. Polyhedral KIFS (Icosahedron / Dodecahedron)
fn polyhedral_kifs_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let sym = u32(ext_slot(is_world, obj, 0u));
    let scale = max(ext_slot(is_world, obj, 1u), 1.05);
    let offset = ext_slot3(is_world, obj, 2u);
    let thick = ext_slot(is_world, obj, 5u);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);

    // Golden-ratio reflection normal (Knighty icosahedral fold plane)
    let n_ico = vec3<f32>(-0.5, -0.80901699, 0.30901699);
    let n_dod = vec3<f32>(-0.80901699, 0.30901699, -0.5);
    let n_fold = select(n_ico, n_dod, sym == 1u);

    var z = q;
    var dr = 1.0;
    var trap = TRAP_START;

    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, (length(z) - (1.0 + thick)) / max(dr, 1.0e-30), trap);
        }
        if dr > DERIVATIVE_LIMIT / scale {
            break;
        }
        z = apply_iter_rot(rot, z);
        for (var f = 0u; f < 3u; f = f + 1u) {
            z.x = abs(z.x);
            z.y = abs(z.y);
            let d = dot(z, n_fold);
            if d < 0.0 {
                z = z - 2.0 * d * n_fold;
            }
        }
        z = z * scale - offset * (scale - 1.0);
        dr = dr * scale;
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    let est = (length(z) - (1.0 + thick)) / max(dr, 1.0e-30);
    return fraction_blend(frac, est, trap);
}

// 6. Harmonic Bulb
fn harmonic_bulb_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let oc = orbit_constant(is_world, obj, q);
    let c = oc.xyz;
    let dr_c = oc.w;
    let power = max(ext_slot(is_world, obj, 0u), 1.2);
    let harm_l = ext_slot(is_world, obj, 1u);
    let harm_m = ext_slot(is_world, obj, 2u);
    let amp = ext_slot(is_world, obj, 3u);
    let bailout = pr(is_world, obj, P_BAILOUT);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);

    var z = q;
    var dr = 1.0;
    var r = length(z);
    var trap = TRAP_START;

    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, bulb_estimate(r, dr, bailout), trap);
        }
        if r > bailout {
            break;
        }
        if r < 1.0e-8 {
            return vec2<f32>(0.0, trap);
        }
        let t = apply_iter_rot(rot, z);
        let theta = acos(clamp(t.z / r, -1.0, 1.0));
        let phi = atan2(t.y, t.x);
        let y_lm = cos(harm_l * theta) * cos(harm_m * phi);
        let mod_factor = max(1.0 + amp * y_lm, 0.25);
        let r_pow_m1 = pow(r, power - 1.0);
        let lip = 1.0 + abs(amp) * (abs(harm_l) + abs(harm_m)) * 0.35;
        let growth = power * r_pow_m1 * lip;
        if dr > DERIVATIVE_LIMIT / growth {
            break;
        }
        let mag = r_pow_m1 * r * mod_factor;
        dr = growth * dr + dr_c;
        let a = theta * power;
        let b = phi * power;
        let sa = sin(a);
        z = vec3<f32>(sa * cos(b), sa * sin(b), cos(a)) * mag + c;
        r = length(z);
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, bulb_estimate(r, dr, bailout), trap);
}

// 7. Mandelbar 3D (Tricorn)
fn mandelbar_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let oc = orbit_constant(is_world, obj, q);
    let c = oc.xyz;
    let dr_c = oc.w;
    let power = max(ext_slot(is_world, obj, 0u), 1.2);
    let conj_xy = ext_slot(is_world, obj, 1u);
    let conj_z = ext_slot(is_world, obj, 2u);
    let twist = ext_slot(is_world, obj, 3u);
    let bailout = pr(is_world, obj, P_BAILOUT);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);

    var z = q;
    var dr = 1.0;
    var r = length(z);
    var trap = TRAP_START;

    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, bulb_estimate(r, dr, bailout), trap);
        }
        if r > bailout {
            break;
        }
        if r < 1.0e-8 {
            return vec2<f32>(0.0, trap);
        }
        let t = apply_iter_rot(rot, z);
        let r_pow_m1 = pow(r, power - 1.0);
        let growth = power * r_pow_m1 * max(abs(conj_xy), max(abs(conj_z), 1.0));
        if dr > DERIVATIVE_LIMIT / growth {
            break;
        }
        let theta = acos(clamp(t.z / r, -1.0, 1.0)) * power * conj_z;
        let phi = atan2(t.y, t.x) * power * conj_xy + twist;
        let mag = r_pow_m1 * r;
        dr = growth * dr + dr_c;
        let st = sin(theta);
        z = vec3<f32>(st * cos(phi), st * sin(phi), cos(theta)) * mag + c;
        r = length(z);
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, bulb_estimate(r, dr, bailout), trap);
}

// 8. Mobius Sphere-Inversion IFS
fn mobius_ifs_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let scale = ext_slot(is_world, obj, 0u);
    let min_r2 = max(ext_slot(is_world, obj, 1u), 1.0e-6);
    let fixed_r2 = max(ext_slot(is_world, obj, 2u), min_r2);
    let pitch = max(ext_slot3(is_world, obj, 3u), vec3<f32>(0.1));
    let twist = ext_slot(is_world, obj, 6u);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);

    var z = q;
    var dr = 1.0;
    var trap = TRAP_START;

    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, (length(z) - 0.32) / max(dr, 1.0e-30), trap);
        }
        z = apply_iter_rot(rot, z);
        if twist != 0.0 {
            let ang = twist * z.z;
            let sa = sin(ang);
            let ca = cos(ang);
            z = vec3<f32>(z.x * ca - z.y * sa, z.x * sa + z.y * ca, z.z);
        }
        z = z - pitch * round(z / pitch);
        let r2 = dot(z, z);
        let k = clamp(fixed_r2 / max(r2, min_r2), 1.0, fixed_r2 / min_r2) * scale;
        let abs_k = max(abs(k), 1.0e-6);
        if dr > DERIVATIVE_LIMIT / abs_k {
            break;
        }
        z = z * k;
        dr = dr * abs_k + 1.0;
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    let est = (length(z) - 0.32) / max(dr, 1.0e-30);
    return fraction_blend(frac, est, trap);
}

// 9. Gyroid / Schwarz P / Schwarz D / Neovius TPMS with multi-octave FBM
fn tpms_base_eval(kind: u32, p: vec3<f32>) -> f32 {
    let sx = sin(p.x);
    let sy = sin(p.y);
    let sz = sin(p.z);
    let cx = cos(p.x);
    let cy = cos(p.y);
    let cz = cos(p.z);
    if kind == 1u {
        // Schwarz P
        return (cx + cy + cz) * 0.57735027;
    }
    if kind == 2u {
        // Schwarz D (Diamond)
        return sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz;
    }
    if kind == 3u {
        // Neovius
        return (3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz) * 0.22;
    }
    // Gyroid
    return (sx * cy + sy * cz + sz * cx) * 0.65;
}

fn gyroid_tpms_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let kind = u32(ext_slot(is_world, obj, 0u));
    let freq = max(ext_slot(is_world, obj, 1u), 0.1);
    let thick = ext_slot(is_world, obj, 2u);
    let bias = ext_slot(is_world, obj, 3u);
    let octaves = clamp(u32(ext_slot(is_world, obj, 4u)), 1u, 6u);
    let lacunarity = max(ext_slot(is_world, obj, 5u), 1.05);
    let gain = clamp(ext_slot(is_world, obj, 6u), 0.0, 0.85);
    let twist = ext_slot(is_world, obj, 7u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    var p = q;
    if twist != 0.0 {
        let ang = twist * p.y;
        let sa = sin(ang);
        let ca = cos(ang);
        p = vec3<f32>(p.x * ca - p.z * sa, p.y, p.x * sa + p.z * ca);
    }

    var f = freq;
    var amp = 1.0;
    var val = 0.0;
    var grad_lip = freq * (1.0 + abs(twist));
    var trap = TRAP_START;

    for (var i = 0u; i < octaves; i = i + 1u) {
        let sp = p * f;
        let g = tpms_base_eval(kind, sp);
        val = val + amp * g;
        if i > 0u {
            grad_lip = grad_lip + amp * f * 0.65;
        }
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, sin(sp)));
        }
        p = vec3<f32>(p.y, p.z, p.x) + vec3<f32>(0.37, 0.71, 0.19);
        f = f * lacunarity;
        amp = amp * gain;
    }

    let d = (abs(val - bias) - thick * freq) / max(grad_lip, 1.0e-4);
    return vec2<f32>(d, trap);
}

// 10. Lidinoid / Fischer-Koch S / Split-P TPMS
fn lidinoid_tpms_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let kind = u32(ext_slot(is_world, obj, 0u));
    let freq = max(ext_slot(is_world, obj, 1u), 0.1);
    let thick = ext_slot(is_world, obj, 2u);
    let bias = ext_slot(is_world, obj, 3u);
    let h_mix = ext_slot(is_world, obj, 4u);
    let r_warp = ext_slot(is_world, obj, 5u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    let r = length(q);
    let p = q * (freq * (1.0 + r_warp * r));
    let sx = sin(p.x);
    let sy = sin(p.y);
    let sz = sin(p.z);
    let cx = cos(p.x);
    let cy = cos(p.y);
    let cz = cos(p.z);
    let s2x = sin(2.0 * p.x);
    let s2y = sin(2.0 * p.y);
    let s2z = sin(2.0 * p.z);
    let c2x = cos(2.0 * p.x);
    let c2y = cos(2.0 * p.y);
    let c2z = cos(2.0 * p.z);

    var val = 0.0;
    if kind == 1u {
        // Fischer-Koch S
        val = c2x * sy * cz + cx * c2y * sz + sx * cy * c2z
            + (h_mix - 0.5) * (sx * cy + sy * cz + sz * cx);
    } else if kind == 2u {
        // Split-P
        val = 1.1 * (s2x * sz * cy + s2y * sx * cz + s2z * sy * cx)
            - 0.2 * h_mix * (c2x * c2y + c2y * c2z + c2z * c2x)
            - 0.4 * (c2x + c2y + c2z);
    } else {
        // Lidinoid
        val = 0.5 * (s2x * cy * sz + s2y * cz * sx + s2z * cx * sy)
            - 0.5 * h_mix * (c2x * c2y + c2y * c2z + c2z * c2x);
    }

    let lip = freq * (2.2 + abs(h_mix) + abs(r_warp) * 2.0);
    let d = (abs(val - bias) - thick * freq) / max(lip, 1.0e-4);
    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, vec3<f32>(s2x, s2y, s2z));
    }
    return vec2<f32>(d, trap);
}

// 11. Strange Attractor Flow Tube (Aizawa / Halvorsen / Thomas)
fn attractor_velocity(sys: u32, p: vec3<f32>, a: f32, b: f32, swirl: f32) -> vec3<f32> {
    if sys == 1u {
        // Halvorsen
        let s = max(a, 0.5);
        return vec3<f32>(
            -s * p.x - 4.0 * p.y - 4.0 * p.z - p.y * p.y,
            -s * p.y - 4.0 * p.z - 4.0 * p.x - p.z * p.z,
            -s * p.z - 4.0 * p.x - 4.0 * p.y - p.x * p.x,
        ) * 0.25 * swirl;
    }
    if sys == 2u {
        // Thomas cyclically symmetric attractor
        let damp = max(b * 0.25, 0.05);
        return vec3<f32>(
            sin(p.y * 2.5) - damp * p.x,
            sin(p.z * 2.5) - damp * p.y,
            sin(p.x * 2.5) - damp * p.z,
        ) * 1.8 * swirl;
    }
    // Aizawa attractor
    let r2 = p.x * p.x + p.y * p.y;
    let vx = (p.z - b) * p.x - 3.5 * swirl * p.y;
    let vy = 3.5 * swirl * p.x + (p.z - b) * p.y;
    let vz = 0.6 + a * p.z - (p.z * p.z * p.z) / 3.0 - r2 * (1.0 + 0.25 * p.z);
    return vec3<f32>(vx, vy, vz);
}

fn attractor_tube_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let sys = u32(ext_slot(is_world, obj, 0u));
    let dt = clamp(ext_slot(is_world, obj, 1u), 0.005, 0.15);
    let tube_r = ext_slot(is_world, obj, 2u);
    let swirl = ext_slot(is_world, obj, 3u);
    let param_a = ext_slot(is_world, obj, 4u);
    let param_b = ext_slot(is_world, obj, 5u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    var n_raw = max(pr(is_world, obj, P_ITERATIONS), 1.0);
    if iter_override >= 0.0 {
        n_raw = max(iter_override, 1.0);
    }
    let steps = clamp(u32(n_raw), 2u, 24u);

    var p = q;
    var dr = 1.0;
    var min_shell = 1.0e10;
    var trap = TRAP_START;

    for (var i = 0u; i < steps; i = i + 1u) {
        let v1 = attractor_velocity(sys, p, param_a, param_b, swirl);
        let mid = p + 0.5 * dt * v1;
        let v2 = attractor_velocity(sys, mid, param_a, param_b, swirl);
        p = p + dt * v2;
        dr = dr * (1.0 + 1.8 * dt * swirl);
        let ring = abs(length(vec2<f32>(length(p.xy) - 0.78, p.z)) - 0.22);
        let d_step = (ring - tube_r) / dr;
        min_shell = min(min_shell, d_step);
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, p));
        }
        if dot(p, p) > 25.0 {
            break;
        }
    }
    return vec2<f32>(min_shell, trap);
}

// 12. Iterated Torus Knot Braid
fn torus_knot_braid_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let winds_p = max(round(ext_slot(is_world, obj, 0u)), 1.0);
    let winds_q = max(round(ext_slot(is_world, obj, 1u)), 1.0);
    let major_r = ext_slot(is_world, obj, 2u);
    let minor_r = ext_slot(is_world, obj, 3u);
    let tube_r = ext_slot(is_world, obj, 4u);
    let strands = max(round(ext_slot(is_world, obj, 5u)), 1.0);
    let braid_d = ext_slot(is_world, obj, 6u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    let phi = atan2(q.y, q.x);
    let r_xy = length(q.xy);
    let p_count = u32(clamp(winds_p, 1.0, 12.0));
    var best_d = 1.0e10;
    var best_pt = q;

    for (var k = 0u; k < p_count; k = k + 1u) {
        let t = (phi + 6.2831853 * f32(k)) / winds_p;
        let qt = winds_q * t;
        let core_r = major_r + minor_r * cos(qt);
        let core_z = minor_r * sin(qt);
        let dr_xy = r_xy - core_r;
        let dz = q.z - core_z;
        let pol_r = length(vec2<f32>(dr_xy, dz));
        let pol_ang = atan2(dz, dr_xy);
        // Orthogonal projection factor for slope of (p, q) curve
        let slope = length(vec2<f32>(winds_p * major_r, winds_q * minor_r)) / max(winds_p * major_r, 0.1);
        let braid_wave = braid_d * cos(strands * (pol_ang - winds_q * t));
        let d_branch = (pol_r - tube_r - braid_wave) / max(slope, 1.0);
        if d_branch < best_d {
            best_d = d_branch;
            best_pt = vec3<f32>(dr_xy, dz, sin(qt));
        }
    }

    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, best_pt);
    }
    return vec2<f32>(best_d, trap);
}

// 13. Hopf Fibration Linked Tori
fn hopf_fibration_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let fibers = max(round(ext_slot(is_world, obj, 0u)), 2.0);
    let shells = clamp(u32(ext_slot(is_world, obj, 1u)), 1u, 6u);
    let base_lat = clamp(ext_slot(is_world, obj, 2u), 0.15, 1.45);
    let tube_r = ext_slot(is_world, obj, 3u);
    let twist = ext_slot(is_world, obj, 4u);
    let scale = max(ext_slot(is_world, obj, 5u), 0.1);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    // Inverse stereographic projection from R^3 to unit S^3 = (z1, z2) in C^2
    let p = q / scale;
    let r2 = dot(p, p);
    let denom = 1.0 / (1.0 + r2);
    let x1 = 2.0 * p.x * denom;
    let x2 = 2.0 * p.y * denom;
    let x3 = 2.0 * p.z * denom;
    let x4 = (1.0 - r2) * denom;

    // Conformal factor of stereographic projection from S^3 to R^3 is (1 + r^2) / 2
    let conformal = 0.5 * (1.0 + r2) * scale;

    let r_a = length(vec2<f32>(x1, x2));
    let r_b = length(vec2<f32>(x3, x4));
    let eta = atan2(r_b, r_a); // Clifford torus latitude in [0, pi/2]
    let xi1 = atan2(x2, x1);
    let xi2 = atan2(x4, x3);
    // Villarceau fiber angle across the Clifford torus
    let fiber_phase = xi1 - twist * xi2;
    let sector = 6.2831853 / fibers;

    var best_s3 = 1.0e10;
    for (var s = 0u; s < shells; s = s + 1u) {
        let frac_s = (f32(s) - 0.5 * f32(shells - 1u)) * 0.22;
        let target_eta = clamp(base_lat + frac_s, 0.12, 1.45);
        let d_eta = eta - target_eta;
        let phase_shift = f32(s) * 0.5 * sector;
        let wrapped = fiber_phase + phase_shift - sector * round((fiber_phase + phase_shift) / sector);
        let d_fiber = wrapped * sin(target_eta) * cos(target_eta);
        let dist_s3 = length(vec2<f32>(d_eta, d_fiber));
        best_s3 = min(best_s3, dist_s3);
    }

    let d = best_s3 * conformal - tube_r;
    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, vec3<f32>(eta, sin(fiber_phase), cos(fiber_phase)));
    }
    return vec2<f32>(d, trap);
}

// 14. Dupin Cyclide
fn dupin_cyclide_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let a = max(ext_slot(is_world, obj, 0u), 0.2);
    let b = max(ext_slot(is_world, obj, 1u), 0.2);
    let c = ext_slot(is_world, obj, 2u);
    let d_rad = ext_slot(is_world, obj, 3u);
    let thick = ext_slot(is_world, obj, 4u);
    let rib_freq = ext_slot(is_world, obj, 5u);
    let rib_depth = ext_slot(is_world, obj, 6u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    // Quartic Dupin cyclide F(x, y, z) = (x^2 + y^2 + z^2 + b^2 - d^2)^2 - 4*(a*x - c*d)^2 - 4*b^2*y^2
    let r2 = dot(q, q);
    let k = r2 + b * b - d_rad * d_rad;
    let ax_cd = a * q.x - c * d_rad;
    let f = k * k - 4.0 * ax_cd * ax_cd - 4.0 * b * b * q.y * q.y;

    let gx = 4.0 * k * q.x - 8.0 * a * ax_cd;
    let gy = 4.0 * k * q.y - 8.0 * b * b * q.y;
    let gz = 4.0 * k * q.z;
    let grad_len = max(length(vec3<f32>(gx, gy, gz)), 0.25);

    var dist = f / grad_len;
    if thick > 0.0 {
        dist = abs(dist) - thick;
    }
    if rib_freq > 0.0 && rib_depth != 0.0 {
        let u_ang = atan2(q.y, q.x);
        let v_ang = atan2(q.z, length(q.xy) - a);
        dist = dist + rib_depth * cos(rib_freq * u_ang) * cos(rib_freq * v_ang);
    }

    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, vec3<f32>(ax_cd, q.y, q.z));
    }
    return vec2<f32>(dist * 0.65, trap);
}

// 15. 3D Gielis Superformula (Radiolaria)
fn gielis_r(ang: f32, m: f32, n1: f32, n2: f32, n3: f32) -> f32 {
    let t = 0.25 * m * ang;
    let t1 = pow(max(abs(cos(t)), 1.0e-6), n2);
    let t2 = pow(max(abs(sin(t)), 1.0e-6), n3);
    let sum = max(t1 + t2, 1.0e-6);
    return clamp(pow(sum, -1.0 / max(n1, 0.15)), 0.05, 3.0);
}

fn superformula_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let m1 = ext_slot(is_world, obj, 0u);
    let n1_1 = ext_slot(is_world, obj, 1u);
    let n2_1 = ext_slot(is_world, obj, 2u);
    let n3_1 = ext_slot(is_world, obj, 3u);
    let m2 = ext_slot(is_world, obj, 4u);
    let n1_2 = ext_slot(is_world, obj, 5u);
    let n2_2 = ext_slot(is_world, obj, 6u);
    let n3_2 = ext_slot(is_world, obj, 7u);
    let thick = ext_slot(is_world, obj, 8u);
    let pore_freq = ext_slot(is_world, obj, 9u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    let r = length(q);
    if r < 1.0e-6 {
        return vec2<f32>(-0.5, TRAP_START);
    }
    let phi = atan2(q.y, q.x);
    let theta = asin(clamp(q.z / r, -1.0, 1.0));
    let r1 = gielis_r(phi, m1, n1_1, n2_1, n3_1);
    let r2 = gielis_r(theta, m2, n1_2, n2_2, n3_2);
    let target_r = r1 * r2;

    // Conservative Lipschitz factor for angular lobes
    let lip = 1.0 + 0.12 * (m1 + m2);
    var d = (r - target_r) / lip;
    if thick > 0.0 {
        d = abs(d) - thick;
    }
    if pore_freq > 0.0 {
        let pore_pattern = cos(pore_freq * phi) * cos(pore_freq * theta);
        let pore_sdf = (0.45 - pore_pattern) / max(pore_freq, 1.0);
        d = max(d, -pore_sdf);
    }

    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, q / max(target_r, 0.1));
    }
    return vec2<f32>(d, trap);
}

// 16. Spherical Fibonacci Phyllotaxis (Romanesco)
fn phyllotaxis_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let n_florets = max(round(ext_slot(is_world, obj, 0u)), 8.0);
    let cone_h = ext_slot(is_world, obj, 1u);
    let sharp = max(ext_slot(is_world, obj, 2u), 0.3);
    let twist = ext_slot(is_world, obj, 3u);
    let sub_f = ext_slot(is_world, obj, 4u);
    let base_r = ext_slot(is_world, obj, 5u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    let r = length(q);
    if r < 1.0e-6 {
        return vec2<f32>(-base_r, TRAP_START);
    }
    let dir = q / r;
    // Keinert / Marques closed-form O(1) inverse Spherical Fibonacci point lookup
    let phi_gold = 1.61803399;
    let ga = 2.39996323; // 2*pi * (1 - 1/phi)
    let p_phi = atan2(dir.y, dir.x);
    let cos_theta = dir.z;
    let k = max(2.0, floor(log(n_florets * 3.14159265 * 2.23606798 * max(1.0 - cos_theta * cos_theta, 1.0e-5)) / log(phi_gold * phi_gold)));
    let fk = pow(phi_gold, k) / 2.23606798;
    let f0 = round(fk);
    let f1 = round(fk * phi_gold);

    let b00 = 2.0 * 3.14159265 * fract((f0 + 1.0) * (phi_gold - 1.0)) - ga;
    let b01 = 2.0 * 3.14159265 * fract((f1 + 1.0) * (phi_gold - 1.0)) - ga;
    let b10 = -2.0 * f0 / n_florets;
    let b11 = -2.0 * f1 / n_florets;
    let inv_det = 1.0 / (b00 * b11 - b01 * b10);
    let c0 = b11 * p_phi - b01 * (cos_theta - (1.0 - 1.0 / n_florets));
    let c1 = -b10 * p_phi + b00 * (cos_theta - (1.0 - 1.0 / n_florets));
    let base_coord = vec2<f32>(floor(c0 * inv_det), floor(c1 * inv_det));

    let cell_radius = 2.15 / sqrt(n_florets);
    var min_ang = 10.0;
    var best_seed = dir;

    for (var s = 0u; s < 4u; s = s + 1u) {
        let uv = base_coord + vec2<f32>(f32(s & 1u), f32((s >> 1u) & 1u));
        let idx = clamp(f0 * uv.x + f1 * uv.y, 0.0, n_florets - 1.0);
        let z_i = 1.0 - (2.0 * idx + 1.0) / n_florets;
        let r_i = sqrt(max(1.0 - z_i * z_i, 0.0));
        let a_i = ga * idx;
        let seed = vec3<f32>(r_i * cos(a_i), r_i * sin(a_i), z_i);
        let ang_dist = length(dir - seed);
        if ang_dist < min_ang {
            min_ang = ang_dist;
            best_seed = seed;
        }
    }

    let u_norm = clamp(min_ang / max(cell_radius, 1.0e-4), 0.0, 1.0);
    var profile = pow(1.0 - u_norm, sharp);
    if sub_f > 0.0 {
        let rel = dir - best_seed;
        let sub_ang = atan2(rel.y + rel.z * 0.5, rel.x);
        let sub_wave = 0.22 * (1.0 - u_norm) * u_norm * (0.5 + 0.5 * cos(sub_f * sub_ang + twist * 8.0 * u_norm));
        profile = profile + sub_wave;
    }

    let surf_r = base_r + cone_h * profile;
    let d = (r - surf_r) * 0.55;
    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, best_seed * (1.0 + u_norm));
    }
    return vec2<f32>(d, trap);
}

// 17. Nautilus / Ammonite Logarithmic Helicospiral Shell
fn nautilus_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let b_rate = clamp(ext_slot(is_world, obj, 0u), 0.05, 0.45);
    let tight = ext_slot(is_world, obj, 1u);
    let spire = ext_slot(is_world, obj, 2u);
    let wall = ext_slot(is_world, obj, 3u);
    let rib_freq = ext_slot(is_world, obj, 4u);
    let rib_amp = ext_slot(is_world, obj, 5u);
    let max_turns = ext_slot(is_world, obj, 6u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    let r_xy = max(length(q.xy), 1.0e-5);
    let phi = atan2(q.y, q.x); // in [-pi, pi]
    let two_pi = 6.2831853;
    let r0 = 0.08;
    let n_float = (log(r_xy / r0) / b_rate - phi) / two_pi;
    let n_center = round(n_float);

    var best_d = 1.0e10;
    var best_theta = 0.0;

    for (var k = -1; k <= 1; k = k + 1) {
        let n_turn = clamp(n_center + f32(k), 0.0, max_turns);
        let theta = clamp(phi + two_pi * n_turn, 0.0, two_pi * max_turns);
        let whorl_r = r0 * exp(b_rate * theta);
        let tube_r = whorl_r * (1.0 - exp(-b_rate * two_pi)) * 0.55 * tight;
        let whorl_z = -spire * whorl_r;
        let dr = r_xy - whorl_r;
        let dz = q.z - whorl_z;
        let cross_d = length(vec2<f32>(dr, dz));
        let rib = rib_amp * whorl_r * cos(rib_freq * theta);
        let outer = cross_d - (tube_r + rib);
        var shell_d = abs(outer) - wall;
        // Internal chamber septa (for cutaway or glass views)
        if outer < 0.0 && rib_freq >= 1.0 {
            let septum_phase = sin(0.5 * rib_freq * theta) * whorl_r / max(0.5 * rib_freq, 1.0);
            let septum_d = abs(septum_phase - 0.25 * dr) - 0.65 * wall;
            shell_d = min(shell_d, max(outer, septum_d));
        }
        if shell_d < best_d {
            best_d = shell_d;
            best_theta = theta;
        }
    }

    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, vec3<f32>(cos(best_theta), sin(best_theta), best_theta * 0.1));
    }
    return vec2<f32>(best_d * 0.7, trap);
}

// 18. Multi-Scale 3D Voronoi Trabecular Foam (F2 - F1 struts)
fn voronoi_cell_seed(cell: vec3<i32>, jitter: f32) -> vec3<f32> {
    let ux = bitcast<u32>(cell.x * 73856093);
    let uy = bitcast<u32>(cell.y * 19349663);
    let uz = bitcast<u32>(cell.z * 83492791);
    let h0 = ss_lowbias32(ux ^ uy ^ uz);
    let h1 = ss_lowbias32(h0 ^ 0x9e3779b9u);
    let h2 = ss_lowbias32(h1 ^ 0x85ebca6bu);
    let rand_off = (vec3<f32>(f32(h0), f32(h1), f32(h2)) * (1.0 / 4294967296.0) - vec3<f32>(0.5)) * jitter;
    return vec3<f32>(f32(cell.x), f32(cell.y), f32(cell.z)) + vec3<f32>(0.5) + rand_off;
}

fn voronoi_edge_dist(p: vec3<f32>, jitter: f32) -> vec2<f32> {
    let base_cell = vec3<i32>(floor(p));
    var min_d2 = 1.0e10;
    var best_seed = p;

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let s = voronoi_cell_seed(base_cell + vec3<i32>(dx, dy, dz), jitter);
                let diff = p - s;
                let d2 = dot(diff, diff);
                if d2 < min_d2 {
                    min_d2 = d2;
                    best_seed = s;
                }
            }
        }
    }

    // Exact Voronoi cell-wall distance via plane bisectors to neighbors
    var wall_d = 1.0e10;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let s = voronoi_cell_seed(base_cell + vec3<i32>(dx, dy, dz), jitter);
                let delta = s - best_seed;
                let len_d = length(delta);
                if len_d > 1.0e-4 {
                    let plane_d = dot(0.5 * (best_seed + s) - p, delta / len_d);
                    wall_d = min(wall_d, plane_d);
                }
            }
        }
    }
    return vec2<f32>(wall_d, sqrt(min_d2));
}

fn voronoi_foam_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let density = max(ext_slot(is_world, obj, 0u), 0.2);
    let strut_r = ext_slot(is_world, obj, 1u);
    let jitter = clamp(ext_slot(is_world, obj, 2u), 0.0, 0.95);
    let octaves = clamp(u32(ext_slot(is_world, obj, 3u)), 1u, 3u);
    let lacunarity = max(ext_slot(is_world, obj, 4u), 1.5);
    let hollow = ext_slot(is_world, obj, 5u);
    let bound = max(pr(is_world, obj, P_BOUND_RADIUS), 1.0);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    var freq = density;
    var thick = strut_r;
    var v0 = voronoi_edge_dist(q * freq, jitter);
    var d = (thick - v0.x) / freq; // Inside cell walls (`wall_d < thick`) is solid (`d < 0`)

    for (var i = 1u; i < octaves; i = i + 1u) {
        freq = freq * lacunarity;
        thick = thick * 0.65;
        let vi = voronoi_edge_dist(q * freq + vec3<f32>(1.7, 3.1, 2.3), jitter);
        let di = (thick - vi.x) / freq;
        d = min(d, di);
    }

    let r = length(q);
    if hollow > 0.0 {
        let inner_r = hollow * bound;
        d = max(d, inner_r - r);
    }

    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, vec3<f32>(v0.x, v0.y, r));
    }
    return vec2<f32>(d * 0.8, trap);
}

// 19. Procedural Quartz Crystal Cluster / Geode
fn single_hex_crystal_sdf(p: vec3<f32>, radius: f32, length_h: f32, slope: f32) -> f32 {
    // Hexagonal cross-section in XY using 6-fold reflection
    var xy = abs(p.xy);
    let k = vec2<f32>(-0.8660254, 0.5);
    xy = xy - 2.0 * min(dot(k, xy), 0.0) * k;
    let d_hex = xy.y - radius;
    // Rhombohedral termination pyramid at +Z and base at 0
    let norm_slope = inverseSqrt(1.0 + slope * slope);
    let d_tip = (p.z - length_h + slope * xy.y) * norm_slope;
    let d_base = -p.z;
    return max(max(d_hex, d_tip), d_base);
}

fn crystal_geode_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let count = clamp(u32(ext_slot(is_world, obj, 0u)), 1u, 24u);
    let p_len = ext_slot(is_world, obj, 1u);
    let p_rad = ext_slot(is_world, obj, 2u);
    let slope = max(ext_slot(is_world, obj, 3u), 0.2);
    let spread = ext_slot(is_world, obj, 4u);
    let cavity = ext_slot(is_world, obj, 5u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    let ga = 2.39996323;
    var best_d = 1.0e10;
    var best_idx = 0.0;

    for (var i = 0u; i < count; i = i + 1u) {
        let fi = f32(i);
        let t = (fi + 0.5) / f32(count);
        let tilt = spread * sqrt(t);
        let az = ga * fi;
        let st = sin(tilt);
        let ct = cos(tilt);
        let sa = sin(az);
        let ca = cos(az);

        // Orthonormal basis with Z along the crystal growth axis
        let axis_z = vec3<f32>(st * ca, st * sa, ct);
        let axis_x = vec3<f32>(-sa, ca, 0.0);
        let axis_y = cross(axis_z, axis_x);

        let scale_var = 1.0 - 0.38 * fract(fi * 0.61803399);
        let local_p = vec3<f32>(
            dot(q, axis_x),
            dot(q, axis_y),
            dot(q, axis_z) + 0.35 * p_len,
        );
        let d_c = single_hex_crystal_sdf(local_p, p_rad * scale_var, p_len * (0.65 + 0.5 * scale_var), slope);
        if d_c < best_d {
            best_d = d_c;
            best_idx = fi / f32(max(count, 1u));
        }
    }

    if cavity > 0.0 {
        let cav_r = cavity * p_len * 0.75;
        best_d = max(best_d, cav_r - length(q));
    }

    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, q * (1.0 + best_idx));
    }
    return vec2<f32>(best_d, trap);
}

// 20. Mechanical Astrolabe / Nested Gyroscopic Gimbal Rings
fn astrolabe_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let rings = clamp(u32(ext_slot(is_world, obj, 0u)), 1u, 8u);
    let outer_r = ext_slot(is_world, obj, 1u);
    let spacing = ext_slot(is_world, obj, 2u);
    let width = ext_slot(is_world, obj, 3u);
    let thick = ext_slot(is_world, obj, 4u);
    let teeth = round(ext_slot(is_world, obj, 5u));
    let tooth_d = ext_slot(is_world, obj, 6u);
    let tilt_step = ext_slot(is_world, obj, 7u);
    let spin_step = ext_slot(is_world, obj, 8u);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);

    var p = q;
    var best_d = length(p) - max(outer_r - f32(rings) * spacing, 0.08); // Central sun sphere
    var best_ring = 0.0;

    for (var i = 0u; i < rings; i = i + 1u) {
        let fi = f32(i);
        // Incremental Euler tilt & spin per nested gimbal ring
        let sx = sin(tilt_step);
        let cx = cos(tilt_step);
        let sz = sin(spin_step);
        let cz = cos(spin_step);
        p = vec3<f32>(p.x * cz - p.y * sz, p.x * sz + p.y * cz, p.z);
        p = vec3<f32>(p.x, p.y * cx - p.z * sx, p.y * sx + p.z * cx);

        let ring_r = max(outer_r - fi * spacing, 0.12);
        let r_xy = length(p.xy);
        var gear_mod = 0.0;
        if teeth >= 1.0 && tooth_d > 0.0 {
            let ang = atan2(p.y, p.x);
            gear_mod = tooth_d * clamp(2.0 * cos(teeth * ang), -1.0, 1.0);
        }
        let d_rad = abs(r_xy - ring_r - gear_mod) - thick;
        let d_ax = abs(p.z) - width;
        let d_ring = length(max(vec2<f32>(d_rad, d_ax), vec2<f32>(0.0))) + min(max(d_rad, d_ax), 0.0);
        // Carve decorative sight-holes along the ring band
        let hole_ang = atan2(p.y, p.x);
        let hole_sector = 6.2831853 / 8.0;
        let hole_w = (hole_ang - hole_sector * round(hole_ang / hole_sector)) * ring_r;
        let d_hole = length(vec2<f32>(hole_w, p.z)) - 0.42 * width;
        let d_carved = max(d_ring, -d_hole);

        if d_carved < best_d {
            best_d = d_carved;
            best_ring = fi + 1.0;
        }
    }

    var trap = TRAP_START;
    if trap_mode != 0u {
        trap = eval_trap3(trap_cfg, p * (0.5 + 0.15 * best_ring));
    }
    return vec2<f32>(best_d * 0.8, trap);
}
