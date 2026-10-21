// WarpBro WGPU compute kernels — 1:1 port of fractal-kernels (CUDA) and ofx-fractal (WGSL).
//
// Pre-prepended by `render_wgpu::shader_source()`:
// 1. `standard_surface_bsdf::wgsl_source()`
// 2. All `P_*`, `O_*`, `M_*`, `FAMILY_*`, `MASK_*`, `ARRAY_*`, `PALETTE_SAMPLES` constants.

@group(0) @binding(0) var<uniform> params_u: array<vec4<f32>, 64>;
@group(0) @binding(1) var<storage, read> lut: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> objects: array<f32>;
@group(0) @binding(3) var<storage, read> lights: array<f32>;
@group(0) @binding(4) var<storage, read> active_tiles: array<u32>;
@group(0) @binding(5) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> albedo: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> normal_buf: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> stats: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> active_rw: array<u32>;
@group(0) @binding(10) var<storage, read_write> aux_out: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> sdr_out: array<u32>;
@group(0) @binding(12) var<storage, read_write> summary_out: array<vec4<f32>>;

const TRAP_START: f32 = 3.40282347e38;
const DERIVATIVE_LIMIT: f32 = 1.0e38;
const STEP_CAP_STEPS: f32 = 256.0;
const CONE_START_STEPS: u32 = 256u;
const HIT_REFINEMENTS: u32 = 16u;
const GSAA_KAPPA: f32 = 0.18;

const RAY_PRIMARY: u32 = 0u;
const RAY_BOUNCE: u32 = 1u;
const RAY_VISIBILITY: u32 = 2u;

const MAT_DISPATCH: u32 = 0u;
const MAT_FAST: u32 = 1u;
const MAT_FULL: u32 = 2u;

fn param(slot: u32) -> f32 {
    return params_u[slot >> 2u][slot & 3u];
}

fn param3(slot: u32) -> vec3<f32> {
    return vec3<f32>(param(slot), param(slot + 1u), param(slot + 2u));
}

fn pr(is_world: bool, obj: u32, slot: u32) -> f32 {
    if is_world && ((slot >= P_FAMILY && slot < P_LIGHT_DIR) || (slot >= P_BASE && slot < P_EXPOSURE) || slot == P_MATERIAL_MODEL || (slot >= P_TRANSMISSION && slot < P_COUNT)) {
        return objects[obj * OBJECT_STRIDE + slot];
    }
    return param(slot);
}

fn pr3(is_world: bool, obj: u32, slot: u32) -> vec3<f32> {
    return vec3<f32>(
        pr(is_world, obj, slot),
        pr(is_world, obj, slot + 1u),
        pr(is_world, obj, slot + 2u),
    );
}

fn lp(is_world: bool, light: u32, slot: u32) -> f32 {
    if is_world {
        return lights[light * LIGHT_STRIDE + slot - P_LIGHT_DIR];
    }
    return param(slot);
}

fn lp3(is_world: bool, light: u32, slot: u32) -> vec3<f32> {
    return vec3<f32>(
        lp(is_world, light, slot),
        lp(is_world, light, slot + 1u),
        lp(is_world, light, slot + 2u),
    );
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, param3(P_LUMA));
}

fn is_ofx() -> bool {
    return param(P_OFX) != 0.0;
}

fn frame_origin() -> vec2<i32> {
    if is_ofx() {
        return vec2<i32>(i32(param(P_APERTURE)), i32(param(P_FOCUS_DISTANCE)));
    }
    return vec2<i32>(0, 0);
}

fn tile_origin() -> vec2<i32> {
    if is_ofx() {
        return vec2<i32>(i32(param(P_TILE_X)), i32(param(P_TILE_Y)));
    }
    return vec2<i32>(0, 0);
}

fn tile_size() -> vec2<u32> {
    if is_ofx() {
        return vec2<u32>(u32(param(P_TILE_WIDTH)), u32(param(P_TILE_HEIGHT)));
    }
    return vec2<u32>(u32(param(P_WIDTH)), u32(param(P_HEIGHT)));
}

fn sky_alpha() -> f32 {
    if is_ofx() {
        return param(P_OFX_SKY_ALPHA);
    }
    return 1.0;
}

fn thin_film_energy() -> u32 {
    if is_ofx() {
        return u32(param(P_OFX_THIN_FILM_ENERGY));
    }
    return SS_THIN_FILM_ENERGY_MATERIALX;
}

fn single_no_clip() -> bool {
    return is_ofx() && param(P_OFX_NO_CLIP) != 0.0;
}

fn unit_or(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if l2 > 0.0 {
        return v * inverseSqrt(l2);
    }
    return fallback;
}

fn resolved_family(is_world: bool, obj: u32, family_override: u32) -> u32 {
    if family_override == FAMILY_WORLD {
        return u32(pr(is_world, obj, P_FAMILY));
    }
    return family_override;
}

fn unsigned_family(family: u32) -> bool {
    return !ext_signed_family(family);
}

fn orbit_constant(is_world: bool, obj: u32, p: vec3<f32>) -> vec4<f32> {
    let t = pr(is_world, obj, P_JULIA);
    let julia_c = pr3(is_world, obj, P_JULIA_C);
    var base_pt: vec3<f32>;
    if t == 0.0 {
        base_pt = p;
    } else if t == 1.0 {
        base_pt = julia_c;
    } else {
        base_pt = p * (1.0 - t) + julia_c * t;
    }
    return vec4<f32>(base_pt + pr3(is_world, obj, P_ORBIT_OFFSET), 1.0 - t);
}

struct IterRot {
    enabled: bool,
    r0: vec3<f32>,
    r1: vec3<f32>,
    r2: vec3<f32>,
}

fn load_iter_rot(is_world: bool, obj: u32) -> IterRot {
    let enabled = pr(is_world, obj, P_ITER_ROTATE) != 0.0;
    if !enabled {
        return IterRot(false, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    }
    return IterRot(
        true,
        pr3(is_world, obj, P_ITER_ROT),
        pr3(is_world, obj, P_ITER_ROT + 3u),
        pr3(is_world, obj, P_ITER_ROT + 6u),
    );
}

fn apply_iter_rot(rot: IterRot, v: vec3<f32>) -> vec3<f32> {
    if !rot.enabled {
        return v;
    }
    return vec3<f32>(dot(rot.r0, v), dot(rot.r1, v), dot(rot.r2, v));
}

struct Trap3 {
    mode: u32,
    point: vec3<f32>,
    normal: vec3<f32>,
}

fn load_trap3(is_world: bool, obj: u32, mode: u32) -> Trap3 {
    if mode <= 1u {
        return Trap3(mode, vec3<f32>(0.0), vec3<f32>(0.0));
    }
    return Trap3(mode, pr3(is_world, obj, P_TRAP_POINT), pr3(is_world, obj, P_TRAP_NORMAL));
}

fn eval_trap3(t: Trap3, v: vec3<f32>) -> f32 {
    if t.mode == 1u {
        return length(v);
    }
    let rel = v - t.point;
    let proj = dot(rel, t.normal);
    if t.mode == 2u {
        return abs(proj);
    }
    return length(rel - t.normal * proj);
}

fn eval_trap4(t: Trap3, q: vec4<f32>) -> f32 {
    if t.mode == 1u {
        return length(q);
    }
    let rel = q.xyz - t.point;
    let proj = dot(rel, t.normal);
    if t.mode == 2u {
        return abs(proj);
    }
    let o = rel - t.normal * proj;
    return sqrt(dot(o, o) + q.w * q.w);
}

struct Fraction {
    run: u32,
    f: f32,
    saved_d: f32,
    saved_trap: f32,
    has_saved: bool,
}

fn fraction_new(is_world: bool, obj: u32, iter_override: f32) -> Fraction {
    var n = max(pr(is_world, obj, P_ITERATIONS), 0.0);
    if iter_override >= 0.0 {
        n = max(iter_override, 0.0);
    }
    let whole = u32(floor(n));
    let f = n - f32(whole);
    let run = whole + select(0u, 1u, f > 0.0);
    return Fraction(run, f, 0.0, TRAP_START, false);
}

fn fraction_due(frac: Fraction, i: u32) -> bool {
    return frac.f > 0.0 && i + 1u == frac.run;
}

fn fraction_save(frac: ptr<function, Fraction>, d: f32, trap: f32) {
    (*frac).saved_d = d;
    (*frac).saved_trap = trap;
    (*frac).has_saved = true;
}

fn fraction_blend(frac: Fraction, d: f32, trap: f32) -> vec2<f32> {
    if frac.has_saved {
        return vec2<f32>(
            frac.saved_d + (d - frac.saved_d) * frac.f,
            frac.saved_trap + (trap - frac.saved_trap) * frac.f,
        );
    }
    return vec2<f32>(d, trap);
}

fn bulb_estimate(r: f32, dr: f32, bailout: f32) -> f32 {
    if r <= bailout || dr <= 0.0 {
        return 0.0;
    }
    return 0.5 * log(r) * r / dr;
}

struct BulbConfig {
    fast8: bool,
    power: f32,
    growth_scale: f32,
    theta_power: f32,
    theta_phase: f32,
    phi_power: f32,
    phi_phase: f32,
}

fn load_bulb_config(is_world: bool, obj: u32) -> BulbConfig {
    let fast8 = pr(is_world, obj, P_BULB_FAST8) != 0.0;
    if fast8 {
        return BulbConfig(true, 8.0, 1.0, 8.0, 0.0, 8.0, 0.0);
    }
    return BulbConfig(
        false,
        pr(is_world, obj, P_BULB_POWER),
        pr(is_world, obj, P_BULB_GROWTH),
        pr(is_world, obj, P_BULB_THETA_POWER),
        pr(is_world, obj, P_BULB_THETA_PHASE),
        pr(is_world, obj, P_BULB_PHI_POWER),
        pr(is_world, obj, P_BULB_PHI_PHASE),
    );
}

fn bulb_step_cfg(
    cfg: BulbConfig,
    rot: IterRot,
    z: ptr<function, vec3<f32>>,
    dr: ptr<function, f32>,
    r: f32,
    c: vec3<f32>,
    dr_constant: f32,
) -> bool {
    if cfg.fast8 {
        let r2 = r * r;
        let r4 = r2 * r2;
        let r7 = r4 * r2 * r;
        let growth = 8.0 * r7;
        if *dr > DERIVATIVE_LIMIT / growth {
            return false;
        }
        let x2 = (*z).x * (*z).x;
        let y2 = (*z).y * (*z).y;
        let xy2 = x2 + y2;
        if xy2 <= 1.0e-30 {
            *dr = growth * (*dr) + dr_constant;
            *z = vec3<f32>(0.0, 0.0, r4 * r4) + c;
            return true;
        }
        let inv_r = 1.0 / r;
        let inv_xy = inverseSqrt(xy2);
        var ct = (*z).z * inv_r;
        var st = xy2 * inv_xy * inv_r;
        var cp = (*z).x * inv_xy;
        var sp = (*z).y * inv_xy;
        for (var k = 0u; k < 3u; k = k + 1u) {
            let nct = ct * ct - st * st;
            st = 2.0 * st * ct;
            ct = nct;
            let ncp = cp * cp - sp * sp;
            sp = 2.0 * sp * cp;
            cp = ncp;
        }
        let r8 = r4 * r4;
        *dr = growth * (*dr) + dr_constant;
        *z = vec3<f32>(st * cp, st * sp, ct) * r8 + c;
        return true;
    }
    let turned = apply_iter_rot(rot, *z);
    let theta = acos(clamp(turned.z / r, -1.0, 1.0));
    let phi = atan2(turned.y, turned.x);
    let r_pow_m1 = pow(r, cfg.power - 1.0);
    let growth = cfg.power * r_pow_m1 * cfg.growth_scale;
    if *dr > DERIVATIVE_LIMIT / growth {
        return false;
    }
    let magnitude = r_pow_m1 * r;
    *dr = growth * (*dr) + dr_constant;
    let a = theta * cfg.theta_power + cfg.theta_phase;
    let b = phi * cfg.phi_power + cfg.phi_phase;
    let sa = sin(a);
    let ca = cos(a);
    let sb = sin(b);
    let cb = cos(b);
    *z = vec3<f32>(sa * cb, sa * sb, ca) * magnitude + c;
    return true;
}

fn bulb_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let oc = orbit_constant(is_world, obj, q);
    let c = oc.xyz;
    let dr_constant = oc.w;
    var z = q;
    var dr = 1.0;
    var r = length(z);
    var trap = TRAP_START;
    let bailout = pr(is_world, obj, P_BAILOUT);
    let cfg = load_bulb_config(is_world, obj);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);
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
        if !bulb_step_cfg(cfg, rot, &z, &dr, r, c, dr_constant) {
            break;
        }
        r = length(z);
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, bulb_estimate(r, dr, bailout), trap);
}

fn bulb_distance_endpoints(is_world: bool, obj: u32, q: vec3<f32>) -> vec2<f32> {
    let oc = orbit_constant(is_world, obj, q);
    let c = oc.xyz;
    let dr_constant = oc.w;
    var z = q;
    var dr = 1.0;
    var r = length(z);
    let bailout = pr(is_world, obj, P_BAILOUT);
    let cfg = load_bulb_config(is_world, obj);
    let rot = load_iter_rot(is_world, obj);
    let whole = u32(floor(max(pr(is_world, obj, P_ITERATIONS), 0.0)));
    let run = whole + 1u;
    var d0 = 0.0;
    var has_d0 = false;
    for (var i = 0u; i < run; i = i + 1u) {
        if i + 1u == run {
            d0 = bulb_estimate(r, dr, bailout);
            has_d0 = true;
        }
        if r > bailout {
            break;
        }
        if r < 1.0e-8 {
            return vec2<f32>(0.0, 0.0);
        }
        if !bulb_step_cfg(cfg, rot, &z, &dr, r, c, dr_constant) {
            break;
        }
        r = length(z);
    }
    let d1 = bulb_estimate(r, dr, bailout);
    return vec2<f32>(select(d1, d0, has_d0), d1);
}

fn box_fold(v: vec3<f32>, fold: f32) -> vec3<f32> {
    return clamp(v, vec3<f32>(-fold), vec3<f32>(fold)) * 2.0 - v;
}

struct BoxConfig {
    scale: f32,
    fold: f32,
    min_r2: f32,
    fixed_r2: f32,
}

fn load_box_config(is_world: bool, obj: u32) -> BoxConfig {
    return BoxConfig(
        pr(is_world, obj, P_BOX_SCALE),
        pr(is_world, obj, P_BOX_FOLD),
        pr(is_world, obj, P_BOX_MIN_R2),
        pr(is_world, obj, P_BOX_FIXED_R2),
    );
}

fn box_step_cfg(
    cfg: BoxConfig,
    rot: IterRot,
    z: ptr<function, vec3<f32>>,
    dr: ptr<function, f32>,
    c: vec3<f32>,
    dr_constant: f32,
) -> bool {
    var t = apply_iter_rot(rot, *z);
    t = box_fold(t, cfg.fold);
    let r2 = dot(t, t);
    var factor: f32;
    if r2 < cfg.min_r2 {
        factor = cfg.fixed_r2 / cfg.min_r2;
    } else if r2 < cfg.fixed_r2 {
        factor = cfg.fixed_r2 / r2;
    } else {
        factor = 1.0;
    }
    let step_scale = factor * cfg.scale;
    let abs_step = abs(step_scale);
    if *dr > DERIVATIVE_LIMIT / abs_step {
        return false;
    }
    *z = t * step_scale + c;
    *dr = (*dr) * abs_step + dr_constant;
    return true;
}

fn box_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let oc = orbit_constant(is_world, obj, q);
    let c = oc.xyz;
    let dr_constant = oc.w;
    var z = q;
    var dr = 1.0;
    var trap = TRAP_START;
    let cfg = load_box_config(is_world, obj);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);
    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, length(z) / dr, trap);
        }
        if !box_step_cfg(cfg, rot, &z, &dr, c, dr_constant) {
            break;
        }
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, length(z) / dr, trap);
}

fn quat_distance(
    is_world: bool,
    obj: u32,
    p: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    var q = vec4<f32>(
        dot(pr3(is_world, obj, P_QUAT_ROWS), p) + pr(is_world, obj, P_QUAT_OFFSET),
        dot(pr3(is_world, obj, P_QUAT_ROWS + 3u), p) + pr(is_world, obj, P_QUAT_OFFSET + 1u),
        dot(pr3(is_world, obj, P_QUAT_ROWS + 6u), p) + pr(is_world, obj, P_QUAT_OFFSET + 2u),
        dot(pr3(is_world, obj, P_QUAT_ROWS + 9u), p) + pr(is_world, obj, P_QUAT_OFFSET + 3u),
    );
    let c = vec4<f32>(
        pr(is_world, obj, P_QUAT_C),
        pr(is_world, obj, P_QUAT_C + 1u),
        pr(is_world, obj, P_QUAT_C + 2u),
        pr(is_world, obj, P_QUAT_C + 3u),
    );
    var dr = 1.0;
    var r = length(q);
    var trap = TRAP_START;
    let bailout = pr(is_world, obj, P_BAILOUT);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);
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
        let growth = 2.0 * r;
        if dr > DERIVATIVE_LIMIT / growth {
            break;
        }
        dr = dr * growth;
        let v = q.yzw;
        let two_x = 2.0 * q.x;
        q = vec4<f32>(
            q.x * q.x - dot(v, v) + c.x,
            two_x * v.x + c.y,
            two_x * v.y + c.z,
            two_x * v.z + c.w,
        );
        r = length(q);
        if trap_mode != 0u {
            trap = min(trap, eval_trap4(trap_cfg, q));
        }
    }
    return fraction_blend(frac, bulb_estimate(r, dr, bailout), trap);
}

struct KifsConfig {
    kind: u32,
    fold_height: f32,
    scale: f32,
    bound: f32,
    shift: vec3<f32>,
}

fn load_kifs_config(is_world: bool, obj: u32) -> KifsConfig {
    return KifsConfig(
        u32(pr(is_world, obj, P_KIFS_KIND)),
        pr(is_world, obj, P_KIFS_FOLD_HEIGHT),
        pr(is_world, obj, P_KIFS_SCALE),
        pr(is_world, obj, P_KIFS_BOUND),
        pr3(is_world, obj, P_KIFS_SHIFT),
    );
}

fn kifs_fold_cfg(cfg: KifsConfig, v: vec3<f32>) -> vec3<f32> {
    var x = v.x;
    var y = v.y;
    var z = v.z;
    if cfg.kind == 0u {
        if x + y < 0.0 {
            let nx = -y;
            let ny = -x;
            x = nx;
            y = ny;
        }
        if x + z < 0.0 {
            let nx = -z;
            let nz = -x;
            x = nx;
            z = nz;
        }
        if y + z < 0.0 {
            let ny = -z;
            let nz = -y;
            y = ny;
            z = nz;
        }
        return vec3<f32>(x, y, z);
    }
    x = abs(x);
    y = abs(y);
    z = abs(z);
    if x < y {
        let t = x;
        x = y;
        y = t;
    }
    if x < z {
        let t = x;
        x = z;
        z = t;
    }
    if y < z {
        let t = y;
        y = z;
        z = t;
    }
    if cfg.kind == 2u {
        z = cfg.fold_height - abs(z - cfg.fold_height);
    }
    return vec3<f32>(x, y, z);
}

fn unit_box_distance(p: vec3<f32>) -> f32 {
    let q = abs(p) - vec3<f32>(1.0);
    let outside = vec3<f32>(max(q.x, 0.0), max(q.y, 0.0), max(q.z, 0.0));
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return length(outside) + inside;
}

fn kifs_step_cfg(
    cfg: KifsConfig,
    rot: IterRot,
    z: ptr<function, vec3<f32>>,
    dr: ptr<function, f32>,
) -> bool {
    if *dr > DERIVATIVE_LIMIT / cfg.scale {
        return false;
    }
    var t = apply_iter_rot(rot, *z);
    t = kifs_fold_cfg(cfg, t);
    *z = t * cfg.scale - cfg.shift;
    *dr = (*dr) * cfg.scale;
    return true;
}

fn kifs_estimate_cfg(cfg: KifsConfig, z: vec3<f32>, dr: f32) -> f32 {
    if cfg.kind == 2u {
        return unit_box_distance(z) / dr;
    }
    return (length(z) - cfg.bound) / dr;
}

fn kifs_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    var z = q;
    var dr = 1.0;
    var trap = TRAP_START;
    let cfg = load_kifs_config(is_world, obj);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);
    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, kifs_estimate_cfg(cfg, z, dr), trap);
        }
        if !kifs_step_cfg(cfg, rot, &z, &dr) {
            break;
        }
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, kifs_estimate_cfg(cfg, z, dr), trap);
}

fn kleinian_wrap(x: f32, a: f32, s: f32) -> f32 {
    return x - s * floor(x / s + 0.5 * (1.0 - a));
}

fn exact_sign(x: f32) -> f32 {
    if x < 0.0 {
        return -1.0;
    }
    if x > 0.0 {
        return 1.0;
    }
    return 0.0;
}

fn kleinian_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let a = pr(is_world, obj, P_KLEIN_A);
    let b = pr(is_world, obj, P_KLEIN_B);
    let f = select(1.0, -1.0, b < 0.0);
    let disc = max(a * a - b * (b - 2.0), 0.0);
    let e = 0.5 * f * sqrt(disc);
    let l1 = 0.5 * a + e;
    let l2 = b - 1.0;
    let l3 = 0.5 * a - e;
    let skew = pr(is_world, obj, P_KLEIN_SKEW);
    let amp = pr(is_world, obj, P_KLEIN_LINE_AMP);
    let rate = pr(is_world, obj, P_KLEIN_LINE_RATE);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    let face_camera = vec3<f32>(q.x, q.z, -q.y);
    let oc = orbit_constant(is_world, obj, face_camera);
    var z = oc.xyz;
    var dr = 1.0;
    var trap = TRAP_START;
    var frac = fraction_new(is_world, obj, iter_override);
    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, min(z.y, 0.05) / max(dr, 1.0), trap);
        }
        z = apply_iter_rot(rot, z);
        if amp != 0.0 {
            let push = exp(-abs(z.y));
            z.x = z.x + push * amp * sin(rate * z.x);
            z.z = z.z + push * amp * cos(rate * z.z);
        }
        z.x = z.x + b / a * z.y;
        z.x = kleinian_wrap(z.x, skew, 2.0);
        z.z = kleinian_wrap(z.z, skew, 2.0);
        z.x = z.x - b / a * z.y;
        let line = l2 + l1 * z.y * (l3 - z.y);
        if z.x >= line {
            z = vec3<f32>(-b - z.x, a - z.y, -z.z);
        }
        let inv = 1.0 / max(dot(z, z), 1.0e-30);
        if dr > DERIVATIVE_LIMIT / inv {
            break;
        }
        z = z * inv;
        dr = dr * inv;
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
        let x0 = z.x + 0.5 * b;
        let fl = floor(z.z + 0.5);
        let sq = x0 * x0 + z.y * z.y + (z.z - fl) * (z.z - fl);
        let diff = z.z - 0.5 - fl;
        if exact_sign(diff) * sq < diff {
            break;
        }
    }
    return fraction_blend(frac, min(z.y, 0.05) / max(dr, 1.0), trap);
}

fn pseudo_kleinian_estimate(z: vec3<f32>, dr: f32, offset: f32, thickness: f32) -> f32 {
    let o = z - vec3<f32>(0.0, 0.0, offset);
    let rxy = length(o.xy);
    let thingy = abs(rxy * o.z) - thickness;
    return 0.5 * thingy / max(sqrt(dot(o, o) + abs(thickness)), 1.0e-30) / max(dr, 1.0e-30);
}

fn pseudo_kleinian_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let csize = pr3(is_world, obj, P_PK_BOX);
    let size = pr(is_world, obj, P_PK_SIZE);
    let c = pr3(is_world, obj, P_PK_C);
    let offset = pr(is_world, obj, P_PK_OFFSET);
    let thickness = pr(is_world, obj, P_PK_THICKNESS);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var p = q;
    var dr = 1.0;
    var trap = TRAP_START;
    var frac = fraction_new(is_world, obj, iter_override);
    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, pseudo_kleinian_estimate(p, dr, offset, thickness), trap);
        }
        p = apply_iter_rot(rot, p);
        p = clamp(p, -csize, csize) * 2.0 - p;
        let r2 = dot(p, p);
        if r2 <= 0.0 {
            break;
        }
        let k = max(size / r2, 1.0);
        if dr > DERIVATIVE_LIMIT / k {
            break;
        }
        p = p * k + c;
        dr = dr * k;
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, p));
        }
    }
    return fraction_blend(frac, pseudo_kleinian_estimate(p, dr, offset, thickness), trap);
}

fn apollonian_step_cfg(
    scale: f32,
    rot: IterRot,
    z: ptr<function, vec3<f32>>,
    dr: ptr<function, f32>,
) -> bool {
    var p = apply_iter_rot(rot, *z);
    p = vec3<f32>(
        2.0 * fract(0.5 * (p.x + 1.0)) - 1.0,
        2.0 * fract(0.5 * (p.y + 1.0)) - 1.0,
        2.0 * fract(0.5 * (p.z + 1.0)) - 1.0,
    );
    let r2 = dot(p, p);
    if r2 <= 0.0 {
        return false;
    }
    let k = max(scale / r2, 0.1);
    if *dr > DERIVATIVE_LIMIT / k {
        return false;
    }
    *z = p * k;
    *dr = (*dr) * k;
    return true;
}

fn apollonian_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    var z = q + vec3<f32>(0.0, 1.0, 0.0);
    var dr = 1.0;
    var trap = TRAP_START;
    let scale = pr(is_world, obj, P_APOLLO_SCALE);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);
    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, 0.25 * abs(z.y) / dr, trap);
        }
        if !apollonian_step_cfg(scale, rot, &z, &dr) {
            break;
        }
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, 0.25 * abs(z.y) / dr, trap);
}

fn hybrid_estimate(r: f32, dr: f32, bailout: f32) -> f32 {
    if r <= bailout || dr <= 0.0 {
        return 0.0;
    }
    return 0.5 * r / dr;
}

fn hybrid_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    var z = q;
    let c = q;
    var dr = 1.0;
    var trap = TRAP_START;
    let bailout = pr(is_world, obj, P_BAILOUT);
    let count = clamp(u32(pr(is_world, obj, P_HYBRID_COUNT)), 1u, 4u);
    let steps = u32(pr(is_world, obj, P_HYBRID_STEPS));
    let bulb_cfg = load_bulb_config(is_world, obj);
    let box_cfg = load_box_config(is_world, obj);
    let kifs_cfg = load_kifs_config(is_world, obj);
    let apollo_scale = pr(is_world, obj, P_APOLLO_SCALE);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj, iter_override);
    for (var i = 0u; i < frac.run; i = i + 1u) {
        let r = length(z);
        if fraction_due(frac, i) {
            fraction_save(&frac, hybrid_estimate(r, dr, bailout), trap);
        }
        if r > bailout {
            break;
        }
        let step = (steps >> (3u * (i % count))) & 7u;
        var ok = false;
        if step == 0u {
            ok = true;
        } else if step == 1u {
            ok = r >= 1.0e-8 && bulb_step_cfg(bulb_cfg, rot, &z, &dr, r, c, 1.0);
        } else if step == 2u {
            ok = box_step_cfg(box_cfg, rot, &z, &dr, c, 1.0);
        } else if step == 3u {
            ok = kifs_step_cfg(kifs_cfg, rot, &z, &dr);
        } else if step == 4u {
            ok = apollo_scale != 0.0 && apollonian_step_cfg(apollo_scale, rot, &z, &dr);
        }
        if !ok {
            break;
        }
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, hybrid_estimate(length(z), dr, bailout), trap);
}

fn family_distance(
    is_world: bool,
    obj: u32,
    family_override: u32,
    p: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let family = resolved_family(is_world, obj, family_override);
    switch family {
        case FAMILY_BOX: {
            return box_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_QUAT: {
            return quat_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_KIFS: {
            return kifs_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_KLEINIAN: {
            return kleinian_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_PSEUDO_KLEINIAN: {
            return pseudo_kleinian_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_APOLLONIAN: {
            return apollonian_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_HYBRID: {
            return hybrid_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_AMAZING_SURFACE: {
            return amazing_surface_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_BRISTORBROT: {
            return bristorbrot_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_BURNING_SHIP: {
            return burning_ship_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_KALEIDO_TWIST: {
            return kaleido_twist_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_POLYHEDRAL_KIFS: {
            return polyhedral_kifs_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_HARMONIC_BULB: {
            return harmonic_bulb_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_MANDELBAR: {
            return mandelbar_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_MOBIUS_IFS: {
            return mobius_ifs_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_GYROID_TPMS: {
            return gyroid_tpms_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_LIDINOID_TPMS: {
            return lidinoid_tpms_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_ATTRACTOR_TUBE: {
            return attractor_tube_distance(is_world, obj, p, trap_mode, iter_override);
        }
        case FAMILY_TORUS_KNOT: {
            return torus_knot_braid_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_HOPF_FIBRATION: {
            return hopf_fibration_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_DUPIN_CYCLIDE: {
            return dupin_cyclide_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_SUPERFORMULA: {
            return superformula_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_PHYLLOTAXIS: {
            return phyllotaxis_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_NAUTILUS: {
            return nautilus_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_VORONOI_FOAM: {
            return voronoi_foam_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_CRYSTAL_GEODE: {
            return crystal_geode_distance(is_world, obj, p, trap_mode);
        }
        case FAMILY_ASTROLABE: {
            return astrolabe_distance(is_world, obj, p, trap_mode);
        }
        default: {
            return bulb_distance(is_world, obj, p, trap_mode, iter_override);
        }
    }
}

fn formula_bound(is_world: bool, obj: u32, p: vec3<f32>, d: f32) -> f32 {
    let bound = pr(is_world, obj, P_BOUND_RADIUS);
    if bound > 0.0 {
        return max(d, length(p) - bound);
    }
    return d;
}

fn object_point(is_world: bool, obj: u32, x: vec3<f32>) -> vec3<f32> {
    if is_world {
        let base = obj * OBJECT_STRIDE + O_INVERSE;
        let rx = vec3<f32>(objects[base], objects[base + 1u], objects[base + 2u]);
        let tx = objects[base + 3u];
        let ry = vec3<f32>(objects[base + 4u], objects[base + 5u], objects[base + 6u]);
        let ty = objects[base + 7u];
        let rz = vec3<f32>(objects[base + 8u], objects[base + 9u], objects[base + 10u]);
        let tz = objects[base + 11u];
        return vec3<f32>(dot(rx, x) + tx, dot(ry, x) + ty, dot(rz, x) + tz);
    }
    let rel = (x - param3(P_OBJ_OFFSET)) / param(P_OBJ_SCALE);
    return vec3<f32>(
        dot(param3(P_OBJ_AXES), rel),
        dot(param3(P_OBJ_AXES + 3u), rel),
        dot(param3(P_OBJ_AXES + 6u), rel),
    );
}

fn object_vector(obj: u32, v: vec3<f32>) -> vec3<f32> {
    let base = obj * OBJECT_STRIDE + O_INVERSE;
    let rx = vec3<f32>(objects[base], objects[base + 1u], objects[base + 2u]);
    let ry = vec3<f32>(objects[base + 4u], objects[base + 5u], objects[base + 6u]);
    let rz = vec3<f32>(objects[base + 8u], objects[base + 9u], objects[base + 10u]);
    return vec3<f32>(dot(rx, v), dot(ry, v), dot(rz, v));
}

fn mask_point(base: u32, x: vec3<f32>) -> vec3<f32> {
    let m = base + M_INVERSE;
    let rx = vec3<f32>(objects[m], objects[m + 1u], objects[m + 2u]);
    let tx = objects[m + 3u];
    let ry = vec3<f32>(objects[m + 4u], objects[m + 5u], objects[m + 6u]);
    let ty = objects[m + 7u];
    let rz = vec3<f32>(objects[m + 8u], objects[m + 9u], objects[m + 10u]);
    let tz = objects[m + 11u];
    return vec3<f32>(dot(rx, x) + tx, dot(ry, x) + ty, dot(rz, x) + tz);
}

fn cell_removed(cell: vec3<i32>, density: f32, seed: u32) -> bool {
    var h = ss_lowbias32(bitcast<u32>(cell.x) ^ seed);
    h = ss_lowbias32(h ^ bitcast<u32>(cell.y));
    h = ss_lowbias32(h ^ bitcast<u32>(cell.z));
    return f32(h) * (1.0 / 4294967296.0) < density;
}

fn shape_distance(shape: u32, q: vec3<f32>, size: vec3<f32>) -> f32 {
    if shape == MASK_SPHERE {
        return length(q) - size.x;
    }
    let b = abs(q) - size;
    let outside = vec3<f32>(max(b.x, 0.0), max(b.y, 0.0), max(b.z, 0.0));
    let inside = min(max(b.x, max(b.y, b.z)), 0.0);
    return length(outside) + inside;
}

fn ring_basis(axis: u32) -> array<vec3<f32>, 3> {
    switch axis {
        case 0u: {
            return array<vec3<f32>, 3>(
                vec3<f32>(0.0, 1.0, 0.0),
                vec3<f32>(0.0, 0.0, 1.0),
                vec3<f32>(1.0, 0.0, 0.0),
            );
        }
        case 2u: {
            return array<vec3<f32>, 3>(
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, 1.0, 0.0),
                vec3<f32>(0.0, 0.0, 1.0),
            );
        }
        default: {
            return array<vec3<f32>, 3>(
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, 0.0, 1.0),
                vec3<f32>(0.0, 1.0, 0.0),
            );
        }
    }
}

fn mask_array_angle(index: u32, count: u32, sweep_deg: f32) -> f32 {
    if count <= 1u {
        return 0.0;
    }
    let sweep = clamp(sweep_deg, -360.0, 360.0);
    let step = select(sweep / f32(count - 1u), sweep / f32(count), abs(sweep) >= 359.999);
    return radians(step * f32(index));
}

fn ring_point(q: vec3<f32>, axis: u32, angle: f32, radius: f32, align: bool) -> vec3<f32> {
    let basis = ring_basis(axis);
    let u = basis[0];
    let v = basis[1];
    let w = basis[2];
    let s = sin(angle);
    let c = cos(angle);
    let radial = u * c + v * s;
    let rel = q - radial * radius;
    if !align {
        return rel;
    }
    let tangent = v * c - u * s;
    return u * dot(rel, radial) + v * dot(rel, tangent) + w * dot(rel, w);
}

fn array_copy_distance(
    base: u32,
    shape: u32,
    q: vec3<f32>,
    size: vec3<f32>,
    mode: u32,
    count: u32,
    index: u32,
) -> f32 {
    var local = q;
    if mode == ARRAY_LINEAR {
        let off = vec3<f32>(
            objects[base + M_ARRAY_OFFSET],
            objects[base + M_ARRAY_OFFSET + 1u],
            objects[base + M_ARRAY_OFFSET + 2u],
        );
        local = q - off * f32(index);
    } else if mode == ARRAY_RADIAL {
        let axis = u32(objects[base + M_ARRAY_AXIS]);
        let sweep = objects[base + M_ARRAY_SWEEP];
        let radius = objects[base + M_ARRAY_RADIUS];
        let align = objects[base + M_ARRAY_ALIGN] != 0.0;
        let angle = mask_array_angle(index, count, sweep);
        local = ring_point(q, axis, angle, radius, align);
    }
    return shape_distance(shape, local, size);
}

fn array_union(base: u32, shape: u32, q: vec3<f32>, size: vec3<f32>) -> f32 {
    let mode = u32(objects[base + M_ARRAY]);
    let count = clamp(u32(objects[base + M_ARRAY_COUNT]), 1u, ARRAY_MAX);
    var d = array_copy_distance(base, shape, q, size, mode, count, 0u);
    for (var i = 1u; i < count; i = i + 1u) {
        d = min(d, array_copy_distance(base, shape, q, size, mode, count, i));
    }
    return d;
}

fn mask_distance(base: u32, x: vec3<f32>) -> f32 {
    let q = mask_point(base, x);
    let scale = objects[base + M_SCALE];
    let shape = u32(objects[base + M_SHAPE]);
    let size = vec3<f32>(
        objects[base + M_SIZE],
        objects[base + M_SIZE + 1u],
        objects[base + M_SIZE + 2u],
    );
    if shape == MASK_CELLS {
        let pitch = max(size.x, 1.0e-4);
        let fill = clamp(size.y, 0.05, 1.0);
        let half = vec3<f32>(0.5 * pitch * fill);
        let density = clamp(objects[base + M_DENSITY], 0.0, 1.0);
        if density <= 0.0 {
            return -TRAP_START;
        }
        let seed = bitcast<u32>(objects[base + M_SEED]);
        let inv = 1.0 / pitch;
        let cx = i32(floor(q.x * inv));
        let cy = i32(floor(q.y * inv));
        let cz = i32(floor(q.z * inv));
        var d = 1.5 * pitch;
        for (var dz = -1; dz <= 1; dz = dz + 1) {
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    let cell = vec3<i32>(cx + dx, cy + dy, cz + dz);
                    if cell_removed(cell, density, seed) {
                        let center = vec3<f32>(
                            (f32(cell.x) + 0.5) * pitch,
                            (f32(cell.y) + 0.5) * pitch,
                            (f32(cell.z) + 0.5) * pitch,
                        );
                        d = min(d, shape_distance(MASK_BOX, q - center, half));
                    }
                }
            }
        }
        return -d * scale;
    }
    var d = array_union(base, shape, q, size) * scale;
    if objects[base + M_INVERT] != 0.0 {
        d = -d;
    }
    return d;
}

fn masked(obj: u32, x: vec3<f32>, unmasked: f32) -> f32 {
    let base = obj * OBJECT_STRIDE;
    let count = min(u32(objects[base + O_MASK_COUNT]), MAX_MASKS);
    var d = unmasked;
    for (var m = 0u; m < count; m = m + 1u) {
        let m_base = base + O_MASKS + m * MASK_STRIDE;
        let shape = u32(objects[m_base + M_SHAPE]);
        let pitch = objects[m_base + M_SIZE];
        let scale = objects[m_base + M_SCALE];
        if !(shape == MASK_CELLS && d >= 1.5 * pitch * scale) {
            d = max(d, mask_distance(m_base, x));
        }
    }
    return d;
}

fn distance_and_trap(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let q = object_point(is_world, obj, x);
    let s = pr(is_world, obj, P_INVERT_STRENGTH);
    if s != 0.0 {
        let center = pr3(is_world, obj, P_INVERT_CENTER);
        let radius = pr(is_world, obj, P_INVERT_RADIUS);
        let u = q - center;
        let r2 = max(dot(u, u), 1.0e-30);
        let g = (1.0 - s) + s * radius * radius / r2;
        let p = center + u * g;
        let dt = family_distance(is_world, obj, family_override, p, trap_mode, iter_override);
        return vec2<f32>(formula_bound(is_world, obj, p, dt.x) / g, dt.y);
    }
    let dt = family_distance(is_world, obj, family_override, q, trap_mode, iter_override);
    return vec2<f32>(formula_bound(is_world, obj, q, dt.x), dt.y);
}

fn clipped_distance_and_trap_q(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    q: vec3<f32>,
    len_q: f32,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let s = pr(is_world, obj, P_INVERT_STRENGTH);
    var d: f32;
    var trap: f32;
    if s != 0.0 {
        let center = pr3(is_world, obj, P_INVERT_CENTER);
        let radius = pr(is_world, obj, P_INVERT_RADIUS);
        let u = q - center;
        let r2 = max(dot(u, u), 1.0e-30);
        let g = (1.0 - s) + s * radius * radius / r2;
        let p = center + u * g;
        let dt = family_distance(is_world, obj, family_override, p, trap_mode, iter_override);
        d = formula_bound(is_world, obj, p, dt.x) / g;
        trap = dt.y;
    } else {
        let dt = family_distance(is_world, obj, family_override, q, trap_mode, iter_override);
        d = formula_bound(is_world, obj, q, dt.x);
        trap = dt.y;
    }
    if is_world {
        let base = obj * OBJECT_STRIDE;
        let clip_r = objects[base + O_CLIP_RADIUS];
        d = max(d, len_q - clip_r);
        let scale = objects[base + O_DISTANCE_SCALE];
        return vec2<f32>(masked(obj, x, d * scale), trap);
    }
    return vec2<f32>(d * param(P_OBJ_SCALE), trap);
}

fn clipped_distance_and_trap(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    trap_mode: u32,
    iter_override: f32,
) -> vec2<f32> {
    let q = object_point(is_world, obj, x);
    let len_q = select(0.0, length(q), is_world);
    return clipped_distance_and_trap_q(is_world, obj, family_override, x, q, len_q, trap_mode, iter_override);
}

struct SceneSample {
    d: f32,
    trap: f32,
    object: u32,
}

fn footprint(base: f32, slope: f32, t: f32, x: vec3<f32>) -> f32 {
    return max(base + slope * t, length(x) * 0.0000019073486);
}

fn march_sample(
    is_world: bool,
    family_override: u32,
    x: vec3<f32>,
    kind: u32,
) -> SceneSample {
    if is_world {
        let count = u32(param(P_OBJECT_COUNT));
        if count <= 1u {
            let color_mode = u32(pr(true, 0u, P_COLOR_MODE));
            let trap_mode = select(color_mode, 0u, kind == RAY_VISIBILITY);
            let dt = clipped_distance_and_trap(true, 0u, family_override, x, trap_mode, -1.0);
            return SceneSample(max(dt.x, 0.0), dt.y, 0u);
        }
        var first_q = object_point(true, 0u, x);
        var first_len_q = length(first_q);
        var min_reach = (first_len_q - objects[O_CLIP_RADIUS]) * objects[O_DISTANCE_SCALE];
        var first = 0u;
        for (var i = 1u; i < count; i = i + 1u) {
            let row = i * OBJECT_STRIDE;
            let q = object_point(true, i, x);
            let len_q = length(q);
            let reach = (len_q - objects[row + O_CLIP_RADIUS]) * objects[row + O_DISTANCE_SCALE];
            if reach < min_reach {
                min_reach = reach;
                first = i;
                first_q = q;
                first_len_q = len_q;
            }
        }
        let first_color_mode = u32(pr(true, first, P_COLOR_MODE));
        let first_trap_mode = select(first_color_mode, 0u, kind == RAY_VISIBILITY);
        let first_dt = clipped_distance_and_trap_q(
            true,
            first,
            family_override,
            x,
            first_q,
            first_len_q,
            first_trap_mode,
            -1.0,
        );
        var best = SceneSample(max(first_dt.x, 0.0), first_dt.y, first);
        for (var i = 0u; i < count; i = i + 1u) {
            if i == first {
                continue;
            }
            let row = i * OBJECT_STRIDE;
            let q = object_point(true, i, x);
            let scale = max(objects[row + O_DISTANCE_SCALE], 1.0e-30);
            let clip_r = objects[row + O_CLIP_RADIUS];
            let rhs = clip_r + best.d / scale;
            let q2 = dot(q, q);
            if rhs >= 0.0 && q2 >= rhs * rhs {
                continue;
            }
            let len_q = sqrt(q2);
            let color_mode = u32(pr(true, i, P_COLOR_MODE));
            let trap_mode = select(color_mode, 0u, kind == RAY_VISIBILITY);
            let dt = clipped_distance_and_trap_q(true, i, family_override, x, q, len_q, trap_mode, -1.0);
            let d = max(dt.x, 0.0);
            if d < best.d || (d == best.d && i < best.object) {
                best = SceneSample(d, dt.y, i);
            }
        }
        return best;
    }
    let color_mode = u32(param(P_COLOR_MODE));
    let trap_mode = select(color_mode, 0u, kind == RAY_VISIBILITY);
    let dt = clipped_distance_and_trap(false, 0u, family_override, x, trap_mode, -1.0);
    return SceneSample(max(dt.x, 0.0), dt.y, 0u);
}

fn fractional_bulb(is_world: bool, obj: u32, family_override: u32) -> bool {
    let family = resolved_family(is_world, obj, family_override);
    if family != FAMILY_BULB || pr(is_world, obj, P_INVERT_STRENGTH) != 0.0 {
        return false;
    }
    let n = max(pr(is_world, obj, P_ITERATIONS), 0.0);
    return n - floor(n) > 0.0;
}

fn has_fractional_bulb(is_world: bool, family_override: u32) -> bool {
    if is_world {
        let count = u32(param(P_OBJECT_COUNT));
        for (var i = 0u; i < count; i = i + 1u) {
            if fractional_bulb(true, i, family_override) {
                return true;
            }
        }
        return false;
    }
    return fractional_bulb(false, 0u, family_override);
}

fn fractional_bulb_endpoints(is_world: bool, obj: u32, x: vec3<f32>) -> vec2<f32> {
    let q = object_point(is_world, obj, x);
    let end = bulb_distance_endpoints(is_world, obj, q);
    let d0 = formula_bound(is_world, obj, q, end.x);
    let d1 = formula_bound(is_world, obj, q, end.y);
    if is_world {
        let base = obj * OBJECT_STRIDE;
        let clip = length(q) - objects[base + O_CLIP_RADIUS];
        let scale = objects[base + O_DISTANCE_SCALE];
        return vec2<f32>(
            masked(obj, x, max(d0, clip) * scale),
            masked(obj, x, max(d1, clip) * scale),
        );
    }
    let scale = param(P_OBJ_SCALE);
    return vec2<f32>(d0 * scale, d1 * scale);
}

fn fractional_object_march_distance(
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    sample_d: f32,
    nearest: u32,
) -> f32 {
    if fractional_bulb(true, obj, family_override) {
        let end = fractional_bulb_endpoints(true, obj, x);
        let d0 = max(end.x, 0.0);
        let d1 = max(end.y, 0.0);
        return max(d0, d1);
    }
    if obj == nearest {
        return sample_d;
    }
    return max(clipped_distance_and_trap(true, obj, family_override, x, 0u, -1.0).x, 0.0);
}

fn fractional_march_distance(
    is_world: bool,
    family_override: u32,
    x: vec3<f32>,
    sample_d: f32,
    nearest: u32,
) -> f32 {
    if is_world {
        let count = u32(param(P_OBJECT_COUNT));
        let first = min(nearest, max(count, 1u) - 1u);
        var best = fractional_object_march_distance(first, family_override, x, sample_d, nearest);
        for (var i = 0u; i < count; i = i + 1u) {
            if i == first {
                continue;
            }
            let row = i * OBJECT_STRIDE;
            let q = object_point(true, i, x);
            let scale = max(objects[row + O_DISTANCE_SCALE], 1.0e-30);
            let rhs = objects[row + O_CLIP_RADIUS] + best / scale;
            if rhs >= 0.0 && dot(q, q) >= rhs * rhs {
                continue;
            }
            let d = fractional_object_march_distance(i, family_override, x, sample_d, nearest);
            if d < best {
                best = d;
            }
        }
        return best;
    }
    if fractional_bulb(false, 0u, family_override) {
        let end = fractional_bulb_endpoints(false, 0u, x);
        return max(max(end.x, 0.0), max(end.y, 0.0));
    }
    return sample_d;
}

fn march_span(is_world: bool, origin: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    let max_distance = param(P_MAX_DISTANCE);
    let r = param(P_CLIP_RADIUS);
    if (!is_world && param(P_DIRECT) != 0.0) || single_no_clip() || r <= 0.0 {
        return vec2<f32>(0.0, max_distance);
    }
    let oc = origin - param3(P_CLIP_CENTER);
    let b = dot(oc, dir);
    let disc = b * b - (dot(oc, oc) - r * r);
    if disc <= 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let root = sqrt(disc);
    let t_exit = -b + root;
    if t_exit <= 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    let global_span = vec2<f32>(max(-b - root, 0.0), min(t_exit, max_distance));
    if is_world {
        let count = u32(param(P_OBJECT_COUNT));
        if count > 1u && count <= 16u {
            var min_enter = max_distance + 1.0;
            var max_exit = -1.0;
            for (var i = 0u; i < count; i = i + 1u) {
                let row = i * OBJECT_STRIDE;
                let obj_r = objects[row + O_CLIP_RADIUS];
                if obj_r <= 0.0 {
                    return global_span;
                }
                let q0 = object_point(true, i, origin);
                let qd = object_vector(i, dir);
                let a = dot(qd, qd);
                if a <= 1.0e-20 {
                    continue;
                }
                let hb = dot(q0, qd);
                let c = dot(q0, q0) - obj_r * obj_r;
                let d_disc = hb * hb - a * c;
                if d_disc <= 0.0 {
                    continue;
                }
                let inv_a = 1.0 / a;
                let d_root = sqrt(d_disc);
                let t1 = (-hb + d_root) * inv_a;
                if t1 <= 0.0 {
                    continue;
                }
                let t0 = max((-hb - d_root) * inv_a, 0.0);
                if t0 <= max_distance {
                    min_enter = min(min_enter, t0);
                    max_exit = max(max_exit, min(t1, max_distance));
                }
            }
            if min_enter > max_exit {
                return vec2<f32>(1.0, 0.0);
            }
            return vec2<f32>(max(global_span.x, min_enter), min(global_span.y, max_exit));
        }
    }
    return global_span;
}

fn cone_start(is_world: bool, family_override: u32, origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    let span = march_span(is_world, origin, dir);
    if span.x > span.y {
        return 0.0;
    }
    let t_enter = span.x;
    let t_exit = span.y;
    let step_cap = 2.0 * (t_exit - t_enter) / STEP_CAP_STEPS;
    let slope = param(P_FOOTPRINT);
    let w = 2.0 * param(P_HALF_W) / param(P_WIDTH);
    let h = 2.0 * param(P_HALF_H) / param(P_HEIGHT);
    let cone = 0.5 * sqrt(w * w + h * h);
    var t = t_enter;
    for (var i = 0u; i < CONE_START_STEPS; i = i + 1u) {
        if t > t_exit {
            break;
        }
        let point = origin + dir * t;
        let d = march_sample(is_world, family_override, point, RAY_VISIBILITY).d;
        let eps = footprint(0.0, slope, t, point);
        let step = min((0.5 * d - cone * t - eps) / (1.0 + cone), step_cap);
        if step <= 0.0 {
            return t;
        }
        t = t + step;
    }
    return t;
}

struct MarchResult {
    hit: bool,
    unresolved: bool,
    point: vec3<f32>,
    trap: f32,
    eps: f32,
    ratio: f32,
    first: bool,
    object: u32,
}

fn march_miss(origin: vec3<f32>) -> MarchResult {
    return MarchResult(false, false, origin, TRAP_START, 0.0, 0.0, false, 0u);
}

fn march_visibility(
    is_world: bool,
    family_override: u32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    base: f32,
) -> MarchResult {
    let max_steps = u32(param(P_SECONDARY_STEPS));
    let span = march_span(is_world, origin, dir);
    if span.x > span.y {
        return march_miss(origin);
    }
    let t_enter = span.x;
    let t_exit = span.y;
    let step_cap = 2.0 * (t_exit - t_enter) / STEP_CAP_STEPS;
    let step_factor = param(P_STEP_FACTOR);
    let frac_bulb = has_fractional_bulb(is_world, family_override);
    var t = t_enter;
    var prev_t = t_enter;
    var prev_rad = 0.0;
    var last_step = 0.0;
    var relax = select(1.25, 1.0, frac_bulb);
    for (var i = 0u; i < max_steps; i = i + 1u) {
        if t > t_exit {
            return march_miss(origin);
        }
        let point = origin + dir * t;
        let eps = footprint(base, 0.0, t, point);
        let s = march_sample(is_world, family_override, point, RAY_VISIBILITY);
        var d = s.d;
        let curr_rad = step_factor * d;
        if prev_rad > 0.0 && prev_rad + curr_rad < last_step {
            t = prev_t + prev_rad;
            prev_rad = 0.0;
            relax = 1.0;
            continue;
        }
        if d < eps {
            return MarchResult(true, false, point, TRAP_START, eps, 0.0, i == 0u, 0u);
        }
        var march_d = d;
        if frac_bulb && d < 4.0 * eps {
            march_d = min(d, fractional_march_distance(is_world, family_override, point, d, s.object));
        }
        let unrelaxed = min(max(step_factor * march_d, 0.5 * eps), step_cap);
        prev_t = t;
        if d > 4.0 * eps && relax > 1.0 {
            prev_rad = unrelaxed;
            last_step = min(unrelaxed * relax, step_cap);
        } else {
            prev_rad = 0.0;
            last_step = unrelaxed;
        }
        t = t + last_step;
    }
    if t > t_exit {
        return march_miss(origin);
    }
    return MarchResult(false, true, origin, TRAP_START, 0.0, 0.0, false, 0u);
}

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
    retreat: vec3<f32>,
}

fn leave_ray(
    point: vec3<f32>,
    normal: vec3<f32>,
    dir_in: vec3<f32>,
    dir_out: vec3<f32>,
    eps: f32,
) -> Ray {
    let offset = 4.0 * eps;
    if dot(dir_out, normal) < 0.0 {
        let origin = point - normal * offset;
        return Ray(origin, dir_out, origin);
    }
    return Ray(point + normal * offset, dir_out, point - dir_in * offset);
}

fn visibility_leaving(
    is_world: bool,
    family_override: u32,
    origin: vec3<f32>,
    retreat: vec3<f32>,
    dir: vec3<f32>,
    base: f32,
) -> MarchResult {
    let m = march_visibility(is_world, family_override, origin, dir, base);
    let r = retreat - origin;
    if m.first && dot(r, r) > 0.0 {
        return march_visibility(is_world, family_override, retreat, dir, base);
    }
    return m;
}

fn march(
    is_world: bool,
    family_override: u32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    kind: u32,
    base: f32,
    slope: f32,
    start: f32,
) -> MarchResult {
    let max_steps = select(u32(param(P_SECONDARY_STEPS)), u32(param(P_MAX_STEPS)), kind == RAY_PRIMARY);
    let span = march_span(is_world, origin, dir);
    if span.x > span.y {
        return march_miss(origin);
    }
    let t_enter = span.x;
    let t_exit = span.y;
    let step_cap = 2.0 * (t_exit - t_enter) / STEP_CAP_STEPS;
    let step_factor = param(P_STEP_FACTOR);
    let frac_bulb = has_fractional_bulb(is_world, family_override);
    var t = max(start, t_enter);
    var prev_t = 0.0;
    var has_prev = false;
    var prev_rad = 0.0;
    var last_step = 0.0;
    var relax = select(1.25, 1.0, frac_bulb);
    var best = MarchResult(false, true, origin, TRAP_START, 0.0, 0.0, false, 0u);
    var has_best = false;
    var first = true;
    for (var i = 0u; i < max_steps; i = i + 1u) {
        let is_first = first;
        first = false;
        if t > t_exit {
            return march_miss(origin);
        }
        let point = origin + dir * t;
        let eps = footprint(base, slope, t, point);
        let s = march_sample(is_world, family_override, point, kind);
        var d = s.d;
        let curr_rad = step_factor * max(d, 0.0);
        if prev_rad > 0.0 && prev_rad + curr_rad < last_step {
            t = prev_t + prev_rad;
            prev_rad = 0.0;
            relax = 1.0;
            continue;
        }
        var trap = s.trap;
        var hit_object = s.object;
        if d <= 0.0 && kind != RAY_VISIBILITY {
            if !has_prev && t > t_enter {
                t = t_enter;
                first = true;
                continue;
            }
            if has_prev {
                var lo = prev_t;
                var hi = t;
                for (var r = 0u; r < HIT_REFINEMENTS; r = r + 1u) {
                    if hi - lo <= 0.5 * eps {
                        break;
                    }
                    let mid = 0.5 * (lo + hi);
                    let ms = march_sample(is_world, family_override, origin + dir * mid, kind);
                    if ms.d > 0.0 {
                        lo = mid;
                        trap = ms.trap;
                        hit_object = ms.object;
                    } else {
                        hi = mid;
                    }
                }
                let hit_t = lo;
                let hit_point = origin + dir * hit_t;
                let hit_eps = footprint(base, slope, hit_t, hit_point);
                return MarchResult(true, false, hit_point, trap, hit_eps, 0.0, false, hit_object);
            }
        }
        let hit = select(d < eps, d <= eps, kind == RAY_PRIMARY);
        if hit {
            return MarchResult(true, false, point, trap, eps, 0.0, is_first, hit_object);
        }
        let ratio = d / eps;
        if !has_best || ratio < best.ratio {
            best = MarchResult(false, true, point, trap, eps, ratio, false, hit_object);
            has_best = true;
        }
        prev_t = t;
        has_prev = true;
        var march_d = d;
        if frac_bulb && d < 4.0 * eps {
            march_d = min(d, fractional_march_distance(is_world, family_override, point, d, hit_object));
        }
        let unrelaxed = min(max(step_factor * march_d, 0.5 * eps), step_cap);
        if d > 4.0 * eps && relax > 1.0 {
            prev_rad = unrelaxed;
            last_step = min(unrelaxed * relax, step_cap);
        } else {
            prev_rad = 0.0;
            last_step = unrelaxed;
        }
        t = t + last_step;
    }
    if t > t_exit {
        return march_miss(origin);
    }
    if has_best {
        return best;
    }
    return MarchResult(false, true, origin, TRAP_START, 0.0, 0.0, false, 0u);
}

fn march_leaving(
    is_world: bool,
    family_override: u32,
    origin: vec3<f32>,
    retreat: vec3<f32>,
    dir: vec3<f32>,
    kind: u32,
    base: f32,
    slope: f32,
    start: f32,
) -> MarchResult {
    let m = march(is_world, family_override, origin, dir, kind, base, slope, start);
    let r = retreat - origin;
    if m.first && dot(r, r) > 0.0 {
        return march(is_world, family_override, retreat, dir, kind, base, slope, 0.0);
    }
    return m;
}

fn object_bound_radius(is_world: bool, obj: u32) -> f32 {
    if is_world {
        let base = obj * OBJECT_STRIDE;
        return max(objects[base + O_CLIP_RADIUS] * objects[base + O_DISTANCE_SCALE], 1.0e-4);
    }
    return max(param(P_CLIP_RADIUS), 1.0e-4);
}

fn march_exit(
    is_world: bool,
    obj: u32,
    family_override: u32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    base: f32,
    slope: f32,
) -> MarchResult {
    let r = object_bound_radius(is_world, obj);
    let probes = u32(clamp(round(pr(is_world, obj, P_GLASS_PROBES)), 4.0, 64.0));
    let probe = (2.0 * r) / f32(probes);
    var t = 0.0;
    var prev_t = 0.0;
    var has_prev = false;
    var last = MarchResult(false, true, origin, TRAP_START, base, 0.0, false, obj);
    for (var i = 0u; i < probes; i = i + 1u) {
        let point = origin + dir * t;
        let eps = footprint(base, slope, t, point);
        let color_mode = u32(pr(is_world, obj, P_COLOR_MODE));
        let dt = clipped_distance_and_trap(is_world, obj, family_override, point, color_mode, -1.0);
        let d = dt.x;
        var m_trap = dt.y;
        last = MarchResult(false, true, point, m_trap, eps, 0.0, false, obj);
        if d > 0.0 {
            var hit_t = t;
            if has_prev {
                var lo = prev_t;
                var hi = t;
                for (var b = 0u; b < HIT_REFINEMENTS; b = b + 1u) {
                    if hi - lo <= 0.5 * eps {
                        break;
                    }
                    let mid = 0.5 * (lo + hi);
                    let ms = clipped_distance_and_trap(
                        is_world,
                        obj,
                        family_override,
                        origin + dir * mid,
                        color_mode,
                        -1.0,
                    );
                    if ms.x <= 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                        m_trap = ms.y;
                    }
                }
                hit_t = lo;
            }
            let hit_point = origin + dir * hit_t;
            let hit_eps = footprint(base, slope, hit_t, hit_point);
            return MarchResult(true, false, hit_point, m_trap, hit_eps, 0.0, false, obj);
        }
        prev_t = t;
        has_prev = true;
        t = t + max(-d, probe);
    }
    return last;
}

fn endpoint_normal(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    dir: vec3<f32>,
    eps: f32,
    n_iter: f32,
) -> vec3<f32> {
    let family = resolved_family(is_world, obj, family_override);
    var p = x;
    if unsigned_family(family) {
        p = x - dir * eps;
    }
    let k0 = vec3<f32>(1.0, -1.0, -1.0);
    let k1 = vec3<f32>(-1.0, -1.0, 1.0);
    let k2 = vec3<f32>(-1.0, 1.0, -1.0);
    let k3 = vec3<f32>(1.0, 1.0, 1.0);
    var h = 0.25 * eps;
    for (var tries = 0u; tries < 4u; tries = tries + 1u) {
        let s = h * 0.57735027;
        let d0 = clipped_distance_and_trap(is_world, obj, family_override, p + k0 * s, 0u, n_iter).x;
        let d1 = clipped_distance_and_trap(is_world, obj, family_override, p + k1 * s, 0u, n_iter).x;
        let d2 = clipped_distance_and_trap(is_world, obj, family_override, p + k2 * s, 0u, n_iter).x;
        let d3 = clipped_distance_and_trap(is_world, obj, family_override, p + k3 * s, 0u, n_iter).x;
        let g = k0 * d0 + k1 * d1 + k2 * d2 + k3 * d3;
        let m = max(abs(g.x), max(abs(g.y), abs(g.z)));
        if m > 0.0 {
            let s_vec = g * (1.0 / m);
            return s_vec * inverseSqrt(dot(s_vec, s_vec));
        }
        h = 0.5 * h;
    }
    return -dir;
}

fn fractional_normal(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    dir: vec3<f32>,
    eps: f32,
) -> vec3<f32> {
    let n = max(pr(is_world, obj, P_ITERATIONS), 0.0);
    let whole = floor(n);
    let f = n - whole;
    let n0 = endpoint_normal(is_world, obj, family_override, x, dir, eps, whole);
    if f <= 0.0 {
        return n0;
    }
    let n1 = endpoint_normal(is_world, obj, family_override, x, dir, eps, whole + 1.0);
    let c = clamp(dot(n0, n1), -1.0, 1.0);
    if c > 0.9995 {
        return unit_or(n0 + (n1 - n0) * f, n0);
    }
    if c < -0.9995 {
        return n0;
    }
    let omega = acos(c);
    let inv_sin = 1.0 / sin(omega);
    let w0 = sin((1.0 - f) * omega) * inv_sin;
    let w1 = sin(f * omega) * inv_sin;
    return unit_or(n0 * w0 + n1 * w1, n0);
}

fn palette_base(is_world: bool, obj: u32) -> u32 {
    if is_world {
        return (obj + 1u) * (PALETTE_SAMPLES + 1u);
    }
    return 0u;
}

fn palette_color(is_world: bool, obj: u32, t: f32) -> vec3<f32> {
    let base = palette_base(is_world, obj);
    let x = clamp(t, 0.0, 1.0) * f32(PALETTE_SAMPLES - 1u);
    let i = min(u32(floor(x)), PALETTE_SAMPLES - 1u);
    let j = min(i + 1u, PALETTE_SAMPLES - 1u);
    let f = x - f32(i);
    return mix(lut[base + i].xyz, lut[base + j].xyz, f);
}

fn hit_palette(is_world: bool, obj: u32, point: vec3<f32>, n: vec3<f32>, trap: f32) -> vec3<f32> {
    if u32(pr(is_world, obj, P_COLOR_MODE)) == 0u {
        let p = object_point(is_world, obj, point);
        let t = clamp(0.4 + 0.1 * length(p) + 0.18 * n.y, 0.0, 1.0);
        return palette_color(is_world, obj, t);
    }
    let t = 1.0 - exp(-max(trap, 0.0) * pr(is_world, obj, P_TRAP_SCALE));
    return palette_color(is_world, obj, t);
}

fn env_on() -> bool {
    return u32(param(P_ENV_WIDTH)) > 0u && u32(param(P_ENV_HEIGHT)) > 0u && param(P_ENV_INTENSITY) > 0.0;
}

fn env_base() -> u32 {
    let count = select(0u, u32(param(P_OBJECT_COUNT)), param(P_WORLD) != 0.0);
    return (1u + count) * (PALETTE_SAMPLES + 1u);
}

fn env_to_map(d: vec3<f32>) -> vec3<f32> {
    let c = param(P_ENV_COS);
    let s = param(P_ENV_SIN);
    return vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}

fn env_to_world(e: vec3<f32>) -> vec3<f32> {
    let c = param(P_ENV_COS);
    let s = param(P_ENV_SIN);
    return vec3<f32>(c * e.x + s * e.z, e.y, -s * e.x + c * e.z);
}

fn equirect_uv(e: vec3<f32>) -> vec2<f32> {
    let theta = atan2(e.z, e.x);
    let phi = asin(clamp(e.y, -1.0, 1.0));
    var u = theta * SS_INV_TWO_PI + 0.5;
    u = u - floor(u);
    let v = 0.5 - phi * MX_PI_INV;
    return vec2<f32>(u, v);
}

fn equirect_dir(u: f32, v: f32) -> vec3<f32> {
    let theta = (u - 0.5) * 2.0 * MX_PI;
    let phi = (0.5 - v) * MX_PI;
    let cos_phi = cos(phi);
    return vec3<f32>(cos_phi * cos(theta), sin(phi), cos_phi * sin(theta));
}

fn env_level(level: u32) -> vec3<u32> {
    var w = u32(param(P_ENV_WIDTH));
    var h = u32(param(P_ENV_HEIGHT));
    if level == 0u {
        return vec3<u32>(env_base(), w, h);
    }
    var at = env_base() + w * h + h + 9u;
    w = (w + 1u) / 2u;
    h = (h + 1u) / 2u;
    for (var l = 1u; l < level; l = l + 1u) {
        at = at + w * h;
        w = (w + 1u) / 2u;
        h = (h + 1u) / 2u;
    }
    return vec3<u32>(at, w, h);
}

fn env_texel(at: u32, w: i32, h: i32, x: i32, y: i32) -> vec3<f32> {
    let col = ((x % w) + w) % w;
    let row = clamp(y, 0, h - 1);
    return lut[at + u32(row * w + col)].xyz;
}

fn env_bilinear(level: u32, uv: vec2<f32>) -> vec3<f32> {
    let lv = env_level(level);
    let at = lv.x;
    let w = i32(lv.y);
    let h = i32(lv.z);
    let fx = uv.x * f32(w) - 0.5;
    let fy = uv.y * f32(h) - 0.5;
    let x0 = floor(fx);
    let y0 = floor(fy);
    let tx = fx - x0;
    let ty = fy - y0;
    let ix = i32(x0);
    let iy = i32(y0);
    let top = mix(env_texel(at, w, h, ix, iy), env_texel(at, w, h, ix + 1, iy), tx);
    let bot = mix(env_texel(at, w, h, ix, iy + 1), env_texel(at, w, h, ix + 1, iy + 1), tx);
    return mix(top, bot, ty);
}

fn env_map_radiance(d: vec3<f32>) -> vec3<f32> {
    return env_bilinear(0u, equirect_uv(env_to_map(d))) * param(P_ENV_INTENSITY);
}

fn sky_gradient(dir: vec3<f32>) -> vec3<f32> {
    let k = clamp(0.5 * (dir.y + 1.0), 0.0, 1.0);
    return mix(param3(P_SKY_HORIZON), param3(P_SKY_ZENITH), k) * param(P_SKY_INTENSITY);
}

fn sky_radiance(is_world: bool, dir: vec3<f32>) -> vec3<f32> {
    var sky = sky_gradient(dir);
    if param(P_CAMERA_SUN) != 0.0 {
        if is_world {
            let n = u32(param(P_LIGHT_COUNT));
            for (var i = 0u; i < n; i = i + 1u) {
                if world_sun_contains(i, dir) {
                    let c = lp3(true, i, P_LIGHT_COLOR) * (lp(true, i, P_LIGHT_INTENSITY) * lp(true, i, P_SUN_CONE_PDF));
                    sky = sky + c;
                }
            }
        } else if sun_contains(dir) {
            let c = param3(P_LIGHT_COLOR) * (param(P_LIGHT_INTENSITY) * param(P_SUN_CONE_PDF));
            sky = sky + c;
        }
    }
    return sky;
}

fn background(is_world: bool, dir: vec3<f32>) -> vec3<f32> {
    if param(P_BACKGROUND) == 0.0 {
        return vec3<f32>(0.0);
    }
    if env_on() {
        return env_map_radiance(dir);
    }
    return sky_radiance(is_world, dir);
}

fn env_irradiance(n: vec3<f32>) -> vec3<f32> {
    let d = env_to_map(n);
    let basis = array<f32, 9>(
        0.2820948,
        0.4886025 * d.y,
        0.4886025 * d.z,
        0.4886025 * d.x,
        1.0925484 * d.x * d.y,
        1.0925484 * d.y * d.z,
        0.31539157 * (3.0 * d.z * d.z - 1.0),
        1.0925484 * d.x * d.z,
        0.5462742 * (d.x * d.x - d.y * d.y),
    );
    let w = u32(param(P_ENV_WIDTH));
    let h = u32(param(P_ENV_HEIGHT));
    let at = env_base() + w * h + h;
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < 9u; i = i + 1u) {
        e = e + lut[at + i].xyz * basis[i];
    }
    return max(e, vec3<f32>(0.0)) * param(P_ENV_INTENSITY);
}

fn env_specular(d: vec3<f32>, alpha: f32) -> vec3<f32> {
    let h = param(P_ENV_HEIGHT);
    let levels = max(u32(param(P_ENV_LEVELS)), 1u);
    let texels = max(alpha * h * MX_PI_INV, 1.0);
    let lod = min(log2(texels), f32(levels - 1u));
    let uv = equirect_uv(env_to_map(d));
    let base = floor(lod);
    let t = lod - base;
    let lo = u32(base);
    let lower = env_bilinear(lo, uv);
    let intensity = param(P_ENV_INTENSITY);
    if t == 0.0 {
        return lower * intensity;
    }
    let upper = env_bilinear(lo + 1u, uv);
    return (lower + (upper - lower) * t) * intensity;
}

fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {
    if env_on() {
        return env_irradiance(n);
    }
    let k = clamp(0.5 + n.y / 3.0, 0.0, 1.0);
    return mix(param3(P_SKY_HORIZON), param3(P_SKY_ZENITH), k) * param(P_SKY_INTENSITY);
}

fn sky_specular(is_world: bool, d: vec3<f32>, alpha: f32) -> vec3<f32> {
    if env_on() {
        return env_specular(d, alpha);
    }
    return sky_radiance(is_world, d);
}

fn env_step_offset(xi: f32, lo: f32, hi: f32) -> f32 {
    if hi > lo {
        return clamp((xi - lo) / (hi - lo), 0.0, 1.0);
    }
    return 0.5;
}

fn env_texel_pdf(x: u32, y: u32, v: f32) -> f32 {
    let w = u32(param(P_ENV_WIDTH));
    let h = u32(param(P_ENV_HEIGHT));
    let base = env_base();
    let row_start = base + w * h;
    let row_hi = lut[row_start + y].x;
    let row_lo = select(lut[row_start + y - 1u].x, 0.0, y == 0u);
    let marginal_pdf = row_hi - row_lo;
    let col_start = base + y * w;
    let col_hi = lut[col_start + x].w;
    let col_lo = select(lut[col_start + x - 1u].w, 0.0, x == 0u);
    let conditional_pdf = col_hi - col_lo;
    let sin_theta = sin(MX_PI * v);
    if sin_theta <= 0.0 {
        return 0.0;
    }
    return max(marginal_pdf * conditional_pdf, 0.0) * f32(w * h) / (2.0 * MX_PI * MX_PI * sin_theta);
}

fn env_map_sample(r1: f32, r2: f32) -> vec4<f32> {
    let w = u32(param(P_ENV_WIDTH));
    let h = u32(param(P_ENV_HEIGHT));
    let base = env_base();
    let row_start = base + w * h;
    var lo_y = 0u;
    var hi_y = h;
    while lo_y < hi_y {
        let mid = (lo_y + hi_y) / 2u;
        if lut[row_start + mid].x < r2 {
            lo_y = mid + 1u;
        } else {
            hi_y = mid;
        }
    }
    let y = min(lo_y, h - 1u);
    let col_start = base + y * w;
    var lo_x = 0u;
    var hi_x = w;
    while lo_x < hi_x {
        let mid = (lo_x + hi_x) / 2u;
        if lut[col_start + mid].w < r1 {
            lo_x = mid + 1u;
        } else {
            hi_x = mid;
        }
    }
    let x = min(lo_x, w - 1u);
    let row_lo = select(lut[row_start + y - 1u].x, 0.0, y == 0u);
    let row_hi = lut[row_start + y].x;
    let col_lo = select(lut[col_start + x - 1u].w, 0.0, x == 0u);
    let col_hi = lut[col_start + x].w;
    let dv = env_step_offset(r2, row_lo, row_hi);
    let du = env_step_offset(r1, col_lo, col_hi);
    let u = (f32(x) + du) / f32(w);
    let v = (f32(y) + dv) / f32(h);
    let dir = env_to_world(equirect_dir(u, v));
    return vec4<f32>(dir, env_texel_pdf(x, y, v));
}

fn env_map_pdf(dir: vec3<f32>) -> f32 {
    let w = u32(param(P_ENV_WIDTH));
    let h = u32(param(P_ENV_HEIGHT));
    let uv = equirect_uv(env_to_map(dir));
    let x = min(u32(uv.x * f32(w)), w - 1u);
    let y = min(u32(uv.y * f32(h)), h - 1u);
    return env_texel_pdf(x, y, uv.y);
}

fn sun_select() -> f32 {
    if env_on() {
        return param(P_ENV_SUN_SELECT);
    }
    let sun = luminance(param3(P_LIGHT_COLOR)) * param(P_LIGHT_INTENSITY);
    let sky = luminance(param3(P_SKY_HORIZON) + param3(P_SKY_ZENITH)) * (0.5 * param(P_SKY_INTENSITY));
    let total = sun + MX_PI * sky;
    var p = 0.5;
    if total > 0.0 {
        p = sun / total;
    }
    return clamp(p, 0.1, 0.9);
}

fn sun_contains(wi: vec3<f32>) -> bool {
    if param(P_SUN_CONE_PDF) <= 0.0 {
        return false;
    }
    let d = wi - param3(P_LIGHT_DIR);
    return 0.5 * dot(d, d) <= param(P_SUN_ONE_MINUS_COS);
}

fn world_sun_contains(light: u32, wi: vec3<f32>) -> bool {
    if lp(true, light, P_SUN_CONE_PDF) <= 0.0 {
        return false;
    }
    let d = wi - lp3(true, light, P_LIGHT_DIR);
    return 0.5 * dot(d, d) <= lp(true, light, P_SUN_ONE_MINUS_COS);
}

fn world_sun_weight(light: u32) -> f32 {
    if lp(true, light, P_SUN_CONE_PDF) <= 0.0 {
        return 0.0;
    }
    return max(luminance(lp3(true, light, P_LIGHT_COLOR)) * lp(true, light, P_LIGHT_INTENSITY), 0.0);
}

fn world_sky_weight() -> f32 {
    if env_on() {
        return max(4.0 * MX_PI * param(P_ENV_MEAN) * param(P_ENV_INTENSITY), 0.0);
    }
    let sky = luminance(param3(P_SKY_HORIZON) + param3(P_SKY_ZENITH)) * (0.5 * param(P_SKY_INTENSITY));
    return max(MX_PI * sky, 0.0);
}

fn world_light_total() -> f32 {
    let n = u32(param(P_LIGHT_COUNT));
    var total = world_sky_weight();
    for (var i = 0u; i < n; i = i + 1u) {
        total = total + world_sun_weight(i);
    }
    return total;
}

fn env_radiance(is_world: bool, dir: vec3<f32>) -> vec3<f32> {
    var rad: vec3<f32>;
    if env_on() {
        rad = env_map_radiance(dir);
    } else {
        rad = sky_gradient(dir);
    }
    if is_world {
        let n = u32(param(P_LIGHT_COUNT));
        for (var i = 0u; i < n; i = i + 1u) {
            if world_sun_contains(i, dir) {
                let c = lp3(true, i, P_LIGHT_COLOR) * (lp(true, i, P_LIGHT_INTENSITY) * lp(true, i, P_SUN_CONE_PDF));
                rad = rad + c;
            }
        }
    } else if sun_contains(dir) {
        let c = param3(P_LIGHT_COLOR) * (param(P_LIGHT_INTENSITY) * param(P_SUN_CONE_PDF));
        rad = rad + c;
    }
    return rad;
}

fn world_env_pdf(dir: vec3<f32>) -> f32 {
    let total = world_light_total();
    if total <= 0.0 {
        return 0.0;
    }
    let sky_pdf = select(1.0 / (4.0 * MX_PI), env_map_pdf(dir), env_on());
    var pdf = (world_sky_weight() / total) * sky_pdf;
    let n = u32(param(P_LIGHT_COUNT));
    for (var i = 0u; i < n; i = i + 1u) {
        if world_sun_contains(i, dir) {
            pdf = pdf + (world_sun_weight(i) / total) * lp(true, i, P_SUN_CONE_PDF);
        }
    }
    return pdf;
}

fn single_env_pdf(dir: vec3<f32>) -> f32 {
    let sun_pdf = param(P_SUN_CONE_PDF);
    var sky = 1.0 / (4.0 * MX_PI);
    if env_on() {
        sky = env_map_pdf(dir);
    } else if sun_pdf <= 0.0 {
        return 0.0;
    }
    let p_sun = select(0.0, sun_select(), sun_pdf > 0.0);
    var pdf = (1.0 - p_sun) * sky;
    if sun_contains(dir) {
        pdf = pdf + p_sun * sun_pdf;
    }
    return pdf;
}

fn env_pdf(is_world: bool, dir: vec3<f32>) -> f32 {
    if is_world {
        return world_env_pdf(dir);
    }
    return single_env_pdf(dir);
}

fn sample_sun_cone(l_dir: vec3<f32>, omc: f32, u1: f32, u2: f32) -> vec3<f32> {
    let basis = mx_orthonormal_basis(l_dir);
    return ss_to_world(ss_sample_cone(vec2<f32>(u1, u2), omc), basis[0], basis[1], l_dir);
}

fn sample_uniform_sphere(u: f32, v: f32) -> vec3<f32> {
    let z = 1.0 - 2.0 * u;
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = 2.0 * MX_PI * v;
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

fn world_env_sample(r1: f32, r2: f32) -> vec4<f32> {
    let total = world_light_total();
    if total <= 0.0 {
        return vec4<f32>(0.0, 1.0, 0.0, 0.0);
    }
    let goal = r1 * total;
    var acc = 0.0;
    let n = u32(param(P_LIGHT_COUNT));
    var dir = vec3<f32>(0.0);
    var picked = false;
    for (var i = 0u; i < n; i = i + 1u) {
        let w = world_sun_weight(i);
        if w > 0.0 {
            let next = acc + w;
            if goal < next {
                let u = clamp((goal - acc) / w, 0.0, 1.0 - 1.0e-7);
                dir = sample_sun_cone(lp3(true, i, P_LIGHT_DIR), lp(true, i, P_SUN_ONE_MINUS_COS), u, r2);
                picked = true;
                break;
            }
            acc = next;
        }
    }
    if !picked {
        let w = max(total - acc, 1.0e-8);
        let u = clamp((goal - acc) / w, 0.0, 1.0 - 1.0e-7);
        if env_on() {
            dir = env_map_sample(u, r2).xyz;
        } else {
            dir = sample_uniform_sphere(u, r2);
        }
    }
    return vec4<f32>(dir, world_env_pdf(dir));
}

fn single_env_sample(r1: f32, r2: f32) -> vec4<f32> {
    let sun_pdf = param(P_SUN_CONE_PDF);
    let p_sun = select(0.0, sun_select(), sun_pdf > 0.0);
    var dir: vec3<f32>;
    if p_sun > 0.0 && r1 < p_sun {
        dir = sample_sun_cone(param3(P_LIGHT_DIR), param(P_SUN_ONE_MINUS_COS), r1 / p_sun, r2);
    } else {
        let u = (r1 - p_sun) / max(1.0 - p_sun, 1.0e-8);
        if env_on() {
            dir = env_map_sample(u, r2).xyz;
        } else if sun_pdf > 0.0 {
            dir = sample_uniform_sphere(u, r2);
        } else {
            return vec4<f32>(0.0, 1.0, 0.0, 0.0);
        }
    }
    return vec4<f32>(dir, single_env_pdf(dir));
}

fn env_sample(is_world: bool, r1: f32, r2: f32) -> vec4<f32> {
    if is_world {
        return world_env_sample(r1, r2);
    }
    return single_env_sample(r1, r2);
}

struct FastMaterial {
    albedo: vec3<f32>,
    f0: vec3<f32>,
    alpha: f32,
    diffuse_w: f32,
    spec_prob: f32,
    emission: vec3<f32>,
}

fn basis(n: vec3<f32>) -> array<vec3<f32>, 2> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return array<vec3<f32>, 2>(
        vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3<f32>(b, s + n.y * n.y * a, -n.y),
    );
}

fn fast_spec_probability(luminance_f0: f32, diffuse_energy: f32, diffuse_weight: f32) -> f32 {
    if diffuse_weight <= 0.0 {
        return 1.0;
    }
    return clamp(luminance_f0 / (luminance_f0 + diffuse_energy + 1.0e-4), 0.15, 0.9);
}

fn fast_ggx_sample_local(wo_local: vec3<f32>, alpha: f32, xi: vec2<f32>) -> vec3<f32> {
    let h = mx_ggx_importance_sample_vndf(xi, wo_local, vec2<f32>(alpha, alpha));
    let vh = dot(wo_local, h);
    return h * (2.0 * vh) - wo_local;
}

fn fast_ggx_pdf(ndf: f32, g1_view: f32, n_dot_view: f32) -> f32 {
    if n_dot_view > 0.0 {
        return ndf * g1_view / (4.0 * n_dot_view);
    }
    return 0.0;
}

fn fast_schlick(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let m = clamp(1.0 - cos_theta, 0.0, 1.0);
    let m2 = m * m;
    let m5 = m2 * m2 * m;
    return f0 + (vec3<f32>(1.0) - f0) * m5;
}

fn fast_smith_g1(a2: f32, nv: f32) -> f32 {
    return 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
}

fn fast_material(
    is_world: bool,
    obj: u32,
    palette_rgb: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
) -> FastMaterial {
    var color = palette_rgb;
    if u32(pr(is_world, obj, P_COLOR_SOURCE)) != 0u {
        color = pr3(is_world, obj, P_BASE_COLOR);
    }
    var roughness = pr(is_world, obj, P_SPECULAR_ROUGHNESS);
    var metalness = pr(is_world, obj, P_METALNESS);
    let facing_exp = pr(is_world, obj, P_FACING_EXPONENT);
    if facing_exp > 0.0 {
        let weight = pow(max(1.0 - abs(dot(normal, view)), 0.0), facing_exp);
        let fc = pr3(is_world, obj, P_FACING_COLOR);
        let fr = pr(is_world, obj, P_FACING_ROUGHNESS);
        let fm = pr(is_world, obj, P_FACING_METALLIC);
        color = color + (fc - color) * weight;
        roughness = roughness + (fr - roughness) * weight;
        metalness = metalness + (fm - metalness) * weight;
    }
    let base = (color * pr(is_world, obj, P_BASE)) * pr3(is_world, obj, P_BASE_TINT);
    let ior = pr(is_world, obj, P_SPECULAR_IOR);
    let r0 = (ior - 1.0) / (ior + 1.0);
    let f0d = r0 * r0 * pr(is_world, obj, P_SPECULAR);
    let f0 = (pr3(is_world, obj, P_SPECULAR_COLOR) * f0d) * (1.0 - metalness) + base * metalness;
    let rough = max(roughness, 0.03);
    let diffuse_w = 1.0 - metalness;
    let lf = luminance(f0);
    let spec_prob = fast_spec_probability(lf, diffuse_w * luminance(base), diffuse_w);
    let emission = pr3(is_world, obj, P_EMISSION_COLOR) * max(pr(is_world, obj, P_EMISSION), 0.0);
    return FastMaterial(base, f0, rough * rough, diffuse_w, spec_prob, emission);
}

fn fast_eval(mat: FastMaterial, n: vec3<f32>, wo: vec3<f32>, wi: vec3<f32>) -> vec4<f32> {
    let nl = dot(n, wi);
    let nv = dot(n, wo);
    if nl <= 0.0 || nv <= 0.0 {
        return vec4<f32>(0.0);
    }
    let h = normalize(wo + wi);
    let nh = max(dot(n, h), 0.0);
    let vh = max(dot(wo, h), 1.0e-6);
    let a2 = mat.alpha * mat.alpha;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (MX_PI * dd * dd);
    let g1v = fast_smith_g1(a2, nv);
    let g = g1v * fast_smith_g1(a2, nl);
    let f = fast_schlick(mat.f0, vh);
    let spec = f * (d * g / (4.0 * nv));
    let diff = ((vec3<f32>(1.0) - f) * mat.diffuse_w * mat.albedo) * (nl * MX_PI_INV);
    let pdf = mat.spec_prob * fast_ggx_pdf(d, g1v, nv) + (1.0 - mat.spec_prob) * (nl * MX_PI_INV);
    return vec4<f32>(spec + diff, pdf);
}

struct FastSample {
    wi: vec3<f32>,
    throughput: vec3<f32>,
    pdf: f32,
    valid: bool,
}

fn fast_sample(mat: FastMaterial, n: vec3<f32>, wo: vec3<f32>, u: vec3<f32>) -> FastSample {
    let tb = basis(n);
    let t = tb[0];
    let b = tb[1];
    var wi: vec3<f32>;
    if u.x < mat.spec_prob {
        let local = fast_ggx_sample_local(vec3<f32>(dot(wo, t), dot(wo, b), dot(wo, n)), mat.alpha, u.yz);
        wi = t * local.x + b * local.y + n * local.z;
    } else {
        let r = sqrt(u.y);
        let phi = 2.0 * MX_PI * u.z;
        wi = t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0 - u.y, 0.0));
    }
    let ev = fast_eval(mat, n, wo, wi);
    if ev.w <= 0.0 {
        return FastSample(wi, vec3<f32>(0.0), 0.0, false);
    }
    return FastSample(wi, ev.xyz / ev.w, ev.w, true);
}

fn surface_inputs(
    is_world: bool,
    obj: u32,
    palette_rgb: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
) -> SsInputs {
    var color = palette_rgb;
    if u32(pr(is_world, obj, P_COLOR_SOURCE)) != 0u {
        color = pr3(is_world, obj, P_BASE_COLOR);
    }
    var roughness = pr(is_world, obj, P_SPECULAR_ROUGHNESS);
    var metalness = pr(is_world, obj, P_METALNESS);
    let facing_exp = pr(is_world, obj, P_FACING_EXPONENT);
    if facing_exp > 0.0 {
        let weight = pow(max(1.0 - abs(dot(normal, view)), 0.0), facing_exp);
        let fc = pr3(is_world, obj, P_FACING_COLOR);
        let fr = pr(is_world, obj, P_FACING_ROUGHNESS);
        let fm = pr(is_world, obj, P_FACING_METALLIC);
        color = color + (fc - color) * weight;
        roughness = roughness + (fr - roughness) * weight;
        metalness = metalness + (fm - metalness) * weight;
    }
    let trans_col = select(
        pr3(is_world, obj, P_TRANSMISSION_COLOR),
        vec3<f32>(1.0),
        pr(is_world, obj, P_TRANSMISSION_DEPTH) > 0.0,
    );
    return SsInputs(
        pr(is_world, obj, P_BASE),
        color * pr3(is_world, obj, P_BASE_TINT),
        pr(is_world, obj, P_DIFFUSE_ROUGHNESS),
        metalness,
        pr(is_world, obj, P_SPECULAR),
        pr3(is_world, obj, P_SPECULAR_COLOR),
        roughness,
        pr(is_world, obj, P_SPECULAR_IOR),
        pr(is_world, obj, P_SPECULAR_ANISOTROPY),
        pr(is_world, obj, P_SPECULAR_ROTATION),
        pr(is_world, obj, P_TRANSMISSION),
        trans_col,
        pr(is_world, obj, P_TRANSMISSION_EXTRA_ROUGHNESS),
        0.0,
        vec3<f32>(1.0),
        vec3<f32>(1.0),
        1.0,
        0.0,
        pr(is_world, obj, P_SHEEN),
        pr3(is_world, obj, P_SHEEN_COLOR),
        pr(is_world, obj, P_SHEEN_ROUGHNESS),
        pr(is_world, obj, P_COAT),
        pr3(is_world, obj, P_COAT_COLOR),
        pr(is_world, obj, P_COAT_ROUGHNESS),
        0.0,
        0.0,
        pr(is_world, obj, P_COAT_IOR),
        pr(is_world, obj, P_COAT_AFFECT_COLOR),
        pr(is_world, obj, P_COAT_AFFECT_ROUGHNESS),
        pr(is_world, obj, P_THIN_FILM_THICKNESS),
        pr(is_world, obj, P_THIN_FILM_IOR),
        pr(is_world, obj, P_EMISSION),
        pr3(is_world, obj, P_EMISSION_COLOR),
        thin_film_energy(),
    );
}

fn object_tangent_y(is_world: bool, obj: u32) -> vec3<f32> {
    if is_world {
        let base = obj * OBJECT_STRIDE + O_TANGENT;
        return vec3<f32>(objects[base], objects[base + 1u], objects[base + 2u]);
    }
    return vec3<f32>(
        param(P_OBJ_AXES + 1u),
        param(P_OBJ_AXES + 4u),
        param(P_OBJ_AXES + 7u),
    );
}

struct Rng {
    px: u32,
    py: u32,
    sample: u32,
    dim: u32,
}

fn rand(r: ptr<function, Rng>) -> f32 {
    let u = ss_sobol((*r).px, (*r).py, (*r).sample, (*r).dim);
    (*r).dim = (*r).dim + 1u;
    return u;
}

fn pixel_rng(px: i32, py: i32, sample: u32) -> Rng {
    let seed = select(
        u32(param(P_SEED)),
        (u32(param(P_OFX_SEED_HIGH)) << 16u) | u32(param(P_SEED)),
        is_ofx(),
    );
    return Rng(bitcast<u32>(px) ^ seed, bitcast<u32>(py), sample, 0u);
}

fn concentric_disk(u1: f32, u2: f32) -> vec2<f32> {
    let ox = 2.0 * u1 - 1.0;
    let oy = 2.0 * u2 - 1.0;
    if ox == 0.0 && oy == 0.0 {
        return vec2<f32>(0.0, 0.0);
    }
    var r: f32;
    var theta: f32;
    if abs(ox) > abs(oy) {
        r = ox;
        theta = (0.25 * MX_PI) * (oy / ox);
    } else {
        r = oy;
        theta = (0.5 * MX_PI) - (0.25 * MX_PI) * (ox / oy);
    }
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

fn camera_ray(px: i32, py: i32, jx: f32, jy: f32, lx: f32, ly: f32) -> Ray {
    let fo = frame_origin();
    let fx = f32(px - fo.x) + jx;
    let fy = f32(py - fo.y) + jy;
    let nx = 2.0 * fx / param(P_WIDTH) - 1.0;
    let ny = select(1.0 - 2.0 * fy / param(P_HEIGHT), 2.0 * fy / param(P_HEIGHT) - 1.0, is_ofx());
    let origin = param3(P_CAM_ORIGIN);
    let right = param3(P_CAM_RIGHT);
    let up = param3(P_CAM_UP);
    let fwd = param3(P_CAM_FORWARD);
    let hw = param(P_HALF_W);
    let hh = param(P_HALF_H);
    let dir = normalize(fwd + right * (nx * hw) + up * (ny * hh));
    let aperture = select(param(P_APERTURE), 0.0, is_ofx());
    if aperture <= 0.0 {
        return Ray(origin, dir, origin);
    }
    let focus_t = param(P_FOCUS_DISTANCE) / max(dot(dir, fwd), 1.0e-4);
    let focus = origin + dir * focus_t;
    let lens = concentric_disk(lx, ly);
    let lens_origin = origin + right * (lens.x * aperture) + up * (lens.y * aperture);
    return Ray(lens_origin, normalize(focus - lens_origin), lens_origin);
}

struct TraceOut {
    radiance: vec3<f32>,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    unresolved: bool,
    alpha: f32,
}

const AIR_MEDIUM: u32 = 0xffffffffu;

fn volume_transmittance(color: vec3<f32>, depth: f32, distance: f32) -> vec3<f32> {
    if depth <= 0.0 || distance <= 0.0 {
        return vec3<f32>(1.0);
    }
    let c = clamp(color, vec3<f32>(1.0e-4), vec3<f32>(1.0));
    let factor = -distance / depth;
    return exp(log(c) * factor);
}

fn trace_path(
    is_world: bool,
    family_override: u32,
    mat_spec: u32,
    px: i32,
    py: i32,
    sample: u32,
    start: f32,
) -> TraceOut {
    var rng = pixel_rng(px, py, sample);
    let jx = rand(&rng);
    let jy = rand(&rng);
    let lx = rand(&rng);
    let ly = rand(&rng);
    let cam = camera_ray(px, py, jx, jy, lx, ly);
    var ro = cam.origin;
    var retreat = cam.retreat;
    var rd = cam.dir;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    var guide_albedo = vec3<f32>(0.0);
    var guide_normal = vec3<f32>(0.0);
    var unresolved = false;
    var alpha = 1.0;

    let max_bounces = u32(param(P_MAX_BOUNCES));
    let slope = param(P_FOOTPRINT);
    let cam_origin = param3(P_CAM_ORIGIN);
    var medium = AIR_MEDIUM;
    var medium_eps = 0.0;
    var interface_point = ro;
    var mis_bsdf_pdf = 0.0;
    var mis_env = false;

    for (var bounce = 0u; bounce <= max_bounces + 1u; bounce = bounce + 1u) {
        let inside = medium != AIR_MEDIUM;
        var m: MarchResult;
        if inside {
            let base = max(
                slope * param(P_SECONDARY_EPS) * length(ro - cam_origin),
                medium_eps,
            );
            m = march_exit(is_world, medium, family_override, ro, rd, base, 0.0);
        } else if bounce == 0u {
            m = march(is_world, family_override, ro, rd, RAY_PRIMARY, 0.0, slope, start);
        } else {
            let base = slope * param(P_SECONDARY_EPS) * length(ro - cam_origin);
            m = march_leaving(is_world, family_override, ro, retreat, rd, RAY_BOUNCE, base, 0.0, 0.0);
        }

        if m.unresolved {
            unresolved = true;
            if bounce == 0u {
                let bg = background(is_world, rd);
                guide_albedo = clamp(bg, vec3<f32>(0.0), vec3<f32>(1.0));
                alpha = sky_alpha();
                radiance = radiance + throughput * bg;
            }
            break;
        }
        if !m.hit {
            if bounce == 0u && param(P_CAMERA_SUN) == 0.0 {
                let bg = background(is_world, rd);
                guide_albedo = clamp(bg, vec3<f32>(0.0), vec3<f32>(1.0));
                alpha = sky_alpha();
                radiance = radiance + throughput * bg;
            } else {
                let w = select(1.0, ss_power_heuristic(mis_bsdf_pdf, env_pdf(is_world, rd)), mis_env);
                let env = env_radiance(is_world, rd);
                if bounce == 0u {
                    let bg = background(is_world, rd);
                    guide_albedo = clamp(bg, vec3<f32>(0.0), vec3<f32>(1.0));
                    alpha = sky_alpha();
                    if param(P_BACKGROUND) != 0.0 {
                        radiance = radiance + throughput * env * w;
                    }
                } else {
                    radiance = radiance + throughput * env * w;
                }
            }
            break;
        }
        if inside {
            let tc = pr3(is_world, medium, P_TRANSMISSION_COLOR);
            let td = pr(is_world, medium, P_TRANSMISSION_DEPTH);
            let dist = length(m.point - interface_point);
            throughput = throughput * volume_transmittance(tc, td, dist);
        }
        if bounce > max_bounces {
            break;
        }

        var outward: vec3<f32>;
        if bounce == 0u || inside {
            outward = fractional_normal(is_world, m.object, family_override, m.point, rd, m.eps);
        } else {
            outward = endpoint_normal(is_world, m.object, family_override, m.point, rd, m.eps, -1.0);
        }
        let n = select(outward, -outward, inside);
        let wo = -rd;
        let geo_normal = select(-n, n, dot(n, wo) > 0.0);
        let pal = hit_palette(is_world, m.object, m.point, outward, m.trap);
        let shadow_base = max(
            slope * param(P_SECONDARY_EPS) * length(m.point - cam_origin),
            m.eps,
        );
        let use_fast = !inside
            && (mat_spec == MAT_FAST
                || (mat_spec == MAT_DISPATCH && u32(pr(is_world, m.object, P_MATERIAL_MODEL)) == 0u));

        if use_fast {
            let mat = fast_material(is_world, m.object, pal, geo_normal, wo);
            if bounce == 0u {
                guide_albedo = mat.albedo;
                guide_normal = geo_normal;
            }
            radiance = radiance + throughput * mat.emission;

            if bounce > 0u {
                let p_continue = min(max(throughput.x, max(throughput.y, throughput.z)), 1.0);
                if rand(&rng) >= p_continue {
                    break;
                }
                throughput = throughput / p_continue;
            }

            if is_world {
                let n_lights = u32(param(P_LIGHT_COUNT));
                for (var i = 0u; i < n_lights; i = i + 1u) {
                    if lp(true, i, P_SUN_CONE_PDF) <= 0.0 {
                        let ld = lp3(true, i, P_LIGHT_DIR);
                        let ev = fast_eval(mat, geo_normal, wo, ld);
                        if any(ev.xyz > vec3<f32>(0.0)) {
                            let sray = leave_ray(m.point, geo_normal, rd, ld, m.eps);
                            let sm = visibility_leaving(true, family_override, sray.origin, sray.retreat, sray.dir, shadow_base);
                            if sm.unresolved {
                                unresolved = true;
                            }
                            if !sm.hit && !sm.unresolved {
                                let sun = lp3(true, i, P_LIGHT_COLOR) * lp(true, i, P_LIGHT_INTENSITY);
                                radiance = radiance + throughput * ev.xyz * sun;
                            }
                        }
                    }
                }
            } else if param(P_SUN_CONE_PDF) <= 0.0 {
                let ld = param3(P_LIGHT_DIR);
                let ev = fast_eval(mat, geo_normal, wo, ld);
                if any(ev.xyz > vec3<f32>(0.0)) {
                    let sray = leave_ray(m.point, geo_normal, rd, ld, m.eps);
                    let sm = visibility_leaving(false, family_override, sray.origin, sray.retreat, sray.dir, shadow_base);
                    if sm.unresolved {
                        unresolved = true;
                    }
                    if !sm.hit && !sm.unresolved {
                        let sun = param3(P_LIGHT_COLOR) * param(P_LIGHT_INTENSITY);
                        radiance = radiance + throughput * ev.xyz * sun;
                    }
                }
            }

            let u1 = rand(&rng);
            let u2 = rand(&rng);
            let ls = env_sample(is_world, u1, u2);
            let ld = ls.xyz;
            let l_pdf = ls.w;
            if l_pdf > 0.0 {
                let ev = fast_eval(mat, geo_normal, wo, ld);
                if any(ev.xyz > vec3<f32>(0.0)) {
                    let sray = leave_ray(m.point, geo_normal, rd, ld, m.eps);
                    let sm = visibility_leaving(is_world, family_override, sray.origin, sray.retreat, sray.dir, shadow_base);
                    if sm.unresolved {
                        unresolved = true;
                    }
                    if !sm.hit && !sm.unresolved {
                        let w = ss_power_heuristic(l_pdf, ev.w);
                        let li = env_radiance(is_world, ld);
                        radiance = radiance + throughput * ev.xyz * (li * (w / l_pdf));
                    }
                }
            }

            let u = vec3<f32>(rand(&rng), rand(&rng), rand(&rng));
            let s = fast_sample(mat, geo_normal, wo, u);
            if !s.valid {
                break;
            }
            throughput = throughput * s.throughput;
            mis_bsdf_pdf = s.pdf;
            mis_env = true;
            let next_ray = leave_ray(m.point, geo_normal, rd, s.wi, m.eps);
            ro = next_ray.origin;
            retreat = next_ray.retreat;
            rd = next_ray.dir;
        } else {
            let inputs = surface_inputs(is_world, m.object, pal, geo_normal, wo);
            let frame = SsFrame(geo_normal, object_tangent_y(is_world, m.object), inside, 1.0);
            if bounce == 0u {
                guide_albedo = ss_guide_albedo(inputs);
                guide_normal = geo_normal;
            }
            if !inside {
                radiance = radiance + throughput * ss_eval_emission(inputs, frame, wo);
            }

            if bounce > 0u {
                let p_continue = min(max(throughput.x, max(throughput.y, throughput.z)), 1.0);
                if rand(&rng) >= p_continue {
                    break;
                }
                throughput = throughput / p_continue;
            }

            let layers = ss_layers(inputs, frame, wo);
            let lobes = ss_lobe_weights_prepared(inputs, layers, wo);

            if !inside {
                if is_world {
                    let n_lights = u32(param(P_LIGHT_COUNT));
                    for (var i = 0u; i < n_lights; i = i + 1u) {
                        if lp(true, i, P_SUN_CONE_PDF) <= 0.0 {
                            let ld = lp3(true, i, P_LIGHT_DIR);
                            let bsdf_lobes = ss_eval_light_prepared(inputs, frame, layers, wo, ld);
                            let bsdf = bsdf_lobes.base + bsdf_lobes.specular + bsdf_lobes.transmission;
                            if any(bsdf > vec3<f32>(0.0)) {
                                let sray = leave_ray(m.point, geo_normal, rd, ld, m.eps);
                                let sm = visibility_leaving(true, family_override, sray.origin, sray.retreat, sray.dir, shadow_base);
                                if sm.unresolved {
                                    unresolved = true;
                                }
                                if !sm.hit && !sm.unresolved {
                                    let sun = lp3(true, i, P_LIGHT_COLOR) * lp(true, i, P_LIGHT_INTENSITY);
                                    radiance = radiance + throughput * bsdf * sun;
                                }
                            }
                        }
                    }
                } else if param(P_SUN_CONE_PDF) <= 0.0 {
                    let ld = param3(P_LIGHT_DIR);
                    let bsdf_lobes = ss_eval_light_prepared(inputs, frame, layers, wo, ld);
                    let bsdf = bsdf_lobes.base + bsdf_lobes.specular + bsdf_lobes.transmission;
                    if any(bsdf > vec3<f32>(0.0)) {
                        let sray = leave_ray(m.point, geo_normal, rd, ld, m.eps);
                        let sm = visibility_leaving(false, family_override, sray.origin, sray.retreat, sray.dir, shadow_base);
                        if sm.unresolved {
                            unresolved = true;
                        }
                        if !sm.hit && !sm.unresolved {
                            let sun = param3(P_LIGHT_COLOR) * param(P_LIGHT_INTENSITY);
                            radiance = radiance + throughput * bsdf * sun;
                        }
                    }
                }

                let u1 = rand(&rng);
                let u2 = rand(&rng);
                let ls = env_sample(is_world, u1, u2);
                let ld = ls.xyz;
                let l_pdf = ls.w;
                if l_pdf > 0.0 {
                    let bsdf_lobes = ss_eval_light_prepared(inputs, frame, layers, wo, ld);
                    let bsdf = bsdf_lobes.base + bsdf_lobes.specular + bsdf_lobes.transmission;
                    if any(bsdf > vec3<f32>(0.0)) {
                        let sray = leave_ray(m.point, geo_normal, rd, ld, m.eps);
                        let sm = visibility_leaving(is_world, family_override, sray.origin, sray.retreat, sray.dir, shadow_base);
                        if sm.unresolved {
                            unresolved = true;
                        }
                        if !sm.hit && !sm.unresolved {
                            let b_pdf = ss_pdf_prepared(layers, wo, ld, lobes);
                            let w = ss_power_heuristic(l_pdf, b_pdf);
                            let li = env_radiance(is_world, ld);
                            radiance = radiance + throughput * bsdf * (li * (w / l_pdf));
                        }
                    }
                }
            }

            let u = vec3<f32>(rand(&rng), rand(&rng), rand(&rng));
            let s = ss_sample_prepared(inputs, frame, layers, wo, u, lobes);
            if !s.valid {
                break;
            }
            throughput = throughput * s.weight;
            let next_inside = select(inside, !inside, s.lobe == SS_LOBE_TRANSMISSION);
            medium = select(AIR_MEDIUM, m.object, next_inside);
            mis_bsdf_pdf = s.pdf;
            mis_env = !next_inside && !s.delta;
            if next_inside {
                medium_eps = m.eps;
                interface_point = m.point;
            }
            let next_ray = leave_ray(m.point, geo_normal, rd, s.wi, m.eps);
            ro = next_ray.origin;
            retreat = next_ray.retreat;
            rd = next_ray.dir;
        }
    }

    return TraceOut(radiance, guide_albedo, guide_normal, unresolved, alpha);
}

fn direct_camera_ray(px: i32, py: i32, jx: f32, jy: f32) -> Ray {
    let fo = frame_origin();
    let fx = f32(px - fo.x) + jx;
    let fy = f32(py - fo.y) + jy;
    let nx = 2.0 * fx / param(P_WIDTH) - 1.0;
    let ny = 1.0 - 2.0 * fy / param(P_HEIGHT);
    let origin = param3(P_CAM_ORIGIN);
    let right = param3(P_CAM_RIGHT);
    let up = param3(P_CAM_UP);
    let fwd = param3(P_CAM_FORWARD);
    let hw = param(P_HALF_W);
    let hh = param(P_HALF_H);
    let dir = normalize(fwd + right * (nx * hw) + up * (ny * hh));
    return Ray(origin, dir, origin);
}

fn direct_stencil(
    is_world: bool,
    obj: u32,
    family_override: u32,
    center: vec3<f32>,
    h_in: f32,
    fallback: vec3<f32>,
) -> vec3<f32> {
    let family = resolved_family(is_world, obj, family_override);
    let signed_fam = !unsigned_family(family);
    let k0 = vec3<f32>(1.0, -1.0, -1.0);
    let k1 = vec3<f32>(-1.0, -1.0, 1.0);
    let k2 = vec3<f32>(-1.0, 1.0, -1.0);
    let k3 = vec3<f32>(1.0, 1.0, 1.0);
    var h = h_in;
    for (var k = 0u; k <= 3u; k = k + 1u) {
        let s = h * 0.57735027;
        let d0 = clipped_distance_and_trap(is_world, obj, family_override, center + k0 * s, 0u, -1.0).x;
        let d1 = clipped_distance_and_trap(is_world, obj, family_override, center + k1 * s, 0u, -1.0).x;
        let d2 = clipped_distance_and_trap(is_world, obj, family_override, center + k2 * s, 0u, -1.0).x;
        let d3 = clipped_distance_and_trap(is_world, obj, family_override, center + k3 * s, 0u, -1.0).x;
        if signed_fam || (d0 > 0.0 && d1 > 0.0 && d2 > 0.0 && d3 > 0.0) {
            let g = k0 * d0 + k1 * d1 + k2 * d2 + k3 * d3;
            if length(g) >= 1.0e-7 {
                return normalize(g);
            }
            return fallback;
        }
        h = 0.5 * h;
    }
    return fallback;
}

fn direct_gsaa_variance(n: vec3<f32>, fine: vec3<f32>) -> f32 {
    let d = n - fine;
    return min(dot(d, d), GSAA_KAPPA);
}

fn direct_filter_roughness(r: f32, variance: f32) -> f32 {
    if variance == 0.0 {
        return r;
    }
    let alpha = r * r;
    return min(sqrt(sqrt(alpha * alpha + variance)), 1.0);
}

fn direct_shadow(
    is_world: bool,
    family_override: u32,
    point: vec3<f32>,
    n: vec3<f32>,
    l: vec3<f32>,
    eps: f32,
    angle: f32,
) -> f32 {
    let strength = param(P_SHADOW_STRENGTH);
    let steps = u32(param(P_SHADOW_STEPS));
    if strength == 0.0 || steps == 0u {
        return 1.0;
    }
    let max_distance = param(P_MAX_DISTANCE);
    var t = 4.0 * eps;
    let origin = point + n * t;
    let cap = 2.0 * max_distance / f32(steps);
    var limit = max_distance;
    if is_world {
        let span = march_span(true, origin, l);
        if span.x > span.y {
            return 1.0;
        }
        limit = span.y;
    } else if !single_no_clip() {
        let r = param(P_CLIP_RADIUS);
        if r > 0.0 {
            let oc = origin - param3(P_CLIP_CENTER);
            let b = dot(oc, l);
            let disc = b * b - (dot(oc, oc) - r * r);
            if disc <= 0.0 {
                return 1.0;
            }
            let t_exit = -b + sqrt(disc);
            if t_exit <= 0.0 {
                return 1.0;
            }
            limit = min(t_exit, max_distance);
        }
    }
    let sharpness = select(0.0, 1.0 / tan(angle), angle > 0.0);
    var visibility = 1.0;
    var i = 0u;
    while i < steps && t <= limit {
        let d = march_sample(is_world, family_override, origin + l * t, RAY_VISIBILITY).d;
        if d < eps {
            return max(1.0 - strength, 0.0);
        }
        if angle > 0.0 {
            visibility = min(visibility, clamp(sharpness * d / t, 0.0, 1.0));
        }
        t = t + min(max(0.5 * d, 2.0 * eps), cap);
        i = i + 1u;
    }
    return max(1.0 - strength * (1.0 - visibility), 0.0);
}

fn direct_ao(is_world: bool, family_override: u32, point: vec3<f32>, n: vec3<f32>) -> f32 {
    let strength = param(P_AO_STRENGTH);
    let steps = u32(param(P_AO_STEPS));
    if strength == 0.0 || steps == 0u {
        return 1.0;
    }
    let axis = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let tangent = normalize(cross(axis, n));
    let bitangent = cross(n, tangent);
    let directions = array<vec3<f32>, 5>(
        n,
        n * 0.6 + tangent * 0.8,
        n * 0.6 - tangent * 0.8,
        n * 0.6 + bitangent * 0.8,
        n * 0.6 - bitangent * 0.8,
    );
    let radius = param(P_AO_RADIUS);
    let inv_steps = radius / f32(steps);
    let max_sum = f32(steps * 5u);
    var sum = 0.0;
    for (var j = 0u; j < 5u; j = j + 1u) {
        let component = select(0.6, 1.0, j == 0u);
        for (var i = 1u; i <= steps; i = i + 1u) {
            let t = f32(i) * inv_steps;
            let clearance = t * component;
            let d = march_sample(is_world, family_override, point + directions[j] * t, RAY_VISIBILITY).d;
            sum = sum + clamp((clearance - d) / clearance, 0.0, 1.0);
        }
        if strength * sum >= max_sum {
            return 0.0;
        }
    }
    return max(1.0 - strength * sum / max_sum, 0.0);
}

fn direct_specular_ao(nv: f32, ao: f32, roughness: f32) -> f32 {
    return clamp(pow(nv + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

fn direct_shade(
    is_world: bool,
    obj: u32,
    family_override: u32,
    point: vec3<f32>,
    n: vec3<f32>,
    view: vec3<f32>,
    palette_rgb: vec3<f32>,
    eps: f32,
    variance: f32,
) -> vec3<f32> {
    let facing = select(-n, n, dot(n, view) > 0.0);
    let frame = SsFrame(facing, object_tangent_y(is_world, obj), false, 0.0);
    var inputs = surface_inputs(is_world, obj, palette_rgb, facing, view);
    inputs.specular_roughness = direct_filter_roughness(inputs.specular_roughness, variance);
    inputs.coat_roughness = direct_filter_roughness(inputs.coat_roughness, variance);
    let ao = direct_ao(is_world, family_override, point, n);
    var spec_ao = ao;
    var coat_ao = ao;
    if is_world {
        let nv = max(dot(facing, view), 0.0);
        spec_ao = direct_specular_ao(nv, ao, inputs.specular_roughness);
        coat_ao = direct_specular_ao(nv, ao, inputs.coat_roughness);
    }
    let specular_alpha = mx_average_alpha(ss_main_alpha(inputs));
    let coat_alpha = mx_average_alpha(ss_coat_alpha(inputs));
    let specular_dir = ss_dominant_dir(frame, view, specular_alpha);
    let coat_dir = ss_dominant_dir(frame, view, coat_alpha);
    var transmission_radiance = vec3<f32>(0.0);
    if is_world && inputs.transmission > 0.0 {
        let trans_alpha = mx_average_alpha(ss_transmission_alpha(inputs));
        let refracted = ss_refract(view, facing, inputs.specular_ior);
        let trans_dir = select(specular_dir, refracted, dot(refracted, refracted) > 0.0);
        transmission_radiance = sky_specular(true, trans_dir, trans_alpha) * spec_ao;
    }
    let environment = SsEnvironment(
        sky_irradiance(facing) * ao,
        sky_specular(is_world, specular_dir, specular_alpha) * spec_ao,
        sky_specular(is_world, coat_dir, coat_alpha) * coat_ao,
        transmission_radiance,
    );
    let reflected_lobes = ss_eval_environment(inputs, frame, view, environment);
    let reflected = reflected_lobes.base + reflected_lobes.specular + reflected_lobes.transmission;
    let emitted = ss_eval_emission(inputs, frame, view);
    var key = vec3<f32>(0.0);
    if is_world {
        let count = u32(param(P_LIGHT_COUNT));
        for (var i = 0u; i < count; i = i + 1u) {
            let intensity = lp(true, i, P_LIGHT_INTENSITY);
            if intensity > 0.0 {
                let l = lp3(true, i, P_LIGHT_DIR);
                let angle = acos(clamp(1.0 - lp(true, i, P_SUN_ONE_MINUS_COS), -1.0, 1.0));
                let sun_lobes = ss_eval_light_disc(inputs, frame, view, l, angle);
                let sun = sun_lobes.base + sun_lobes.specular + sun_lobes.transmission;
                if max(sun.x, max(sun.y, sun.z)) > 0.0 {
                    let vis = direct_shadow(true, family_override, point, n, l, eps, angle);
                    key = key + (lp3(true, i, P_LIGHT_COLOR) * sun) * (intensity * vis);
                }
            }
        }
    } else {
        let intensity = param(P_LIGHT_INTENSITY);
        if intensity > 0.0 {
            let l = param3(P_LIGHT_DIR);
            let angle = param(P_LIGHT_HALF_ANGLE);
            let sun_lobes = ss_eval_light_disc(inputs, frame, view, l, angle);
            let sun = sun_lobes.base + sun_lobes.specular + sun_lobes.transmission;
            if max(sun.x, max(sun.y, sun.z)) > 0.0 {
                let vis = direct_shadow(false, family_override, point, n, l, eps, angle);
                key = (param3(P_LIGHT_COLOR) * sun) * (intensity * vis);
            }
        }
    }
    return key + reflected + emitted;
}

fn direct_hit(m: MarchResult) -> bool {
    return m.hit || (m.unresolved && m.ratio <= 0.5 * param(P_SAMPLE_CONE));
}

fn trace_direct(
    is_world: bool,
    family_override: u32,
    px: i32,
    py: i32,
    sample: u32,
    start: f32,
) -> TraceOut {
    let grid = max(u32(param(P_DIRECT_GRID)), 1u);
    let sx = sample % grid;
    let sy = sample / grid;
    let inv = 1.0 / f32(grid);
    let jx = (f32(sx) + 0.5) * inv;
    let jy = (f32(sy) + 0.5) * inv;
    let ray = direct_camera_ray(px, py, jx, jy);
    let slope = param(P_FOOTPRINT);
    let m = march(is_world, family_override, ray.origin, ray.dir, RAY_PRIMARY, 0.0, slope, start);
    let unresolved = m.unresolved;
    if !direct_hit(m) {
        let bg = background(is_world, ray.dir);
        return TraceOut(bg, vec3<f32>(0.0), vec3<f32>(0.0), unresolved, sky_alpha());
    }
    let family = resolved_family(is_world, m.object, family_override);
    let center = select(m.point, m.point - ray.dir * m.eps, unsigned_family(family));
    let fine = direct_stencil(is_world, m.object, family_override, center, 0.5 * m.eps, -ray.dir);
    let cone = param(P_SAMPLE_CONE);
    var n = fine;
    if cone > 1.0 {
        n = direct_stencil(
            is_world,
            m.object,
            family_override,
            m.point + fine * (m.eps * cone),
            0.5 * m.eps * cone,
            fine,
        );
    }
    let variance = direct_gsaa_variance(n, fine);
    let view = -ray.dir;
    let pal = hit_palette(is_world, m.object, m.point, n, m.trap);
    var guide_albedo = pal;
    if u32(pr(is_world, m.object, P_COLOR_SOURCE)) != 0u {
        guide_albedo = pr3(is_world, m.object, P_BASE_COLOR);
    }
    let facing_exp = pr(is_world, m.object, P_FACING_EXPONENT);
    let facing = select(-n, n, dot(n, view) > 0.0);
    if facing_exp > 0.0 {
        let f = pow(max(1.0 - abs(dot(facing, view)), 0.0), facing_exp);
        guide_albedo = guide_albedo + (pr3(is_world, m.object, P_FACING_COLOR) - guide_albedo) * f;
    }
    let radiance = direct_shade(
        is_world,
        m.object,
        family_override,
        m.point,
        n,
        view,
        pal,
        m.eps,
        variance,
    );
    return TraceOut(radiance, guide_albedo, n, unresolved, 1.0);
}

fn trace_pixel(
    idx: u32,
    is_world: bool,
    family_override: u32,
    mat_spec: u32,
    is_direct: bool,
) {
    let i = idx;
    let tile = i / 32u;
    let lane = i % 32u;
    let size = tile_size();
    let width = size.x;
    let height = size.y;
    let tiles_x = (width + 7u) / 8u;
    let tiles_y = (height + 3u) / 4u;
    if tile >= tiles_x * tiles_y {
        return;
    }
    let tx = (tile % tiles_x) * 8u + lane % 8u;
    let ty = (tile / tiles_x) * 4u + lane / 8u;
    if tx >= width || ty >= height || active_tiles[tile] == 0u {
        return;
    }
    let to = tile_origin();
    let px = to.x + i32(tx);
    let py = to.y + i32(ty);
    if px < 0 || py < 0 || u32(px) >= u32(param(P_WIDTH)) || u32(py) >= u32(param(P_HEIGHT)) {
        return;
    }

    var acc = accum[i];
    var alb = albedo[i];
    var nrm = normal_buf[i];
    var st = stats[i];

    let sample_begin = u32(param(P_SAMPLE_BEGIN));
    let spp = u32(param(P_SPP));
    let aperture = param(P_APERTURE);

    var start = 0.0;
    if (is_direct && !is_world) || (!is_ofx() && aperture > 0.0) {
        start = 0.0;
    } else if sample_begin != 0u {
        start = st.z;
    } else {
        let center_ray = direct_camera_ray(px, py, 0.5, 0.5);
        start = cone_start(is_world, family_override, center_ray.origin, center_ray.dir);
    }

    let sample_end = sample_begin + spp;
    let ofx = is_ofx();
    var primary_hits = 0.0;
    for (var s = sample_begin; s < sample_end; s = s + 1u) {
        var out: TraceOut;
        if is_direct {
            out = trace_direct(is_world, family_override, px, py, s, start);
        } else {
            out = trace_path(is_world, family_override, mat_spec, px, py, s, start);
        }
        let rad = select(out.radiance, out.radiance * out.alpha, ofx);
        primary_hits = primary_hits + out.alpha;
        acc = vec4<f32>(acc.xyz + rad, acc.w + 1.0);
        alb = vec4<f32>(alb.xyz + out.albedo, alb.w + 1.0);
        nrm = vec4<f32>(nrm.xyz + out.normal, nrm.w + select(1.0, out.alpha, ofx));
        let lum = luminance(rad);
        st.x = st.x + lum * lum;
        if out.unresolved {
            st.y = st.y + 1.0;
        }
    }
    st.z = start;
    st.w = 0.0;
    accum[i] = acc;
    albedo[i] = alb;
    normal_buf[i] = nrm;
    stats[i] = st;
}

@compute @workgroup_size(128, 1, 1)
fn world(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, true, FAMILY_WORLD, MAT_DISPATCH, false);
}

@compute @workgroup_size(128, 1, 1)
fn world_fast(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, true, FAMILY_WORLD, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn world_fast_bulb(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, true, FAMILY_BULB, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn world_direct(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, true, FAMILY_WORLD, MAT_FULL, true);
}

@compute @workgroup_size(128, 1, 1)
fn fast_bulb(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_BULB, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn fast_box(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_BOX, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn fast_quat(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_QUAT, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn fast_kifs(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_KIFS, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn fast_kleinian(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_KLEINIAN, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn fast_pseudo_kleinian(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_PSEUDO_KLEINIAN, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn fast_apollonian(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_APOLLONIAN, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn fast_hybrid(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_HYBRID, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn full_bulb(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_BULB, MAT_FULL, false);
}

@compute @workgroup_size(128, 1, 1)
fn full_box(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_BOX, MAT_FULL, false);
}

@compute @workgroup_size(128, 1, 1)
fn full_quat(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_QUAT, MAT_FULL, false);
}

@compute @workgroup_size(128, 1, 1)
fn full_kifs(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_KIFS, MAT_FULL, false);
}

@compute @workgroup_size(128, 1, 1)
fn full_kleinian(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_KLEINIAN, MAT_FULL, false);
}

@compute @workgroup_size(128, 1, 1)
fn full_pseudo_kleinian(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_PSEUDO_KLEINIAN, MAT_FULL, false);
}

@compute @workgroup_size(128, 1, 1)
fn full_apollonian(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_APOLLONIAN, MAT_FULL, false);
}

@compute @workgroup_size(128, 1, 1)
fn full_hybrid(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_HYBRID, MAT_FULL, false);
}

@compute @workgroup_size(128, 1, 1)
fn direct_bulb(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_BULB, MAT_FULL, true);
}

@compute @workgroup_size(128, 1, 1)
fn direct_box(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_BOX, MAT_FULL, true);
}

@compute @workgroup_size(128, 1, 1)
fn direct_quat(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_QUAT, MAT_FULL, true);
}

@compute @workgroup_size(128, 1, 1)
fn direct_kifs(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_KIFS, MAT_FULL, true);
}

@compute @workgroup_size(128, 1, 1)
fn direct_kleinian(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_KLEINIAN, MAT_FULL, true);
}

@compute @workgroup_size(128, 1, 1)
fn direct_pseudo_kleinian(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_PSEUDO_KLEINIAN, MAT_FULL, true);
}

@compute @workgroup_size(128, 1, 1)
fn direct_apollonian(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_APOLLONIAN, MAT_FULL, true);
}

@compute @workgroup_size(128, 1, 1)
fn direct_hybrid(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_HYBRID, MAT_FULL, true);
}

@compute @workgroup_size(128, 1, 1)
fn single_fast(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_WORLD, MAT_FAST, false);
}

@compute @workgroup_size(128, 1, 1)
fn single_full(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_WORLD, MAT_FULL, false);
}

@compute @workgroup_size(128, 1, 1)
fn single_direct(@builtin(global_invocation_id) gid: vec3<u32>) {
    trace_pixel(gid.x, false, FAMILY_WORLD, MAT_FULL, true);
}

var<workgroup> wg_tally: array<vec4<f32>, 128>;

@compute @workgroup_size(128, 1, 1)
fn tally(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let threads = 128u;
    let n = arrayLength(&accum);
    var unresolved = 0.0;
    var samples = 0.0;
    var k = tid;
    while k < n {
        unresolved = unresolved + stats[k].y;
        samples = samples + accum[k].w;
        k = k + threads;
    }
    let size = tile_size();
    let tiles_x = (size.x + 7u) / 8u;
    let tiles_y = (size.y + 3u) / 4u;
    let total_tiles = tiles_x * tiles_y;
    var active_count = 0u;
    var t = tid;
    while t < total_tiles {
        if active_rw[t] != 0u {
            active_count = active_count + 1u;
        }
        t = t + threads;
    }
    wg_tally[tid] = vec4<f32>(unresolved, samples, f32(active_count), 0.0);
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if tid < stride {
            wg_tally[tid] = wg_tally[tid] + wg_tally[tid + stride];
        }
        workgroupBarrier();
    }
    if tid == 0u {
        summary_out[0] = wg_tally[0];
    }
}

@compute @workgroup_size(128, 1, 1)
fn adapt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile = gid.x;
    let size = tile_size();
    let width = size.x;
    let height = size.y;
    let tiles_x = (width + 7u) / 8u;
    let tiles_y = (height + 3u) / 4u;
    if tile >= tiles_x * tiles_y {
        return;
    }
    let threshold = param(P_ADAPT_THRESHOLD);
    let min_samples = param(P_ADAPT_MIN);
    let tx0 = (tile % tiles_x) * 8u;
    let ty0 = (tile / tiles_x) * 4u;
    let base = tile * 32u;
    var done = true;
    for (var lane = 0u; lane < 32u; lane = lane + 1u) {
        let x = tx0 + lane % 8u;
        let y = ty0 + lane / 8u;
        if x < width && y < height {
            let a = accum[base + lane];
            let n = a.w;
            if n < min_samples {
                done = false;
            } else {
                let mean = luminance(a.xyz) / n;
                let variance = max(stats[base + lane].x / n - mean * mean, 0.0);
                let err = sqrt(variance / n) / sqrt(max(mean, 0.0) + 1.0e-4);
                if err > threshold {
                    done = false;
                }
            }
        }
    }
    active_rw[tile] = select(1u, 0u, done);
}

fn to_709_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(TO_709_ROW0, c),
        dot(TO_709_ROW1, c),
        dot(TO_709_ROW2, c),
    );
}

fn srgb_u8_channel(v: f32) -> u32 {
    let c = clamp(v, 0.0, 1.0);
    if c <= 0.0 {
        return 0u;
    }
    if c >= 1.0 {
        return 255u;
    }
    if c <= 0.0031308 {
        return u32(c * (12.92 * 255.0) + 0.5);
    }
    return u32((1.055 * pow(c, 1.0 / 2.4) - 0.055) * 255.0 + 0.5);
}

fn pack_srgb_rgba8(rgb: vec3<f32>) -> u32 {
    let r = srgb_u8_channel(rgb.x);
    let g = srgb_u8_channel(rgb.y);
    let b = srgb_u8_channel(rgb.z);
    return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

@compute @workgroup_size(128, 1, 1)
fn tonemap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let size = tile_size();
    let width = size.x;
    let height = size.y;
    if i >= width * height {
        return;
    }
    let x = i % width;
    let y = i / width;
    let tiles_x = (width + 7u) / 8u;
    let tile = (y / 4u) * tiles_x + (x / 8u);
    let a = accum[tile * 32u + (y % 4u) * 8u + (x % 8u)];
    let inv = select(0.0, param(P_EXPOSURE) / a.w, a.w > 0.0);
    let c = a.xyz * inv;
    let l = luminance(c);
    let sat = param(P_SATURATION);
    aux_out[i] = vec4<f32>(l + (c - vec3<f32>(l)) * sat, 1.0);
}

@compute @workgroup_size(128, 1, 1)
fn tonemap_display(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let size = tile_size();
    let width = size.x;
    let height = size.y;
    if i >= width * height {
        return;
    }
    let x = i % width;
    let y = i / width;
    let tiles_x = (width + 7u) / 8u;
    let tile = (y / 4u) * tiles_x + (x / 8u);
    let a = accum[tile * 32u + (y % 4u) * 8u + (x % 8u)];
    let inv = select(0.0, param(P_EXPOSURE) / a.w, a.w > 0.0);
    let c = a.xyz * inv;
    let l = luminance(c);
    let sat = param(P_SATURATION);
    let raw = l + (c - vec3<f32>(l)) * sat;
    var rgb = to_709_rgb(raw);
    if param(P_TONEMAP) != 0.0 {
        let m = max(rgb, vec3<f32>(0.0));
        rgb = m / (vec3<f32>(1.0) + m);
    }
    aux_out[i] = vec4<f32>(rgb, 1.0);
    sdr_out[i] = pack_srgb_rgba8(rgb);
}

@compute @workgroup_size(1, 1, 1)
fn pick(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x != 0u {
        return;
    }
    let q = aux_out[0];
    let origin = param3(P_CAM_ORIGIN);
    let right = param3(P_CAM_RIGHT);
    let up = param3(P_CAM_UP);
    let fwd = param3(P_CAM_FORWARD);
    let hw = param(P_HALF_W);
    let hh = param(P_HALF_H);
    let dir = normalize(fwd + right * (q.x * hw) + up * (q.y * hh));
    let start = cone_start(true, FAMILY_WORLD, origin, dir);
    let m = march(true, FAMILY_WORLD, origin, dir, RAY_PRIMARY, 0.0, param(P_FOOTPRINT), start);
    if m.hit {
        let t = length(m.point - origin);
        let depth = t * dot(dir, fwd);
        aux_out[0] = vec4<f32>(f32(m.object + 1u), t, depth, 0.0);
    } else {
        aux_out[0] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
