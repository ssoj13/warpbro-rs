// WarpBro WGPU compute kernels: a port of fractal-kernels (CUDA, the reference, at the
// fractal-rs revision pinned in Cargo.lock) plus the WGPU-only extended families
// (`formulas.wgsl`). The classic estimates, orbit traps, scene masks and arrays, normals, the
// march, the integrator (light sampling, MIS, media) and the camera sample layout follow CUDA
// 1:1; the `parity_tests` module compares the two backends. Structural WGSL differences are
// documented where they occur: one dynamic entry point per (scope, shade) next to the family
// specializations (`render_wgpu::TraceEntry`), the private `normal_iterations` context instead
// of CUDA's `Context` field, the two-pass `tally`, and the OFX tile frame CUDA has no twin of.
//
// Pre-prepended by `render_wgpu::shader_source(family)` (one module per family + a dynamic one):
// 1. `standard_surface_bsdf::wgsl_source()`
// 2. All `P_*`, `O_*`, `M_*`, `FAMILY_*`, `MASK_*`, `ARRAY_*`, `PALETTE_SAMPLES`,
//    `TALLY_THREADS` constants and the `SIGNED_MASK` / `ITERATING_MASK` family bit sets of
//    `formulas::FORMULAS`.
// Appended by it: `family_distance` (the module family's estimator, or a switch over every
// family for the dynamic module: `render_wgpu::family_distance_wgsl`), the module's trace entry
// points (`render_wgpu::TraceEntry::all`), one line each calling
// `trace_pixel(linear_id(gid, nwg), is_world, family, mat_spec, is_direct)`, and its `pick`
// (`pick_ray(gid, family)`).

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
@group(0) @binding(13) var<storage, read_write> tally_parts: array<vec4<f32>>;

const TRAP_START: f32 = 3.40282347e38;
const DERIVATIVE_LIMIT: f32 = 1.0e38;
const STEP_CAP_STEPS: f32 = 256.0;
const CONE_START_STEPS: u32 = 256u;
const HIT_REFINEMENTS: u32 = 16u;
const GSAA_KAPPA: f32 = 0.18;

const RAY_PRIMARY: u32 = 0u;
const RAY_BOUNCE: u32 = 1u;
// A Direct camera ray: primary, its unresolved result may stand in for the surface
// (`trace_direct`).
const RAY_DIRECT: u32 = 3u;

const MAT_DISPATCH: u32 = 0u;
const MAT_FAST: u32 = 1u;
const MAT_FULL: u32 = 2u;

// The `family_override` of an entry that reads each object's `P_FAMILY` at run time. Outside
// the family codes on purpose: fractal-core's `FAMILY_WORLD` (8) is also the extended family
// Amazing Surface, so it cannot double as this sentinel (plan1 2.8).
const FAMILY_DYNAMIC: u32 = 0xffffffffu;

// The invocation index of a 128-wide entry over the 2D dispatch of `WgpuEngine::groups`: rows
// of `nwg.x` workgroups, so frames past the per-dimension workgroup limit (65,535 x 128
// invocations, ~8.4 MP) spill into `y`. Entries drop the indices past their range.
fn linear_id(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 128u;
}

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
    if family_override == FAMILY_DYNAMIC {
        return u32(pr(is_world, obj, P_FAMILY));
    }
    return family_override;
}

// Whether `family`'s estimate is signed (negative inside): its normal stencil is centred on the
// hit, an unsigned family's one footprint back along the ray. `SIGNED_MASK` is emitted from
// `formulas::FORMULAS` (fractal-kernels `family_signed` for the classic eight).
fn family_signed(family: u32) -> bool {
    return ((SIGNED_MASK >> family) & 1u) != 0u;
}

// Whether `family` runs an iteration loop controlled by `P_ITERATIONS` (`ITERATING_MASK`, from
// `formulas::FORMULAS`): only those have a fractional-iteration endpoint to blend.
fn family_iterates(family: u32) -> bool {
    return ((ITERATING_MASK >> family) & 1u) != 0u;
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

// fractal-kernels `trap3`: 1 the orbit's distance to the origin, 2 to the plane through the trap
// point, 3 (TrapPoint) to the trap point itself.
fn eval_trap3(t: Trap3, v: vec3<f32>) -> f32 {
    if t.mode == 1u {
        return length(v);
    }
    let rel = v - t.point;
    if t.mode == 2u {
        return abs(dot(rel, t.normal));
    }
    return length(rel);
}

// The quaternion trap (fractal-kernels `quat_distance`): the trap geometry embedded at w = 0.
fn eval_trap4(t: Trap3, q: vec4<f32>) -> f32 {
    if t.mode == 1u {
        return length(q);
    }
    let rel = q.xyz - t.point;
    if t.mode == 2u {
        return abs(dot(rel, t.normal));
    }
    return sqrt(dot(rel, rel) + q.w * q.w);
}

struct Fraction {
    run: u32,
    f: f32,
    saved_d: f32,
    saved_trap: f32,
    has_saved: bool,
}

// fractal-kernels `Context::normal_iterations`: while a fractional-iteration normal re-marches
// one endpoint ray (`endpoint_ray_normal`), object `object` runs exactly `count` iterations in
// every estimate of that march and its stencil. A per-invocation context, as CUDA's, instead of
// a parameter on every estimate; off (zero-initialized) otherwise.
struct NormalIterations {
    on: bool,
    object: u32,
    count: f32,
}

var<private> normal_iterations: NormalIterations;

// The iteration count object `obj` runs: `P_ITERATIONS` (at least 0), or the endpoint count of
// `normal_iterations` for its object.
fn object_iterations(is_world: bool, obj: u32) -> f32 {
    if normal_iterations.on && normal_iterations.object == obj {
        return normal_iterations.count;
    }
    return max(pr(is_world, obj, P_ITERATIONS), 0.0);
}

fn fraction_new(is_world: bool, obj: u32) -> Fraction {
    let n = object_iterations(is_world, obj);
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
        // Power 8, unit angle scales, no phase, no rotation (fractal-kernels `bulb_step`): z^8
        // from the double angles 2θ / 2φ and two more doublings instead of acos / atan2 / pow.
        let r2 = r * r;
        let r4 = r2 * r2;
        let growth = 8.0 * r4 * r2 * r;
        if *dr > DERIVATIVE_LIMIT / growth {
            return false;
        }
        *dr = growth * (*dr) + dr_constant;
        let x = (*z).x;
        let y = (*z).y;
        let w = (*z).z;
        let rxy2 = x * x + y * y;
        let r8 = r4 * r4;
        if rxy2 < 1.0e-20 {
            *z = vec3<f32>(0.0, 0.0, r8) + c;
            return true;
        }
        let inv_r2 = 1.0 / r2;
        var ct = 2.0 * w * w * inv_r2 - 1.0;
        var st = 2.0 * w * sqrt(rxy2) * inv_r2;
        let inv_rxy2 = 1.0 / rxy2;
        var cp = (x * x - y * y) * inv_rxy2;
        var sp = 2.0 * x * y * inv_rxy2;
        for (var k = 0u; k < 2u; k = k + 1u) {
            let nct = ct * ct - st * st;
            st = 2.0 * st * ct;
            ct = nct;
            let ncp = cp * cp - sp * sp;
            sp = 2.0 * sp * cp;
            cp = ncp;
        }
        *z = vec3<f32>(st * cp * r8, st * sp * r8, ct * r8) + c;
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
    var frac = fraction_new(is_world, obj);
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

// The Mandelbulb estimates at object point `q` after `run - 1` and `run` iterations, without a
// trap (fractal-kernels `bulb_distance_endpoints`): the two endpoints of a fractional count.
fn bulb_distance_endpoints(is_world: bool, obj: u32, q: vec3<f32>, run: u32) -> vec2<f32> {
    let oc = orbit_constant(is_world, obj, q);
    let c = oc.xyz;
    let dr_constant = oc.w;
    var z = q;
    var dr = 1.0;
    var r = length(z);
    let bailout = pr(is_world, obj, P_BAILOUT);
    let cfg = load_bulb_config(is_world, obj);
    let rot = load_iter_rot(is_world, obj);
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
    var frac = fraction_new(is_world, obj);
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
    var frac = fraction_new(is_world, obj);
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
) -> vec2<f32> {
    var z = q;
    var dr = 1.0;
    var trap = TRAP_START;
    let cfg = load_kifs_config(is_world, obj);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj);
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

// `x` wrapped into [left, left + width).
fn kleinian_wrap(x: f32, width: f32, left: f32) -> f32 {
    let shifted = x - left;
    return shifted - width * floor(shifted / width) + left;
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
) -> vec2<f32> {
    // fractal-kernels `kleinian_distance`: the Maskit slice wrapped into its strip, reflected
    // across the separation line, then inverted; it stops on escape or on a 2-cycle.
    let a = pr(is_world, obj, P_KLEIN_A);
    let b = pr(is_world, obj, P_KLEIN_B);
    let skew = pr(is_world, obj, P_KLEIN_SKEW);
    let line_amp = pr(is_world, obj, P_KLEIN_LINE_AMP);
    let line_rate = pr(is_world, obj, P_KLEIN_LINE_RATE);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj);
    let half_a = 0.5 * a;
    let half_b = 0.5 * b;
    let sign_b = exact_sign(b);
    // face_camera: (x, z, -y), then half a cell up.
    var z = vec3<f32>(q.x, q.z + half_a, -q.y);
    var dr = 1.0;
    var previous = z;
    var before = z;
    var trap = TRAP_START;
    for (var n = 0u; n < frac.run; n = n + 1u) {
        if fraction_due(frac, n) {
            fraction_save(&frac, min(z.y, 0.05) / max(dr, 1.0), trap);
        }
        var x = z.x;
        var y = z.y;
        var w = z.z;
        let s = skew * y;
        x = kleinian_wrap(x + s, 2.0, -1.0) - s;
        w = kleinian_wrap(w, 2.0, -1.0);
        let u = x + half_b;
        let line = half_a + sign_b * line_amp * exact_sign(u) * (1.0 - exp(-line_rate * abs(u)));
        if y >= line {
            x = -b - x;
            y = a - y;
            w = -w;
        }
        let r2 = x * x + y * y + w * w;
        if r2 < 1.0e-30 {
            break;
        }
        let inverse = 1.0 / r2;
        if inverse > 1.0 && dr > DERIVATIVE_LIMIT / inverse {
            break;
        }
        z = vec3<f32>(x * inverse - b, a - y * inverse, -w * inverse);
        dr = dr * inverse;
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
        if dot(z, z) > 100.0 {
            break;
        }
        if n >= 1u {
            let gap = z - before;
            if dot(gap, gap) < 1.0e-12 {
                break;
            }
        }
        before = previous;
        previous = z;
    }
    return fraction_blend(frac, min(z.y, 0.05) / max(dr, 1.0), trap);
}

fn pseudo_kleinian_estimate(z: vec3<f32>, dr: f32, offset: vec3<f32>, thickness: f32) -> f32 {
    let o = z - offset;
    let thingy = abs(sqrt(o.x * o.x + o.y * o.y) * o.z) - thickness;
    return 0.5 * thingy / max(sqrt(dot(o, o) + abs(thickness)), 1.0e-30) / dr;
}

fn pseudo_kleinian_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let csize = pr3(is_world, obj, P_PK_BOX);
    let size = pr(is_world, obj, P_PK_SIZE);
    let c = pr3(is_world, obj, P_PK_C);
    let offset = pr3(is_world, obj, P_PK_OFFSET);
    let thickness = pr(is_world, obj, P_PK_THICKNESS);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var p = q;
    var dr = 1.0;
    var trap = TRAP_START;
    var frac = fraction_new(is_world, obj);
    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, pseudo_kleinian_estimate(p, dr, offset, thickness), trap);
        }
        p = clamp(p, -csize, csize) * 2.0 - p;
        let r2 = dot(p, p);
        if r2 < 1.0e-30 {
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

// fractal-kernels `apollonian_step`: the lattice wrap into [-1, 1)^3, then the inversion
// k = max(scale / r2, 0.1). No per-iteration rotation, also inside a Hybrid. Returns false to stop.
fn apollonian_step_cfg(scale: f32, z: ptr<function, vec3<f32>>, dr: ptr<function, f32>) -> bool {
    let h = (*z) * 0.5 + vec3<f32>(0.5);
    let w = vec3<f32>(-1.0) + (h - floor(h)) * 2.0;
    let r2 = dot(w, w);
    if r2 < 1.0e-30 {
        return false;
    }
    let k = max(scale / r2, 0.1);
    if *dr > DERIVATIVE_LIMIT / k {
        return false;
    }
    *z = w * k;
    *dr = (*dr) * k;
    return true;
}

fn apollonian_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    var z = q + vec3<f32>(0.0, 1.0, 0.0);
    var dr = 1.0;
    var trap = TRAP_START;
    let scale = pr(is_world, obj, P_APOLLO_SCALE);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj);
    for (var i = 0u; i < frac.run; i = i + 1u) {
        if fraction_due(frac, i) {
            fraction_save(&frac, 0.25 * abs(z.y) / dr, trap);
        }
        if !apollonian_step_cfg(scale, &z, &dr) {
            break;
        }
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, 0.25 * abs(z.y) / dr, trap);
}

// fractal-kernels `hybrid_distance`: up to four step slots (3 bits each, 1 bulb, 2 box, 3 KIFS
// fold, 4 Apollonian inversion, any other code a no-op) cycled per iteration, estimate
// 0.5 r / dr with no logarithm (a Hybrid's bailout may be below 1).
fn hybrid_distance(
    is_world: bool,
    obj: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    var z = q;
    let c = q;
    var dr = 1.0;
    var trap = TRAP_START;
    let bailout = pr(is_world, obj, P_BAILOUT);
    let count = max(u32(pr(is_world, obj, P_HYBRID_COUNT)), 1u);
    let steps = u32(pr(is_world, obj, P_HYBRID_STEPS));
    let bulb_cfg = load_bulb_config(is_world, obj);
    let box_cfg = load_box_config(is_world, obj);
    let kifs_cfg = load_kifs_config(is_world, obj);
    let apollo_scale = pr(is_world, obj, P_APOLLO_SCALE);
    let rot = load_iter_rot(is_world, obj);
    let trap_cfg = load_trap3(is_world, obj, trap_mode);
    var frac = fraction_new(is_world, obj);
    for (var i = 0u; i < frac.run; i = i + 1u) {
        let r = length(z);
        if fraction_due(frac, i) {
            fraction_save(&frac, 0.5 * r / dr, trap);
        }
        if r > bailout {
            break;
        }
        let step = (steps >> (3u * (i % count))) & 7u;
        var ok = true;
        if step == 1u {
            ok = r >= 1.0e-8 && bulb_step_cfg(bulb_cfg, rot, &z, &dr, r, c, 1.0);
        } else if step == 2u {
            ok = box_step_cfg(box_cfg, rot, &z, &dr, c, 1.0);
        } else if step == 3u {
            ok = kifs_step_cfg(kifs_cfg, rot, &z, &dr);
        } else if step == 4u {
            ok = apollonian_step_cfg(apollo_scale, &z, &dr);
        }
        if !ok {
            break;
        }
        if trap_mode != 0u {
            trap = min(trap, eval_trap3(trap_cfg, z));
        }
    }
    return fraction_blend(frac, 0.5 * length(z) / dr, trap);
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

// Scene masks: a 1:1 port of fractal-kernels `mask_distance` and its array placement
// (`fractal_core::mask_array`), so a mask cuts the same geometry on both backends.

// Whether a Cells mask removes the cell of hash `h`: its top 24 bits as a unit value fall below
// `density`. A cell's hash is `lowbias32(x ^ lowbias32(y ^ lowbias32(z ^ seed)))`.
fn cell_removed(h: u32, density: f32) -> bool {
    return f32(h >> 8u) * (1.0 / 16777216.0) < density;
}

// Signed distance of one sphere or box at the local origin.
fn shape_distance(shape: u32, q: vec3<f32>, size: vec3<f32>) -> f32 {
    if shape == MASK_SPHERE {
        return length(q) - size.x;
    }
    let b = abs(q) - size;
    let outside = vec3<f32>(max(b.x, 0.0), max(b.y, 0.0), max(b.z, 0.0));
    let inside = min(max(b.x, max(b.y, b.z)), 0.0);
    return length(outside) + inside;
}

// `mask_array::angle`: radial copy `index` of `count`, in radians. A `sweep` of 0 is a full
// circle (no copy on top of the first); otherwise the copies span `sweep` end to end.
fn mask_array_angle(index: u32, count: u32, sweep: f32) -> f32 {
    let i = f32(index);
    if sweep == 0.0 {
        return i * MX_TWO_PI / f32(count);
    }
    if count <= 1u {
        return 0.0;
    }
    return i * sweep / (f32(count) - 1.0);
}

// `mask_array::ring_point`: centre of the radial copy at angle (`sn`, `cs`) on a ring of
// `radius` about `axis` (0 X, 1 Y, larger codes Z).
fn ring_point(axis: u32, radius: f32, sn: f32, cs: f32) -> vec3<f32> {
    switch axis {
        case 0u: {
            return vec3<f32>(0.0, radius * cs, radius * sn);
        }
        case 1u: {
            return vec3<f32>(radius * cs, 0.0, radius * sn);
        }
        default: {
            return vec3<f32>(radius * cs, radius * sn, 0.0);
        }
    }
}

// `mask_array::ring_basis`: the aligned copy's local axes as rows (outward, up = the ring axis,
// tangent), so `local = rows * (p - centre)`.
fn ring_basis(axis: u32, sn: f32, cs: f32) -> array<vec3<f32>, 3> {
    switch axis {
        case 0u: {
            return array<vec3<f32>, 3>(
                vec3<f32>(0.0, cs, sn),
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, sn, -cs),
            );
        }
        case 1u: {
            return array<vec3<f32>, 3>(
                vec3<f32>(cs, 0.0, sn),
                vec3<f32>(0.0, 1.0, 0.0),
                vec3<f32>(-sn, 0.0, cs),
            );
        }
        default: {
            return array<vec3<f32>, 3>(
                vec3<f32>(cs, sn, 0.0),
                vec3<f32>(0.0, 0.0, 1.0),
                vec3<f32>(sn, -cs, 0.0),
            );
        }
    }
}

// Distance to copy `index` of the linear or radial array of the mask row at `base`. Align turns
// a box into the copy's ring basis; a sphere ignores it.
fn array_copy_distance(
    base: u32,
    shape: u32,
    p: vec3<f32>,
    size: vec3<f32>,
    count: u32,
    index: u32,
) -> f32 {
    if objects[base + M_ARRAY] != f32(ARRAY_RADIAL) {
        let offset = vec3<f32>(
            objects[base + M_ARRAY_OFFSET],
            objects[base + M_ARRAY_OFFSET + 1u],
            objects[base + M_ARRAY_OFFSET + 2u],
        );
        return shape_distance(shape, p - offset * f32(index), size);
    }
    let angle = mask_array_angle(index, count, objects[base + M_ARRAY_SWEEP]);
    let sn = sin(angle);
    let cs = cos(angle);
    let axis = u32(objects[base + M_ARRAY_AXIS]);
    let d = p - ring_point(axis, objects[base + M_ARRAY_RADIUS], sn, cs);
    if objects[base + M_ARRAY_ALIGN] == 0.0 || shape != MASK_BOX {
        return shape_distance(shape, d, size);
    }
    let rows = ring_basis(axis, sn, cs);
    return shape_distance(shape, vec3<f32>(dot(d, rows[0]), dot(d, rows[1]), dot(d, rows[2])), size);
}

// Union of the array copies (`mask_array::count` of them).
fn array_union(base: u32, shape: u32, p: vec3<f32>, size: vec3<f32>) -> f32 {
    let count = clamp(u32(objects[base + M_ARRAY_COUNT]), 1u, ARRAY_MAX);
    var best = array_copy_distance(base, shape, p, size, count, 0u);
    for (var i = 1u; i < count; i = i + 1u) {
        let d = array_copy_distance(base, shape, p, size, count, i);
        if d < best {
            best = d;
        }
    }
    return best;
}

// Per axis, the distance to the neighbour cell at offset -1 / +1 (0 for offset 0), `t` the
// point's place in its own cell.
fn cell_gap(t: f32, offset: i32) -> f32 {
    if offset < 0 {
        return t;
    }
    if offset > 0 {
        return 1.0 - t;
    }
    return 0.0;
}

// A Cells mask in mask space, in cell edges times `edge`: the distance to the nearest of the 26
// neighbour cells in the other state (exact within the 3x3x3 block), else the block's boundary
// as the lower bound; positive in a removed cell. The zero set is exactly the border between
// removed and kept cells.
fn cells_distance(base: u32, p: vec3<f32>, edge: f32) -> f32 {
    let f = p / edge;
    let cell = floor(f);
    let t = f - cell;
    let c = vec3<i32>(cell);
    // The host packs the integer seed as an f32 (`render_prep::mask_row`).
    let seed = u32(objects[base + M_SEED]);
    let density = objects[base + M_DENSITY];
    // The hash's inner levels of the 3x3x3 block, shared by its cells: z (3), then y and z (9).
    var zy: array<u32, 9>;
    for (var k = 0; k < 3; k = k + 1) {
        let hz = ss_lowbias32(bitcast<u32>(c.z + k - 1) ^ seed);
        for (var j = 0; j < 3; j = j + 1) {
            zy[j * 3 + k] = ss_lowbias32(bitcast<u32>(c.y + j - 1) ^ hz);
        }
    }
    let removed = cell_removed(ss_lowbias32(bitcast<u32>(c.x) ^ zy[4]), density);
    var best = 1.0 + min(min(min(t.x, 1.0 - t.x), min(t.y, 1.0 - t.y)), min(t.z, 1.0 - t.z));
    for (var n = 0; n < 27; n = n + 1) {
        let o = vec3<i32>(n / 9 - 1, (n / 3) % 3 - 1, n % 3 - 1);
        // A cell no nearer than the best so far cannot lower it: its hash is skipped.
        let near = length(vec3<f32>(cell_gap(t.x, o.x), cell_gap(t.y, o.y), cell_gap(t.z, o.z)));
        if n != 13
            && near < best
            && cell_removed(ss_lowbias32(bitcast<u32>(c.x + o.x) ^ zy[n % 9]), density) != removed
        {
            best = near;
        }
    }
    return select(-edge * best, edge * best, removed);
}

// A scene mask's distance at world point `x` (a lower bound in world units, positive where the
// mask removes): the shape's signed distance in mask space, scaled conservatively, and negated
// when inverted (every shape, Cells included).
fn mask_distance(base: u32, x: vec3<f32>) -> f32 {
    let p = mask_point(base, x);
    let shape = u32(objects[base + M_SHAPE]);
    let size = vec3<f32>(
        objects[base + M_SIZE],
        objects[base + M_SIZE + 1u],
        objects[base + M_SIZE + 2u],
    );
    var local: f32;
    if shape == MASK_CELLS {
        local = cells_distance(base, p, size.x);
    } else if objects[base + M_ARRAY] != 0.0 {
        local = array_union(base, shape, p, size);
    } else {
        local = shape_distance(shape, p, size);
    }
    let d = local * objects[base + M_SCALE];
    return select(d, -d, objects[base + M_INVERT] != 0.0);
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

// `fractal_core::inversion::apply` at object point `q`: the transformed point and the map's
// largest local stretch `g` (the estimate there is divided by it), or `q` itself with `g` 1 at
// zero strength, where the division is exact and the estimate stays the uninverted one.
fn inversion(is_world: bool, obj: u32, q: vec3<f32>) -> vec4<f32> {
    let s = pr(is_world, obj, P_INVERT_STRENGTH);
    if s == 0.0 {
        return vec4<f32>(q, 1.0);
    }
    let center = pr3(is_world, obj, P_INVERT_CENTER);
    let radius = pr(is_world, obj, P_INVERT_RADIUS);
    let u = q - center;
    let r2 = max(dot(u, u), 1.0e-30);
    let g = (1.0 - s) + s * radius * radius / r2;
    return vec4<f32>(center + u * g, g);
}

// The (distance, trap) of the object's formula at its own point `q`, intersected with the
// formula's ball (`P_BOUND_RADIUS`), through the spherical inversion pre-transform
// ([`inversion`]), as fractal-kernels `inverted_estimate` / `bounded_distance_and_trap`. The
// one estimate path of every march, stencil and probe, before the object's clip and scale.
fn formula_estimate(
    is_world: bool,
    obj: u32,
    family_override: u32,
    q: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let pg = inversion(is_world, obj, q);
    let dt = family_distance(is_world, obj, family_override, pg.xyz, trap_mode);
    return vec2<f32>(formula_bound(is_world, obj, pg.xyz, dt.x) / pg.w, dt.y);
}

// The object-space estimate `d` at object point `q` (`len_q = |q|`) clipped by a world object's
// sphere, then scaled to the scene and intersected with the world object's masks
// (fractal-kernels `clipped_distance_and_trap`). Scene point `x`.
fn scene_distance(is_world: bool, obj: u32, x: vec3<f32>, len_q: f32, d: f32) -> f32 {
    if is_world {
        let base = obj * OBJECT_STRIDE;
        let clipped = max(d, len_q - objects[base + O_CLIP_RADIUS]);
        return masked(obj, x, clipped * objects[base + O_DISTANCE_SCALE]);
    }
    return d * param(P_OBJ_SCALE);
}

// [`formula_estimate`] at object point `q` (scene point `x`, `len_q = |q|` in a world) as a scene
// distance ([`scene_distance`]) and its trap.
fn clipped_distance_and_trap_q(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    q: vec3<f32>,
    len_q: f32,
    trap_mode: u32,
) -> vec2<f32> {
    let dt = formula_estimate(is_world, obj, family_override, q, trap_mode);
    return vec2<f32>(scene_distance(is_world, obj, x, len_q, dt.x), dt.y);
}

// Object `obj`'s scene estimate and trap at scene point `x` (fractal-kernels
// `signed_distance_and_trap` / `world_estimate` of one object).
fn clipped_distance_and_trap(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    trap_mode: u32,
) -> vec2<f32> {
    let q = object_point(is_world, obj, x);
    let len_q = select(0.0, length(q), is_world);
    return clipped_distance_and_trap_q(is_world, obj, family_override, x, q, len_q, trap_mode);
}

// The march below is a 1:1 port of fractal-kernels' `march` section (`march_sample`,
// `fractional_march_distance`, `march_span`, `cone_start`, `march_visibility`, `leave_ray`,
// `march_leaving`, `visibility_leaving`, `march`, `march_exit` with fractal-core
// `transmission::exit_distance`). A world entry specialized to one family still loops over
// every object (CUDA's specialized world kernel serves one object only); the estimates are the
// same, so the result is.

const F32_MAX: f32 = 3.40282347e38;
const FOOTPRINT_RELATIVE_FLOOR: f32 = 0.0000019073486;

// Neither NaN nor infinite.
fn finite(v: f32) -> bool {
    return abs(v) <= F32_MAX;
}

struct SceneSample {
    d: f32,
    trap: f32,
    object: u32,
}

fn footprint(base: f32, slope: f32, t: f32, x: vec3<f32>) -> f32 {
    return max(base + slope * t, length(x) * FOOTPRINT_RELATIVE_FLOOR);
}

// World object `obj`'s estimate at `x` clamped at 0, with its orbit trap when `traps`.
fn world_sample(obj: u32, family_override: u32, x: vec3<f32>, traps: bool) -> SceneSample {
    let mode = select(0u, u32(pr(true, obj, P_COLOR_MODE)), traps);
    let dt = clipped_distance_and_trap(true, obj, family_override, x, mode);
    return SceneSample(max(dt.x, 0.0), dt.y, obj);
}

// The scene estimate at `x` clamped at 0, its orbit trap (each object's `P_COLOR_MODE` when
// `traps`, else trap mode 0) and its object (fractal-kernels `march_sample`). A world evaluates
// the object whose clip sphere is nearest first, then skips every object whose sphere lies
// beyond the best estimate: its estimate cannot be lower, so the result is the full minimum,
// the lower index winning ties.
fn march_sample(is_world: bool, family_override: u32, x: vec3<f32>, traps: bool) -> SceneSample {
    if !is_world {
        let mode = select(0u, u32(param(P_COLOR_MODE)), traps);
        let dt = clipped_distance_and_trap(false, 0u, family_override, x, mode);
        return SceneSample(max(dt.x, 0.0), dt.y, 0u);
    }
    let count = u32(param(P_OBJECT_COUNT));
    if count == 0u {
        return SceneSample(F32_MAX, TRAP_START, 0u);
    }
    if count == 1u {
        return world_sample(0u, family_override, x, traps);
    }
    var first = 0u;
    var min_reach = (length(object_point(true, 0u, x)) - objects[O_CLIP_RADIUS]) * objects[O_DISTANCE_SCALE];
    for (var i = 1u; i < count; i = i + 1u) {
        let row = i * OBJECT_STRIDE;
        let reach = (length(object_point(true, i, x)) - objects[row + O_CLIP_RADIUS])
            * objects[row + O_DISTANCE_SCALE];
        if reach < min_reach {
            min_reach = reach;
            first = i;
        }
    }
    var best = world_sample(first, family_override, x, traps);
    for (var i = 0u; i < count; i = i + 1u) {
        if i == first {
            continue;
        }
        let row = i * OBJECT_STRIDE;
        let q = object_point(true, i, x);
        let len_q = length(q);
        let reach = (len_q - objects[row + O_CLIP_RADIUS]) * objects[row + O_DISTANCE_SCALE];
        if select(reach >= best.d, reach > best.d, i < best.object) {
            continue;
        }
        let mode = select(0u, u32(pr(true, i, P_COLOR_MODE)), traps);
        let dt = clipped_distance_and_trap_q(true, i, family_override, x, q, len_q, mode);
        let d = max(dt.x, 0.0);
        if d < best.d || (d == best.d && i < best.object) {
            best = SceneSample(d, dt.y, i);
        }
    }
    return best;
}

// Object `obj` is a Mandelbulb with a usable bailout and power and a fractional iteration
// count: the march steps on its two endpoint estimates (fractal-kernels `fractional_bulb`).
fn fractional_bulb(is_world: bool, obj: u32, family_override: u32) -> bool {
    let power = pr(is_world, obj, P_BULB_POWER);
    let bailout = pr(is_world, obj, P_BAILOUT);
    return resolved_family(is_world, obj, family_override) == FAMILY_BULB
        && bailout >= 1.0
        && bailout <= F32_MAX
        && power > 0.0
        && power <= F32_MAX
        && fraction_new(is_world, obj).f > 0.0;
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

// The scene estimates of object `obj` after `run - 1` and `run` iterations and of their blend by
// `frac` (clamped at 0): (lower, upper, blend), through the inversion like every estimate
// (fractal-kernels `fractional_bulb_endpoints`).
fn fractional_bulb_endpoints(
    is_world: bool,
    obj: u32,
    x: vec3<f32>,
    run: u32,
    frac: f32,
) -> vec3<f32> {
    let q = object_point(is_world, obj, x);
    let len_q = select(0.0, length(q), is_world);
    let pg = inversion(is_world, obj, q);
    let ends = bulb_distance_endpoints(is_world, obj, pg.xyz, run);
    let blend = ends.x + (ends.y - ends.x) * frac;
    let a = scene_distance(is_world, obj, x, len_q, formula_bound(is_world, obj, pg.xyz, ends.x) / pg.w);
    let b = scene_distance(is_world, obj, x, len_q, formula_bound(is_world, obj, pg.xyz, ends.y) / pg.w);
    let w = scene_distance(is_world, obj, x, len_q, formula_bound(is_world, obj, pg.xyz, blend) / pg.w);
    return vec3<f32>(a, b, max(w, 0.0));
}

// One object's step distance of a march over a fractional Mandelbulb (fractal-kernels
// `fractional_march_distance`): a weighted hit needs each nonnegative endpoint's contribution
// within `eps`, and `(weight * distance - eps) / (weight * ray_speed + epsilon_slope)` bounds
// the advance to that endpoint's band, so the step is the largest of the blend and both margins.
// An invalid endpoint cannot authorize a longer step.
fn fractional_object_distance(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    eps: f32,
    epsilon_slope: f32,
    ray_speed: f32,
    nearest: u32,
    weighted_distance: f32,
) -> f32 {
    if fractional_bulb(is_world, obj, family_override) {
        let f = fraction_new(is_world, obj);
        let e = fractional_bulb_endpoints(is_world, obj, x, f.run, f.f);
        let weighted = select(e.z, weighted_distance, obj == nearest);
        if e.x >= 0.0 && e.x <= F32_MAX && e.y >= 0.0 && e.y <= F32_MAX {
            let lower_weight = 1.0 - f.f;
            let lower_margin = max(lower_weight * e.x - eps, 0.0) / (lower_weight * ray_speed + epsilon_slope);
            let upper_margin = max(f.f * e.y - eps, 0.0) / (f.f * ray_speed + epsilon_slope);
            return max(weighted, max(lower_margin, upper_margin));
        }
        return weighted;
    }
    if obj == nearest {
        return weighted_distance;
    }
    return max(clipped_distance_and_trap(is_world, obj, family_override, x, 0u).x, 0.0);
}

// The march step distance at `x` when an object is a fractional Mandelbulb: the minimum of the
// objects' [`fractional_object_distance`], nearest-sphere first with the same skip as
// [`march_sample`]. `nearest` / `weighted_distance` are the sample's object and estimate.
fn fractional_march_distance(
    is_world: bool,
    family_override: u32,
    x: vec3<f32>,
    eps: f32,
    epsilon_slope: f32,
    ray_speed: f32,
    nearest: u32,
    weighted_distance: f32,
) -> f32 {
    let count = select(1u, u32(param(P_OBJECT_COUNT)), is_world);
    if count == 0u {
        return F32_MAX;
    }
    let first = min(nearest, count - 1u);
    var best = fractional_object_distance(
        is_world, first, family_override, x, eps, epsilon_slope, ray_speed, nearest, weighted_distance,
    );
    if count == 1u {
        return best;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        if i == first {
            continue;
        }
        let row = i * OBJECT_STRIDE;
        let reach = (length(object_point(true, i, x)) - objects[row + O_CLIP_RADIUS])
            * objects[row + O_DISTANCE_SCALE];
        if reach >= best {
            continue;
        }
        best = min(best, fractional_object_distance(
            true, i, family_override, x, eps, epsilon_slope, ray_speed, nearest, weighted_distance,
        ));
    }
    return best;
}

// The interval `[enter, exit]` a ray marches over, `ok` false when it misses the bounding sphere
// (fractal-kernels `march_span`). Outside the sphere every estimate is an exact "outside", so
// marching there only burns steps; a single object's Direct preview and an unclipped OFX render
// march `[0, P_MAX_DISTANCE]`. `enter` may exceed `exit` when the sphere lies beyond
// `P_MAX_DISTANCE`: every march of that span then ends at once.
struct Span {
    ok: bool,
    enter: f32,
    exit: f32,
}

fn march_span(is_world: bool, origin: vec3<f32>, dir: vec3<f32>) -> Span {
    let max_distance = param(P_MAX_DISTANCE);
    if (!is_world && param(P_DIRECT) != 0.0) || single_no_clip() {
        return Span(true, 0.0, max_distance);
    }
    let oc = origin - param3(P_CLIP_CENTER);
    let radius = param(P_CLIP_RADIUS);
    let b = dot(oc, dir);
    let disc = b * b - (dot(oc, oc) - radius * radius);
    if disc <= 0.0 {
        return Span(false, 0.0, 0.0);
    }
    let root = sqrt(disc);
    let t_exit = -b + root;
    if t_exit <= 0.0 {
        return Span(false, 0.0, 0.0);
    }
    return Span(true, max(-b - root, 0.0), min(t_exit, max_distance));
}

// The ray length below which no camera ray of the pixel with centre ray `dir` can hit
// (fractal-kernels `cone_start`): a cone march along the centre ray with the pixel cone of one
// pixel on the unit image plane, its own step budget (`CONE_START_STEPS`), 0 when the centre ray
// misses the bounding sphere. Valid for a pinhole camera only.
fn cone_start(is_world: bool, family_override: u32, origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    let span = march_span(is_world, origin, dir);
    if !span.ok {
        return 0.0;
    }
    let step_cap = 2.0 * (span.exit - span.enter) / STEP_CAP_STEPS;
    let slope = param(P_FOOTPRINT);
    let w = 2.0 * param(P_HALF_W) / param(P_WIDTH);
    let h = 2.0 * param(P_HALF_H) / param(P_HEIGHT);
    let cone = 0.5 * sqrt(w * w + h * h);
    var t = span.enter;
    for (var i = 0u; i < CONE_START_STEPS; i = i + 1u) {
        if t > span.exit {
            break;
        }
        let point = origin + dir * t;
        let d = march_sample(is_world, family_override, point, false).d;
        let eps = footprint(0.0, slope, t, point);
        let step = min((0.5 * d - cone * t - eps) / (1.0 + cone), step_cap);
        if step <= 0.5 * eps {
            break;
        }
        t = t + step;
    }
    return t;
}

// How a march ended (fractal-kernels `Hit` / `Outcome`): `hit` (the surface within its
// footprint), `unresolved` (the step budget ran out inside the interval: `point` / `ratio` are
// the closest sample, not a surface point), or neither (a miss). `first`: a hit on the march's
// very first sample, the case the leave restart retries.
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

// A hit (or closest-sample) record of sample `s` at `point` with footprint `eps`.
fn march_hit(point: vec3<f32>, s: SceneSample, eps: f32) -> MarchResult {
    return MarchResult(true, false, point, s.trap, eps, s.d / eps, false, s.object);
}

// A shadow ray's march: only whether the segment toward the light is blocked (fractal-kernels
// `march_visibility`). The steps, footprint and step cap of a secondary [`march`], no traps, no
// closest sample, no refinement; unresolved when the budget ran out inside the interval after
// at least one sample.
fn march_visibility(
    is_world: bool,
    family_override: u32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    base_in: f32,
    slope_in: f32,
) -> MarchResult {
    let max_steps = u32(param(P_SECONDARY_STEPS));
    let eps_scale = param(P_SECONDARY_EPS);
    let base = base_in * eps_scale;
    let slope = slope_in * eps_scale;
    let span = march_span(is_world, origin, dir);
    if !span.ok {
        return march_miss(origin);
    }
    let step_cap = 2.0 * (span.exit - span.enter) / STEP_CAP_STEPS;
    let step_factor = param(P_STEP_FACTOR);
    var t = span.enter;
    var sampled = false;
    for (var i = 0u; i < max_steps; i = i + 1u) {
        if t > span.exit {
            break;
        }
        let point = origin + dir * t;
        let d = march_sample(is_world, family_override, point, false).d;
        let eps = footprint(base, slope, t, point);
        if d < eps {
            return MarchResult(true, false, point, TRAP_START, eps, 0.0, i == 0u, 0u);
        }
        sampled = true;
        t = t + min(max(step_factor * d, eps * 0.5), step_cap);
    }
    if sampled && t <= span.exit {
        return MarchResult(false, true, origin, TRAP_START, 0.0, 0.0, false, 0u);
    }
    return march_miss(origin);
}

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
    retreat: vec3<f32>,
}

// The ray leaving the vertex at `point` with geometric normal `geo_n` (facing the arriving ray
// `dir_in`) along `dir_out` (fractal-kernels `leave_ray`): `eps` off the surface on the side
// `dir_out` leaves on; a ray on the normal's side restarts `eps` back along `dir_in`, a ray into
// a medium has no restart point (its origin again).
fn leave_ray(
    point: vec3<f32>,
    geo_n: vec3<f32>,
    dir_in: vec3<f32>,
    dir_out: vec3<f32>,
    eps: f32,
) -> Ray {
    if dot(dir_out, geo_n) < 0.0 {
        let origin = point - geo_n * eps;
        return Ray(origin, dir_out, origin);
    }
    return Ray(point + geo_n * eps, dir_out, point - dir_in * eps);
}

// [`march_visibility`] of a shadow ray leaving a surface, marched again from `retreat` once when
// it starts in a hit (fractal-kernels `visibility_leaving`).
fn visibility_leaving(
    is_world: bool,
    family_override: u32,
    origin: vec3<f32>,
    retreat: vec3<f32>,
    dir: vec3<f32>,
    base: f32,
    slope: f32,
) -> MarchResult {
    let m = march_visibility(is_world, family_override, origin, dir, base, slope);
    if any(retreat != origin) && m.hit && m.first {
        return march_visibility(is_world, family_override, retreat, dir, base, slope);
    }
    return m;
}

// Sphere-trace from `origin` along `dir` (fractal-kernels `march`). `kind`: a primary (camera or
// Direct) ray hits at `d <= eps` with `P_MAX_STEPS`, a secondary one at `d < eps` with
// `P_SECONDARY_STEPS` and its footprint times `P_SECONDARY_EPS`. `start` is a camera ray's pixel
// cone start ([`cone_start`], 0 otherwise). A sample inside the set before any outside one
// restarts at `t_enter`; a hit after stepping inside bisects the bracket from the last outside
// sample; a hit after a fractional step longer than the plain one bisects for the first hit.
// The step budget running out inside the interval reports the closest sample as unresolved.
fn march(
    is_world: bool,
    family_override: u32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    kind: u32,
    base_in: f32,
    slope_in: f32,
    start: f32,
) -> MarchResult {
    let primary = kind == RAY_PRIMARY || kind == RAY_DIRECT;
    let max_steps = u32(select(param(P_SECONDARY_STEPS), param(P_MAX_STEPS), primary));
    let eps_scale = select(param(P_SECONDARY_EPS), 1.0, primary);
    let base = base_in * eps_scale;
    let slope = slope_in * eps_scale;
    let span = march_span(is_world, origin, dir);
    if !span.ok {
        return march_miss(origin);
    }
    let t_enter = span.enter;
    let max_distance = span.exit;
    let step_cap = 2.0 * (max_distance - t_enter) / STEP_CAP_STEPS;
    let step_factor = param(P_STEP_FACTOR);
    let fractional = has_fractional_bulb(is_world, family_override);
    let ray_speed = length(dir);
    let epsilon_slope = max(abs(slope), FOOTPRINT_RELATIVE_FLOOR * ray_speed);
    var t = max(t_enter, start);
    // The last sample outside the surface and its ray length (the refinement bracket).
    var outside = march_miss(origin);
    var outside_t = 0.0;
    var has_outside = false;
    var closest = march_miss(origin);
    var has_closest = false;
    var accelerated = false;
    for (var i = 0u; i < max_steps; i = i + 1u) {
        if t > max_distance {
            break;
        }
        let point = origin + dir * t;
        let s = march_sample(is_world, family_override, point, true);
        let d = s.d;
        if d <= 0.0 && !has_outside && t > t_enter {
            t = t_enter;
            continue;
        }
        let eps = footprint(base, slope, t, point);
        if select(d < eps, d <= eps, primary) {
            if accelerated && has_outside {
                var lo = outside_t;
                var hi = t;
                var best = march_hit(point, s, eps);
                for (var k = 0u; k < HIT_REFINEMENTS; k = k + 1u) {
                    if hi - lo <= 0.5 * best.eps {
                        break;
                    }
                    let mid = 0.5 * (lo + hi);
                    let mid_point = origin + dir * mid;
                    let ms = march_sample(is_world, family_override, mid_point, true);
                    let mid_eps = footprint(base, slope, mid, mid_point);
                    if select(ms.d < mid_eps, ms.d <= mid_eps, primary) {
                        hi = mid;
                        best = march_hit(mid_point, ms, mid_eps);
                    } else {
                        lo = mid;
                    }
                }
                return best;
            }
            if !(has_outside && d <= 0.0) {
                var m = march_hit(point, s, eps);
                m.first = i == 0u;
                return m;
            }
            // Stepped inside: bisect the bracket from the last outside sample.
            var best = outside;
            var lo = outside_t;
            var hi = t;
            for (var k = 0u; k < HIT_REFINEMENTS; k = k + 1u) {
                if hi - lo <= 0.5 * best.eps {
                    break;
                }
                let mid = 0.5 * (lo + hi);
                let inner = origin + dir * mid;
                let ms = march_sample(is_world, family_override, inner, true);
                if ms.d > 0.0 {
                    lo = mid;
                    best = march_hit(inner, ms, footprint(base, slope, mid, inner));
                } else {
                    hi = mid;
                }
            }
            return best;
        }
        let here = march_hit(point, s, eps);
        if !has_closest || here.ratio < closest.ratio {
            closest = here;
            has_closest = true;
        }
        outside = here;
        outside_t = t;
        has_outside = true;
        if fractional {
            let step_distance = fractional_march_distance(
                is_world, family_override, point, eps, epsilon_slope, ray_speed, s.object, d,
            );
            let advance = min(max(step_factor * step_distance, eps * 0.5), step_cap);
            let plain = min(max(step_factor * d, eps * 0.5), step_cap);
            accelerated = advance > plain;
            t = t + advance;
        } else {
            t = t + min(max(step_factor * d, eps * 0.5), step_cap);
        }
    }
    if has_closest && t <= max_distance {
        closest.hit = false;
        closest.unresolved = true;
        return closest;
    }
    return march_miss(origin);
}

// [`march`] of a ray that left a surface at `origin`, marched again from `retreat` once when it
// starts in a hit and has a restart point (fractal-kernels `march_leaving`).
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
    if any(retreat != origin) && m.hit && m.first {
        return march(is_world, family_override, retreat, dir, kind, base, slope, start);
    }
    return m;
}

// The exit of the occupied dielectric `obj` along `dir` from `origin`, on its own field, not the
// union (fractal-kernels `march_exit` with fractal-core `transmission::exit_distance`): probes
// over the object's transformed local sphere, `P_GLASS_PROBES` of them at least, a signed field
// stepping its distance, then 24 bisections of the first inside-to-outside bracket. `eps` (the
// entrance's footprint) is the resolved boundary of an unsigned field. No exit (probes spent,
// or a non-finite estimate) is unresolved: the integrator absorbs the path.
fn march_exit(
    is_world: bool,
    obj: u32,
    family_override: u32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    eps: f32,
) -> MarchResult {
    let unresolved = MarchResult(false, true, origin, TRAP_START, eps, 0.0, false, obj);
    let q = object_point(is_world, obj, origin);
    let v = object_point(is_world, obj, origin + dir) - q;
    let radius = select(
        param(P_CLIP_RADIUS) / param(P_OBJ_SCALE),
        objects[obj * OBJECT_STRIDE + O_CLIP_RADIUS],
        is_world,
    );
    let a = dot(v, v);
    let b = dot(q, v);
    let disc = b * b - a * (dot(q, q) - radius * radius);
    var bound = 2.0 * eps;
    if disc > 0.0 && a > 0.0 {
        bound = max((-b + sqrt(disc)) / a, 0.0) + 2.0 * eps;
    }
    let extent = min(bound, param(P_MAX_DISTANCE));
    let steps = u32(pr(is_world, obj, P_GLASS_PROBES));
    if !(extent > 0.0 && eps > 0.0) || steps == 0u {
        return unresolved;
    }
    let signed = family_signed(resolved_family(is_world, obj, family_override));
    let step_factor = pr(is_world, obj, P_STEP_FACTOR);
    let threshold = select(eps, 0.0, signed);
    let minimum_step = max(extent / f32(steps), eps * 0.5);
    var t = 0.0;
    var previous = 0.0;
    for (var i = 0u; i <= steps; i = i + 1u) {
        let d = clipped_distance_and_trap(is_world, obj, family_override, origin + dir * t, 0u).x;
        if !finite(d) {
            return unresolved;
        }
        if d > threshold {
            var exit_t = 0.0;
            if i != 0u {
                var lo = previous;
                var hi = t;
                for (var k = 0u; k < 24u; k = k + 1u) {
                    if hi - lo <= eps * 0.25 {
                        break;
                    }
                    let mid = (lo + hi) * 0.5;
                    let value = clipped_distance_and_trap(is_world, obj, family_override, origin + dir * mid, 0u).x;
                    if !finite(value) {
                        return unresolved;
                    }
                    if value > threshold {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                exit_t = (lo + hi) * 0.5;
            }
            let point = origin + dir * exit_t;
            let mode = u32(pr(is_world, obj, P_COLOR_MODE));
            let trap = clipped_distance_and_trap(is_world, obj, family_override, point, mode).y;
            return MarchResult(true, false, point, trap, eps, 0.0, false, obj);
        }
        if t >= extent {
            break;
        }
        previous = t;
        let advance = select(minimum_step, max(-d * step_factor, minimum_step), signed);
        t = min(t + advance, extent);
    }
    return unresolved;
}

// Step halvings of the normal stencils (fractal-kernels `NORMAL_STEP_HALVINGS`).
const NORMAL_STEP_HALVINGS: u32 = 3u;

// The normalized central-difference gradient of object `obj`'s estimate around `center`, step
// `h` halved until every probe of an unsigned family lies outside, else `fallback`: the probe
// pattern of fractal-kernels `endpoint_normal` and Direct `direct_stencil`, which differ in what
// they accept. The path tracer (`direct` false) normalizes by the largest component first, so
// tiny but nonzero gradients keep their direction, and halves on until every component is
// finite and one nonzero. Direct takes the first qualifying step, its gradient when at least
// 1e-7 long, else the fallback.
fn stencil_normal(
    is_world: bool,
    obj: u32,
    family_override: u32,
    center: vec3<f32>,
    h_in: f32,
    fallback: vec3<f32>,
    direct: bool,
) -> vec3<f32> {
    let signed = family_signed(resolved_family(is_world, obj, family_override));
    var h = h_in;
    for (var k = 0u; k <= NORMAL_STEP_HALVINGS; k = k + 1u) {
        let xp = clipped_distance_and_trap(is_world, obj, family_override, center + vec3<f32>(h, 0.0, 0.0), 0u).x;
        let xm = clipped_distance_and_trap(is_world, obj, family_override, center - vec3<f32>(h, 0.0, 0.0), 0u).x;
        let yp = clipped_distance_and_trap(is_world, obj, family_override, center + vec3<f32>(0.0, h, 0.0), 0u).x;
        let ym = clipped_distance_and_trap(is_world, obj, family_override, center - vec3<f32>(0.0, h, 0.0), 0u).x;
        let zp = clipped_distance_and_trap(is_world, obj, family_override, center + vec3<f32>(0.0, 0.0, h), 0u).x;
        let zm = clipped_distance_and_trap(is_world, obj, family_override, center - vec3<f32>(0.0, 0.0, h), 0u).x;
        if signed || (xp > 0.0 && xm > 0.0 && yp > 0.0 && ym > 0.0 && zp > 0.0 && zm > 0.0) {
            let g = vec3<f32>(xp - xm, yp - ym, zp - zm);
            if direct {
                if length(g) >= 1.0e-7 {
                    return normalize(g);
                }
                return fallback;
            }
            let scale = max(abs(g.x), max(abs(g.y), abs(g.z)));
            // Every component finite: `max` ignores a NaN, so `scale` alone would pass one on.
            if scale > 0.0 && all(abs(g) <= vec3<f32>(TRAP_START)) {
                return normalize(g / scale);
            }
        }
        h = 0.5 * h;
    }
    return fallback;
}

// The surface normal at hit `x` of object `obj` (fractal-kernels `endpoint_normal`): a
// quarter-footprint stencil centred on the hit for a signed family, one footprint back along
// `dir` for an unsigned one, `-dir` when no step qualifies.
fn endpoint_normal(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    dir: vec3<f32>,
    eps: f32,
) -> vec3<f32> {
    let signed = family_signed(resolved_family(is_world, obj, family_override));
    let center = select(x - dir * eps, x, signed);
    return stencil_normal(is_world, obj, family_override, center, 0.25 * eps, -dir, false);
}

// The ray that found a hit (fractal-kernels `NormalRay`), marched again for each endpoint of a
// fractional iteration count: its origin, restart point, direction, march kind, footprint base
// and slope, the pixel's centre ray when its start is a cone start (`cone`), and the medium it
// travelled in (`inside`, the entrance footprint `medium_eps`).
struct NormalRay {
    origin: vec3<f32>,
    retreat: vec3<f32>,
    dir: vec3<f32>,
    kind: u32,
    base: f32,
    slope: f32,
    cone: bool,
    centre: vec3<f32>,
    inside: bool,
    medium_eps: f32,
}

// The normal of `ray`'s hit with object `obj` running exactly `iterations` iterations
// (fractal-kernels `endpoint_ray_normal`): the ray is marched again in that field, so each
// endpoint normal belongs to its own surface, oriented against the ray; `-dir` on a miss.
fn endpoint_ray_normal(
    is_world: bool,
    obj: u32,
    family_override: u32,
    ray: NormalRay,
    iterations: u32,
) -> vec3<f32> {
    normal_iterations = NormalIterations(true, obj, f32(iterations));
    var hit: MarchResult;
    if ray.inside {
        hit = march_exit(is_world, obj, family_override, ray.origin, ray.dir, ray.medium_eps);
    } else {
        var start = 0.0;
        if ray.cone {
            start = cone_start(is_world, family_override, ray.origin, ray.centre);
        }
        hit = march_leaving(
            is_world,
            family_override,
            ray.origin,
            ray.retreat,
            ray.dir,
            ray.kind,
            ray.base,
            ray.slope,
            start,
        );
    }
    var n = -ray.dir;
    if hit.hit {
        n = endpoint_normal(
            is_world,
            hit.object,
            family_override,
            hit.point,
            select(ray.dir, -ray.dir, ray.inside),
            hit.eps,
        );
        n = select(n, -n, dot(n, ray.dir) > 0.0);
    }
    normal_iterations.on = false;
    return n;
}

// The shading normal of a hit of object `obj` at `x` (fractal-kernels `surface_normal`), its
// stencil looking along `dir` (`-rd` from inside a medium, so it looks from outside). An
// integer `P_ITERATIONS` takes one endpoint stencil. A fractional count n = whole + weight
// re-marches `ray` with `whole` and `whole + 1` iterations and turns the lower endpoint's normal
// toward the upper one along their great circle by `weight`: normalizing the blended field's
// gradient instead jumps where one endpoint's gradient vanishes. WGSL only: a family that does
// not iterate (`ITERATING_MASK`, extended families CUDA has none of) has one field for every
// count, so it takes the one stencil.
fn surface_normal(
    is_world: bool,
    obj: u32,
    family_override: u32,
    x: vec3<f32>,
    dir: vec3<f32>,
    eps: f32,
    ray: NormalRay,
) -> vec3<f32> {
    let n = object_iterations(is_world, obj);
    let whole = u32(n);
    let weight = n - f32(whole);
    if weight == 0.0 || !family_iterates(resolved_family(is_world, obj, family_override)) {
        return endpoint_normal(is_world, obj, family_override, x, dir, eps);
    }
    let lower = endpoint_ray_normal(is_world, obj, family_override, ray, whole);
    let upper = endpoint_ray_normal(is_world, obj, family_override, ray, whole + 1u);
    let cosine = clamp(dot(lower, upper), -1.0, 1.0);
    var tangent = upper - lower * cosine;
    let sine = length(tangent);
    if sine > 0.0 {
        tangent = tangent * (1.0 / sine);
    } else {
        // Exactly parallel endpoints turn by zero; exactly opposite ones need a deterministic
        // great circle.
        let axis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(lower.x) < 0.8);
        tangent = normalize(axis - lower * dot(axis, lower));
    }
    let turn = weight * atan2(sine, cosine);
    return normalize(lower * cos(turn) + tangent * sin(turn));
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

// The sky gradient toward `d`, horizon to zenith (fractal-kernels `sky_radiance`).
fn sky_radiance(d: vec3<f32>) -> vec3<f32> {
    let t = clamp(0.5 + 0.5 * d.y, 0.0, 1.0);
    let horizon = param3(P_SKY_HORIZON);
    return (horizon + (param3(P_SKY_ZENITH) - horizon) * t) * param(P_SKY_INTENSITY);
}

// What a camera miss sees (fractal-kernels `background`): the HDRI or the sky gradient, never a
// sun disc. Whether it is shown is the pixel's decision (`P_BACKGROUND`, `trace_pixel`).
fn background(d: vec3<f32>) -> vec3<f32> {
    if env_on() {
        return env_map_radiance(d);
    }
    return sky_radiance(d);
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
    let t = 0.5 + n.y / 3.0;
    let horizon = param3(P_SKY_HORIZON);
    return (horizon + (param3(P_SKY_ZENITH) - horizon) * t) * param(P_SKY_INTENSITY);
}

fn sky_specular(d: vec3<f32>, alpha: f32) -> vec3<f32> {
    if env_on() {
        return env_specular(d, alpha);
    }
    return sky_radiance(d);
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

// Light sampling: a 1:1 port of fractal-kernels' sun / sky / HDRI section (`sun_select`,
// `sun_contains`, `world_sun_*`, `world_sky_weight`, `env_radiance`, `*_env_pdf`, `env_sample`).

// The probability of sampling the single sun rather than the sky (pathtrace.rs SunSky::p_sun):
// the host's `P_ENV_SUN_SELECT` against an HDRI, else the gradient's share, in [0.1, 0.9].
fn sun_select() -> f32 {
    if env_on() {
        return param(P_ENV_SUN_SELECT);
    }
    let sun = luminance(param3(P_LIGHT_COLOR) * param(P_LIGHT_INTENSITY));
    let sky = luminance((param3(P_SKY_HORIZON) + param3(P_SKY_ZENITH)) * (0.5 * param(P_SKY_INTENSITY)));
    let total = sun + MX_PI * sky;
    var s = 0.5;
    if total > 0.0 {
        s = sun / total;
    }
    return clamp(s, 0.1, 0.9);
}

fn sun_contains(wi: vec3<f32>) -> bool {
    let d = wi - param3(P_LIGHT_DIR);
    return 0.5 * dot(d, d) <= param(P_SUN_ONE_MINUS_COS);
}

fn world_sun_contains(light: u32, wi: vec3<f32>) -> bool {
    let d = wi - lp3(true, light, P_LIGHT_DIR);
    return 0.5 * dot(d, d) <= lp(true, light, P_SUN_ONE_MINUS_COS);
}

fn world_sun_weight(light: u32) -> f32 {
    return max(luminance(lp3(true, light, P_LIGHT_COLOR) * lp(true, light, P_LIGHT_INTENSITY)), 0.0);
}

fn world_sky_weight() -> f32 {
    var mean: f32;
    if env_on() {
        mean = param(P_ENV_MEAN) * param(P_ENV_INTENSITY);
    } else {
        mean = luminance((param3(P_SKY_HORIZON) + param3(P_SKY_ZENITH)) * (0.5 * param(P_SKY_INTENSITY)));
    }
    return max(MX_PI * mean, 0.000001);
}

fn world_light_total() -> f32 {
    var total = world_sky_weight();
    let count = u32(param(P_LIGHT_COUNT));
    for (var i = 0u; i < count; i = i + 1u) {
        total = total + world_sun_weight(i);
    }
    return total;
}

// The radiance arriving along `dir` from the lights (pathtrace.rs SunSky::radiance): a world's
// background plus each sun disc containing `dir`; a single scene's HDRI plus its sun disc, or the
// sun disc alone, or the sky gradient.
fn env_radiance(is_world: bool, dir: vec3<f32>) -> vec3<f32> {
    if is_world {
        var radiance = background(dir);
        let count = u32(param(P_LIGHT_COUNT));
        for (var i = 0u; i < count; i = i + 1u) {
            if world_sun_contains(i, dir) {
                radiance = radiance
                    + lp3(true, i, P_LIGHT_COLOR) * (lp(true, i, P_LIGHT_INTENSITY) * lp(true, i, P_SUN_CONE_PDF));
            }
        }
        return radiance;
    }
    let sun = param3(P_LIGHT_COLOR) * (param(P_LIGHT_INTENSITY) * param(P_SUN_CONE_PDF));
    if env_on() {
        let map = env_map_radiance(dir);
        if sun_contains(dir) {
            return map + sun;
        }
        return map;
    }
    if sun_contains(dir) {
        return sun;
    }
    return sky_radiance(dir);
}

// The density of a world's [`env_sample`] toward `dir`, with its sky weight and light total.
fn world_env_pdf(dir: vec3<f32>, sky_weight: f32, total: f32) -> f32 {
    var sky_pdf = 0.07957747;
    if env_on() {
        sky_pdf = env_map_pdf(dir);
    }
    var pdf = sky_weight / total * sky_pdf;
    let count = u32(param(P_LIGHT_COUNT));
    for (var i = 0u; i < count; i = i + 1u) {
        if world_sun_contains(i, dir) {
            pdf = pdf + world_sun_weight(i) / total * lp(true, i, P_SUN_CONE_PDF);
        }
    }
    return pdf;
}

// The density of a single scene's [`env_sample`] toward `dir`, with its sun probability `s`.
fn single_env_pdf(dir: vec3<f32>, s: f32) -> f32 {
    var sky_pdf = 0.07957747;
    if env_on() {
        sky_pdf = env_map_pdf(dir);
    }
    var pdf = (1.0 - s) * sky_pdf;
    if sun_contains(dir) {
        pdf = pdf + s * param(P_SUN_CONE_PDF);
    }
    return pdf;
}

// pathtrace.rs SunSky::pdf: the sun-cone and sky mixture density of [`env_sample`].
fn env_pdf(is_world: bool, dir: vec3<f32>) -> f32 {
    if is_world {
        return world_env_pdf(dir, world_sky_weight(), world_light_total());
    }
    return single_env_pdf(dir, sun_select());
}

// A direction in the cone of half-angle cosine `1 - omc` about the sun direction `d`, at the
// cone fraction `u` (0 the axis) and azimuth fraction `v`.
fn sun_cone_dir(d: vec3<f32>, omc: f32, u: f32, v: f32) -> vec3<f32> {
    let tb = basis(d);
    let cos_t = 1.0 - u * omc;
    let sin_t = sqrt(max(1.0 - cos_t * cos_t, 0.0));
    let phi = 2.0 * MX_PI * v;
    return tb[0] * (sin_t * cos(phi)) + tb[1] * (sin_t * sin(phi)) + d * cos_t;
}

// A uniform direction on the sphere at fractions (`u` along z, `v` the azimuth).
fn sphere_dir(u: f32, v: f32) -> vec3<f32> {
    let z = 1.0 - 2.0 * u;
    let radius = sqrt(max(1.0 - z * z, 0.0));
    let phi = 2.0 * MX_PI * v;
    return vec3<f32>(radius * cos(phi), radius * sin(phi), z);
}

// One light direction and its density (pathtrace.rs SunSky::sample). A world picks a sun by its
// weight or the sky by `r1` (the remainder of the pick is the cone or sky fraction); a single
// scene samples its sun cone with probability [`sun_select`], else the HDRI's CDFs or the
// uniform sphere.
fn env_sample(is_world: bool, r1: f32, r2: f32) -> vec4<f32> {
    if is_world {
        let sky_weight = world_sky_weight();
        let total = world_light_total();
        let count = u32(param(P_LIGHT_COUNT));
        var selector = r1 * total;
        for (var i = 0u; i < count; i = i + 1u) {
            let weight = world_sun_weight(i);
            if selector < weight {
                let dir = sun_cone_dir(
                    lp3(true, i, P_LIGHT_DIR),
                    lp(true, i, P_SUN_ONE_MINUS_COS),
                    selector / weight,
                    r2,
                );
                return vec4<f32>(dir, world_env_pdf(dir, sky_weight, total));
            }
            selector = selector - weight;
        }
        let u = clamp(selector / sky_weight, 0.0, 0.99999994);
        var dir: vec3<f32>;
        if env_on() {
            dir = env_map_sample(u, r2).xyz;
        } else {
            dir = sphere_dir(u, r2);
        }
        return vec4<f32>(dir, world_env_pdf(dir, sky_weight, total));
    }
    let s = sun_select();
    var dir: vec3<f32>;
    if r1 < s {
        dir = sun_cone_dir(param3(P_LIGHT_DIR), param(P_SUN_ONE_MINUS_COS), r1 / s, r2);
    } else if env_on() {
        dir = env_map_sample((r1 - s) / (1.0 - s), r2).xyz;
    } else {
        dir = sphere_dir((r1 - s) / (1.0 - s), r2);
    }
    return vec4<f32>(dir, single_env_pdf(dir, s));
}

struct FastMaterial {
    albedo: vec3<f32>,
    f0: vec3<f32>,
    alpha: f32,
    diffuse_w: f32,
    spec_prob: f32,
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

// A hit's base colour, specular roughness and metalness: the palette `palette_rgb` (or the
// material's solid colour, `P_COLOR_SOURCE`) mixed toward the facing material at grazing angles
// of `view` to the normal `n` (usd-rs pt-material-ext facing mix; exponent 0 is off). The one
// source of both material models and of the Direct albedo guide, as in fractal-kernels.
struct Facing {
    color: vec3<f32>,
    roughness: f32,
    metalness: f32,
}

fn facing_mix(
    is_world: bool,
    obj: u32,
    palette_rgb: vec3<f32>,
    n: vec3<f32>,
    view: vec3<f32>,
) -> Facing {
    var color = palette_rgb;
    if u32(pr(is_world, obj, P_COLOR_SOURCE)) != 0u {
        color = pr3(is_world, obj, P_BASE_COLOR);
    }
    var roughness = pr(is_world, obj, P_SPECULAR_ROUGHNESS);
    var metalness = pr(is_world, obj, P_METALNESS);
    let facing_exp = pr(is_world, obj, P_FACING_EXPONENT);
    if facing_exp > 0.0 {
        let weight = pow(max(1.0 - abs(dot(n, view)), 0.0), facing_exp);
        let fc = pr3(is_world, obj, P_FACING_COLOR);
        let fr = pr(is_world, obj, P_FACING_ROUGHNESS);
        let fm = pr(is_world, obj, P_FACING_METALLIC);
        color = color + (fc - color) * weight;
        roughness = roughness + (fr - roughness) * weight;
        metalness = metalness + (fm - metalness) * weight;
    }
    return Facing(color, roughness, metalness);
}

fn fast_material(is_world: bool, obj: u32, f: Facing) -> FastMaterial {
    let roughness = f.roughness;
    let metalness = f.metalness;
    let base = (f.color * pr(is_world, obj, P_BASE)) * pr3(is_world, obj, P_BASE_TINT);
    let ior = pr(is_world, obj, P_SPECULAR_IOR);
    let r0 = (ior - 1.0) / (ior + 1.0);
    let f0d = r0 * r0 * pr(is_world, obj, P_SPECULAR);
    let f0 = (pr3(is_world, obj, P_SPECULAR_COLOR) * f0d) * (1.0 - metalness) + base * metalness;
    let rough = max(roughness, 0.03);
    let diffuse_w = 1.0 - metalness;
    let lf = luminance(f0);
    let spec_prob = fast_spec_probability(lf, diffuse_w * luminance(base), diffuse_w);
    return FastMaterial(base, f0, rough * rough, diffuse_w, spec_prob);
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

// The full model's inputs (fractal-kernels `surface_inputs`): the facing-mixed colour times
// `P_BASE_TINT` as base colour; the transmission tint applies at interfaces only when volume
// absorption is off.
fn surface_inputs(is_world: bool, obj: u32, f: Facing) -> SsInputs {
    let color = f.color;
    let roughness = f.roughness;
    let metalness = f.metalness;
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

// The pinhole camera direction through image point (`px` + `jx`, `py` + `jy`) of the frame
// (fractal-kernels `trace_pixel`). WGSL keeps the OFX tile frame (`frame_origin`, y up), which
// WarpBro never sets (`P_OFX` 0).
fn camera_dir(px: i32, py: i32, jx: f32, jy: f32) -> vec3<f32> {
    let fo = frame_origin();
    let fx = f32(px - fo.x) + jx;
    let fy = f32(py - fo.y) + jy;
    let nx = 2.0 * fx / param(P_WIDTH) - 1.0;
    let ny = select(1.0 - 2.0 * fy / param(P_HEIGHT), 2.0 * fy / param(P_HEIGHT) - 1.0, is_ofx());
    return normalize(
        param3(P_CAM_FORWARD)
            + (param3(P_CAM_RIGHT) * (nx * param(P_HALF_W)) + param3(P_CAM_UP) * (ny * param(P_HALF_H))),
    );
}

// One camera sample (fractal-kernels `PathSample`): its radiance, whether the camera ray hit,
// the primary guides (albedo, normal) and whether any march of the path ran out of steps.
struct TraceOut {
    radiance: vec3<f32>,
    hit: bool,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    unresolved: bool,
}

// The medium of a path outside every object (fractal-core `transmission::AIR`).
const AIR_MEDIUM: u32 = 0xffffffffu;

// Beer-Lambert transmittance after `distance` through a medium whose colour is the share left
// after `depth` (fractal-core `transmission::volume_transmittance`); depth 0 tints at the
// interfaces instead.
fn volume_transmittance(color: vec3<f32>, depth: f32, distance: f32) -> vec3<f32> {
    if depth <= 0.0 || distance <= 0.0 {
        return vec3<f32>(1.0);
    }
    return pow(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(distance / depth));
}

// The medium after a scatter at `object` (fractal-core `transmission::medium_after_scatter`):
// only a hemisphere crossing changes it, out of `object` into air or from air into it.
fn medium_after_scatter(current: u32, object: u32, crossed: bool) -> u32 {
    if !crossed {
        return current;
    }
    if current == object {
        return AIR_MEDIUM;
    }
    return object;
}

// The MIS power heuristic (beta 2) with fractal-kernels' fused denominator `fma(a, a, b * b)`.
fn power_heuristic(a: f32, b: f32) -> f32 {
    let denominator = fma(a, a, b * b);
    if denominator > 0.0 {
        return a * a / denominator;
    }
    return 0.0;
}

// The denoiser's albedo guide of a hit of object `obj` with base colour `color`: render-rs
// `SurfaceInputs::guide_albedo` (`ss_guide_albedo`) of the fields fractal-kernels fills for it,
// base * colour * base tint plus the emission, clamped to [0, 1], for both material models.
fn guide_albedo(is_world: bool, obj: u32, color: vec3<f32>) -> vec3<f32> {
    return clamp(
        color * pr3(is_world, obj, P_BASE_TINT) * pr(is_world, obj, P_BASE)
            + pr3(is_world, obj, P_EMISSION_COLOR) * pr(is_world, obj, P_EMISSION),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

// What one path vertex needs from its BSDF (fractal-kernels `Scatter`): `f cos` and the BSDF
// density toward the light sample (zero unless `light`), and the next direction `wi` with its
// weight `f cos / pdf`, density, validity, whether it came from a delta lobe and whether it
// refracts through the surface. The material phase answers both before the shadow ray marches.
struct Scatter {
    light_f: vec3<f32>,
    light_pdf: f32,
    wi: vec3<f32>,
    weight: vec3<f32>,
    pdf: f32,
    ok: bool,
    delta: bool,
    crossed: bool,
}

// The full (MaterialX Standard Surface) model's [`Scatter`]: layers and lobe weights derived once
// for the light evaluation, its density and the BSDF sample (fractal-kernels `full_scatter`).
fn full_scatter(
    inputs: SsInputs,
    frame: SsFrame,
    wo: vec3<f32>,
    ld: vec3<f32>,
    light: bool,
    u: vec3<f32>,
) -> Scatter {
    let layers = ss_layers(inputs, frame, wo);
    let weights = ss_lobe_weights_prepared(inputs, layers, wo);
    var light_f = vec3<f32>(0.0);
    var light_pdf = 0.0;
    if light {
        let lobes = ss_eval_light_prepared(inputs, frame, layers, wo, ld);
        light_f = lobes.base + lobes.specular + lobes.transmission;
        if max(light_f.x, max(light_f.y, light_f.z)) > 0.0 {
            light_pdf = ss_pdf_prepared(layers, wo, ld, weights);
        }
    }
    let s = ss_sample_prepared(inputs, frame, layers, wo, u, weights);
    return Scatter(
        light_f,
        light_pdf,
        s.wi,
        s.weight,
        s.pdf,
        s.valid,
        s.delta,
        inputs.transmission > 0.0 && dot(s.wi, frame.n) < 0.0,
    );
}

// The fast model's [`Scatter`]: Lambert + one GGX lobe, never delta, never transmissive
// (fractal-kernels `fast_scatter`).
fn fast_scatter(
    m: FastMaterial,
    n: vec3<f32>,
    wo: vec3<f32>,
    ld: vec3<f32>,
    light: bool,
    u: vec3<f32>,
) -> Scatter {
    var light_f = vec3<f32>(0.0);
    var light_pdf = 0.0;
    if light {
        let ev = fast_eval(m, n, wo, ld);
        light_f = ev.xyz;
        light_pdf = ev.w;
    }
    let s = fast_sample(m, n, wo, u);
    return Scatter(light_f, light_pdf, s.wi, s.throughput, s.pdf, s.valid, false, false);
}

// One camera path from `origin` along `dir0` (fractal-kernels `trace_path`, a 1:1 port). The
// camera ray marches from `start`; `cone` / `centre` is the pixel's centre ray when that start is
// a cone start (the fractional-normal re-march repeats it). `mat_spec` picks the material model
// (`MAT_DISPATCH`: each object's own). Per vertex: emission, Russian roulette after the first
// bounce, ONE light sample drawn before the BSDF sample (r1, r2, then u: a point sun of a single
// scene, else `env_sample`), the BSDF's `Scatter`, then the shadow ray. An unresolved camera ray
// shows the background, an unresolved bounce ends the path, an interior exit not found absorbs
// it; the final BSDF ray still collects its environment MIS share before the depth limit ends
// the path.
fn trace_path(
    is_world: bool,
    family_override: u32,
    mat_spec: u32,
    origin: vec3<f32>,
    dir0: vec3<f32>,
    start: f32,
    cone: bool,
    centre: vec3<f32>,
    rng: ptr<function, Rng>,
) -> TraceOut {
    let cam = param3(P_CAM_ORIGIN);
    let slope = param(P_FOOTPRINT);
    let max_bounces = u32(param(P_MAX_BOUNCES));

    var ro = origin;
    // The restart point of the ray `ro`, `rd` (`leave_ray`); a camera ray has none.
    var rr = origin;
    var rd = dir0;
    var throughput = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    var mis_bsdf_pdf = 0.0;
    var mis_env = false;
    var primary_hit = false;
    var unresolved = false;
    var primary_albedo = vec3<f32>(0.0);
    var primary_normal = vec3<f32>(0.0);
    var bounce = 0u;
    var medium = AIR_MEDIUM;
    var medium_eps = 0.0;
    // The last exact interface point includes the numerical ray offset in absorption.
    var interface_point = origin;
    loop {
        let base = select(slope * length(ro - cam), 0.0, bounce == 0u);
        let kind = select(RAY_BOUNCE, RAY_PRIMARY, bounce == 0u);
        let inside = medium != AIR_MEDIUM;
        var m: MarchResult;
        if inside {
            m = march_exit(is_world, medium, family_override, ro, rd, medium_eps);
        } else {
            m = march_leaving(
                is_world,
                family_override,
                ro,
                rr,
                rd,
                kind,
                base,
                slope,
                select(0.0, start, bounce == 0u),
            );
        }
        if m.unresolved {
            unresolved = true;
            if bounce > 0u {
                break;
            }
        }
        if inside && !m.hit {
            break;
        }
        if !m.hit {
            // A missed camera ray shows the sun disc only when the host asks for it
            // (`P_CAMERA_SUN`). It has no competing strategy: weight one.
            if bounce == 0u && param(P_CAMERA_SUN) == 0.0 {
                let seen = background(rd);
                primary_albedo = clamp(seen, vec3<f32>(0.0), vec3<f32>(1.0));
                radiance = radiance + throughput * seen;
                break;
            }
            let env = env_radiance(is_world, rd);
            if bounce == 0u {
                primary_albedo = clamp(env, vec3<f32>(0.0), vec3<f32>(1.0));
            }
            var w = 1.0;
            if mis_env {
                w = power_heuristic(mis_bsdf_pdf, env_pdf(is_world, rd));
            }
            radiance = radiance + throughput * env * w;
            break;
        }
        // fractal-core `allows_surface_vertex`: after the miss, so the last BSDF ray still
        // collects the environment.
        if bounce > max_bounces {
            break;
        }
        if bounce == 0u {
            primary_hit = true;
        }
        let obj = m.object;
        let full = mat_spec == MAT_FULL
            || (mat_spec == MAT_DISPATCH && pr(is_world, obj, P_MATERIAL_MODEL) != 0.0);
        if inside {
            throughput = throughput * volume_transmittance(
                pr3(is_world, obj, P_TRANSMISSION_COLOR),
                pr(is_world, obj, P_TRANSMISSION_DEPTH),
                length(m.point - interface_point),
            );
        }
        let ray = NormalRay(
            ro,
            rr,
            rd,
            kind,
            base,
            slope,
            cone && bounce == 0u,
            centre,
            inside,
            medium_eps,
        );
        let n = surface_normal(is_world, obj, family_override, m.point, select(rd, -rd, inside), m.eps, ray);
        let geo_n = select(n, -n, dot(n, rd) > 0.0);
        let eps = 4.0 * m.eps;
        let wo = -rd;
        let f = facing_mix(is_world, obj, hit_palette(is_world, obj, m.point, n, m.trap), geo_n, wo);
        if bounce == 0u {
            primary_albedo = guide_albedo(is_world, obj, f.color);
            primary_normal = geo_n;
        }

        // Emission, then Russian roulette: both read only this vertex and the path.
        let frame = SsFrame(geo_n, object_tangent_y(is_world, obj), inside, 0.0);
        var inputs = SsInputs();
        if full {
            inputs = surface_inputs(is_world, obj, f);
            radiance = radiance + throughput * ss_eval_emission(inputs, frame, wo);
        } else if pr(is_world, obj, P_EMISSION) > 0.0 {
            radiance = radiance
                + throughput * (pr3(is_world, obj, P_EMISSION_COLOR) * pr(is_world, obj, P_EMISSION));
        }
        if bounce > 0u {
            let p_continue = min(max(throughput.x, max(throughput.y, throughput.z)), 1.0);
            if rand(rng) >= p_continue {
                break;
            }
            throughput = throughput * (1.0 / p_continue);
        }

        // One light sample (r1, r2), then the BSDF sample (u). A point sun of a single scene
        // (no cone, unless an HDRI whose sun carries no light) is a delta light: no MIS, and NEE
        // never samples the sky then (escaping BSDF rays see it at weight one).
        let r1 = rand(rng);
        let r2 = rand(rng);
        let point_sun = !is_world
            && param(P_SUN_CONE_PDF) <= 0.0
            && !(env_on() && param(P_ENV_SUN_SELECT) == 0.0);
        var ld = vec3<f32>(0.0);
        var lpdf = 0.0;
        if !inside {
            if point_sun {
                ld = param3(P_LIGHT_DIR);
                lpdf = 1.0;
            } else {
                let ls = env_sample(is_world, r1, r2);
                ld = ls.xyz;
                lpdf = ls.w;
            }
        }
        let light = lpdf > 0.0 && !inside && dot(ld, geo_n) > 0.0;
        let u = vec3<f32>(rand(rng), rand(rng), rand(rng));
        var scatter: Scatter;
        if full {
            scatter = full_scatter(inputs, frame, wo, ld, light, u);
        } else {
            scatter = fast_scatter(fast_material(is_world, obj, f), geo_n, wo, ld, light, u);
        }
        // NEE, MIS-weighted against BSDF sampling; an unresolved shadow ray counts as blocked.
        if light && max(scatter.light_f.x, max(scatter.light_f.y, scatter.light_f.z)) > 0.0 {
            let shadow_ray = leave_ray(m.point, geo_n, rd, ld, eps);
            let shadow_base = slope * length(shadow_ray.origin - cam);
            let shadow = visibility_leaving(
                is_world,
                family_override,
                shadow_ray.origin,
                shadow_ray.retreat,
                ld,
                shadow_base,
                0.0,
            );
            if shadow.unresolved {
                unresolved = true;
            }
            if !shadow.hit && !shadow.unresolved {
                var li: vec3<f32>;
                if point_sun {
                    li = param3(P_LIGHT_COLOR) * param(P_LIGHT_INTENSITY);
                } else {
                    li = env_radiance(is_world, ld) * (power_heuristic(lpdf, scatter.light_pdf) / lpdf);
                }
                radiance = radiance + throughput * (scatter.light_f * li);
            }
        }

        if !scatter.ok {
            break;
        }
        throughput = throughput * scatter.weight;
        mis_bsdf_pdf = scatter.pdf;
        // NEE never connects through an interface, nor samples the sky beside a point sun:
        // those escapes keep weight one.
        mis_env = !scatter.delta && !scatter.crossed && !point_sun;
        medium = medium_after_scatter(medium, obj, scatter.crossed);
        if scatter.crossed && medium != AIR_MEDIUM {
            medium_eps = m.eps;
        }
        interface_point = m.point;
        let next = leave_ray(m.point, geo_n, rd, scatter.wi, eps);
        ro = next.origin;
        rr = next.retreat;
        rd = scatter.wi;
        bounce = bounce + 1u;
    }
    return TraceOut(radiance, primary_hit, primary_albedo, primary_normal, unresolved);
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
    // A world's shadow ray ends where it leaves the bounding sphere; a single object's marches
    // to `P_MAX_DISTANCE` (fractal-kernels Direct `direct_shadow`).
    var limit = max_distance;
    if is_world {
        let span = march_span(true, origin, l);
        if !span.ok {
            return 1.0;
        }
        limit = span.exit;
    }
    let sharpness = select(0.0, 1.0 / tan(angle), angle > 0.0);
    var visibility = 1.0;
    var i = 0u;
    while i < steps && t <= limit {
        let d = march_sample(is_world, family_override, origin + l * t, false).d;
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
            let d = march_sample(is_world, family_override, point + directions[j] * t, false).d;
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
    material: Facing,
    eps: f32,
    variance: f32,
) -> vec3<f32> {
    let facing = select(-n, n, dot(n, view) > 0.0);
    let frame = SsFrame(facing, object_tangent_y(is_world, obj), false, 0.0);
    var inputs = surface_inputs(is_world, obj, material);
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
        transmission_radiance = sky_specular(trans_dir, trans_alpha) * spec_ao;
    }
    let environment = SsEnvironment(
        sky_irradiance(facing) * ao,
        sky_specular(specular_dir, specular_alpha) * spec_ao,
        sky_specular(coat_dir, coat_alpha) * coat_ao,
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

// One Direct sample of the camera ray `origin`, `dir` (fractal-kernels Direct `trace_direct`):
// one primary march (`RAY_DIRECT`); an unresolved ray's closest sample stands in for the surface
// when its screen-space error is under half a pixel (Keinert et al. 2014, 3.2), else the
// background.
fn trace_direct(
    is_world: bool,
    family_override: u32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    start: f32,
) -> TraceOut {
    let m = march(is_world, family_override, origin, dir, RAY_DIRECT, 0.0, param(P_FOOTPRINT), start);
    let unresolved = m.unresolved;
    if !(m.hit || (unresolved && m.ratio <= 0.5 * param(P_SAMPLE_CONE))) {
        return TraceOut(background(dir), false, vec3<f32>(0.0), vec3<f32>(0.0), unresolved);
    }
    let family = resolved_family(is_world, m.object, family_override);
    let center = select(m.point - dir * m.eps, m.point, family_signed(family));
    let fine = stencil_normal(is_world, m.object, family_override, center, 0.5 * m.eps, -dir, true);
    let cone = param(P_SAMPLE_CONE);
    var n = fine;
    if cone > 1.0 {
        n = stencil_normal(
            is_world,
            m.object,
            family_override,
            m.point + fine * (m.eps * cone),
            0.5 * m.eps * cone,
            fine,
            true,
        );
    }
    let variance = direct_gsaa_variance(n, fine);
    let view = -dir;
    let pal = hit_palette(is_world, m.object, m.point, n, m.trap);
    let facing = select(-n, n, dot(n, view) > 0.0);
    let material = facing_mix(is_world, m.object, pal, facing, view);
    let radiance = direct_shade(
        is_world,
        m.object,
        family_override,
        m.point,
        n,
        view,
        material,
        m.eps,
        variance,
    );
    // The albedo guide is the facing-mixed colour (fractal-kernels Direct `trace_direct`).
    return TraceOut(radiance, true, material.color, n, unresolved);
}

// Trace `P_SPP` samples of the pixel of film index `idx` (8x4 tiles) and add them to its
// accumulators (fractal-kernels `trace_pixel`): radiance (rgb sum, sample count), the albedo and
// normal guides, and stats (sum of squared luminance, unresolved samples, the camera-ray start,
// 0). The start is the pixel's cone start, found by the launch of sample 0 and kept in
// `stats.z`; a lens ray or a single object's Direct sample starts at 0. Sample `s` draws its
// jitter (x, y), then a lens point (angle, radius) when the aperture is open, then the path's
// numbers; a Direct sample takes the centre of stratum `s` of the `P_DIRECT_GRID` grid. A
// sample whose camera ray missed adds its radiance only when the background is shown.
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

    let begin = u32(param(P_SAMPLE_BEGIN));
    let spp = u32(param(P_SPP));
    let ofx = is_ofx();
    let origin = param3(P_CAM_ORIGIN);
    let fwd = param3(P_CAM_FORWARD);
    let right = param3(P_CAM_RIGHT);
    let up = param3(P_CAM_UP);
    // OFX packs its frame origin into the lens slots (`frame_origin`): no lens there.
    let aperture = select(param(P_APERTURE), 0.0, ofx);
    let background_shown = param(P_BACKGROUND) != 0.0;
    let centre = camera_dir(px, py, 0.5, 0.5);
    let cone = !is_direct && !(aperture > 0.0);
    var start = 0.0;
    if (is_direct && !is_world) || aperture > 0.0 {
        start = 0.0;
    } else if begin != 0u {
        start = st.z;
    } else {
        start = cone_start(is_world, family_override, origin, centre);
    }

    let grid = u32(param(P_DIRECT_GRID));
    var sum = acc.xyz;
    var albedo_sum = alb.xyz;
    var normal_sum = nrm.xyz;
    var primary_hits = 0.0;
    var luma2 = 0.0;
    var unresolved = 0.0;
    for (var s = 0u; s < spp; s = s + 1u) {
        var rng = pixel_rng(px, py, begin + s);
        let index = begin + s;
        var jx: f32;
        var jy: f32;
        if is_direct {
            jx = (f32(index % grid) + 0.5) / f32(grid);
            jy = (f32(index / grid) + 0.5) / f32(grid);
        } else {
            jx = rand(&rng);
            jy = rand(&rng);
        }
        var ro = origin;
        var dir = camera_dir(px, py, jx, jy);
        if !is_direct && aperture > 0.0 {
            // Thin lens: the focus plane at `P_FOCUS_DISTANCE` along the view axis.
            let focus = origin + dir * (param(P_FOCUS_DISTANCE) / dot(dir, fwd));
            let angle = 2.0 * MX_PI * rand(&rng);
            let radius = aperture * sqrt(rand(&rng));
            ro = origin + (right * (radius * cos(angle)) + up * (radius * sin(angle)));
            dir = normalize(focus - ro);
        }
        var out: TraceOut;
        if is_direct {
            out = trace_direct(is_world, family_override, ro, dir, start);
        } else {
            out = trace_path(is_world, family_override, mat_spec, ro, dir, start, cone, centre, &rng);
        }
        if out.unresolved {
            unresolved = unresolved + 1.0;
        }
        primary_hits = primary_hits + select(select(0.0, sky_alpha(), ofx), 1.0, out.hit);
        albedo_sum = albedo_sum + out.albedo;
        normal_sum = normal_sum + out.normal;
        var l = out.radiance;
        if ofx {
            if !out.hit {
                l = l * sky_alpha();
            }
        } else if !(out.hit || background_shown) {
            l = vec3<f32>(0.0);
        }
        sum = sum + l;
        let y = luminance(l);
        luma2 = luma2 + y * y;
    }
    accum[i] = vec4<f32>(sum, acc.w + f32(spp));
    stats[i] = vec4<f32>(st.x + luma2, st.y + unresolved, start, 0.0);
    albedo[i] = vec4<f32>(albedo_sum, alb.w + f32(spp));
    normal_buf[i] = vec4<f32>(normal_sum, nrm.w + select(f32(spp), primary_hits, ofx));
}

var<workgroup> wg_tally: array<vec4<f32>, 128>;

// The sum of `v` over the 128 invocations of a workgroup (every invocation must call it).
fn workgroup_sum(tid: u32, v: vec4<f32>) -> vec4<f32> {
    wg_tally[tid] = v;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if tid < stride {
            wg_tally[tid] = wg_tally[tid] + wg_tally[tid + stride];
        }
        workgroupBarrier();
    }
    return wg_tally[0];
}

// First `tally` pass, `TALLY_THREADS / 128` workgroups (CUDA `tally`): invocation `t` adds
// `[unresolved, samples, active tiles, 0]` over film samples and tiles `t, t + TALLY_THREADS,
// ..` (coalesced), and each workgroup stores its sum in `tally_parts`.
@compute @workgroup_size(128, 1, 1)
fn tally(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    let first = wid.x * 128u + tid;
    let n = arrayLength(&accum);
    var unresolved = 0.0;
    var samples = 0.0;
    for (var k = first; k < n; k = k + TALLY_THREADS) {
        unresolved = unresolved + stats[k].y;
        samples = samples + accum[k].w;
    }
    let size = tile_size();
    let total_tiles = ((size.x + 7u) / 8u) * ((size.y + 3u) / 4u);
    var active_count = 0u;
    for (var t = first; t < total_tiles; t = t + TALLY_THREADS) {
        if active_rw[t] != 0u {
            active_count = active_count + 1u;
        }
    }
    let sum = workgroup_sum(tid, vec4<f32>(unresolved, samples, f32(active_count), 0.0));
    if tid == 0u {
        tally_parts[wid.x] = sum;
    }
}

// Second `tally` pass, one workgroup (the CUDA host's sum): the total of `tally_parts` into the
// 16-byte summary the host reads back.
@compute @workgroup_size(128, 1, 1)
fn tally_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var part = vec4<f32>(0.0);
    for (var k = tid; k < arrayLength(&tally_parts); k = k + 128u) {
        part = part + tally_parts[k];
    }
    let sum = workgroup_sum(tid, part);
    if tid == 0u {
        summary_out[0] = sum;
    }
}

@compute @workgroup_size(128, 1, 1)
fn adapt(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let tile = linear_id(gid, nwg);
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

// The scene-linear display value of row-major pixel `i` of a `width`-wide frame: the tiled
// accumulator's mean times `P_EXPOSURE`, its saturation scaled about its luminance. Shared by
// `tonemap` and `tonemap_display`.
fn display_raw(i: u32, width: u32) -> vec3<f32> {
    let x = i % width;
    let y = i / width;
    let tiles_x = (width + 7u) / 8u;
    let tile = (y / 4u) * tiles_x + (x / 8u);
    let a = accum[tile * 32u + (y % 4u) * 8u + (x % 8u)];
    let inv = select(0.0, param(P_EXPOSURE) / a.w, a.w > 0.0);
    let c = a.xyz * inv;
    let l = luminance(c);
    let sat = param(P_SATURATION);
    return l + (c - vec3<f32>(l)) * sat;
}

@compute @workgroup_size(128, 1, 1)
fn tonemap(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_id(gid, nwg);
    let size = tile_size();
    if i >= size.x * size.y {
        return;
    }
    aux_out[i] = vec4<f32>(display_raw(i, size.x), 1.0);
}

@compute @workgroup_size(128, 1, 1)
fn tonemap_display(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_id(gid, nwg);
    let size = tile_size();
    if i >= size.x * size.y {
        return;
    }
    var rgb = to_709_rgb(display_raw(i, size.x));
    if param(P_TONEMAP) != 0.0 {
        let m = max(rgb, vec3<f32>(0.0));
        rgb = m / (vec3<f32>(1.0) + m);
    }
    aux_out[i] = vec4<f32>(rgb, 1.0);
    sdr_out[i] = pack_srgb_rgba8(rgb);
}

// The pick ray of the query in `aux_out[0]` (a generated `pick` entry per module calls it with
// the module's family: `FAMILY_DYNAMIC` or one family code).
fn pick_ray(gid: vec3<u32>, family_override: u32) {
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
    let dir = normalize(fwd + (right * (q.x * hw) + up * (q.y * hh)));
    let start = cone_start(true, family_override, origin, dir);
    let m = march(true, family_override, origin, dir, RAY_PRIMARY, 0.0, param(P_FOOTPRINT), start);
    if m.hit {
        let t = length(m.point - origin);
        let depth = t * dot(dir, fwd);
        aux_out[0] = vec4<f32>(f32(m.object + 1u), t, depth, 0.0);
    } else {
        aux_out[0] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
