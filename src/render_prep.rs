//! Backend-independent scene preparation, world parameter packing, residency tracking, and tile untiling.

use std::sync::Arc;

use crate::palette::PaletteScheme;
use crate::params::*;
use crate::scene::{Bulb, Formula, Scene};

pub const BLOCK: u32 = 128;
/// Threads of the `tally` reduction (a multiple of `BLOCK`): CUDA reads back this many
/// `[f64; 2]` partial sums; WGSL sums them per workgroup and then in one more workgroup
/// (`tally` + `tally_sum`, `render_wgpu::WgpuShared::record_tally`).
pub const TALLY_THREADS: usize = 4096;

/// Slots that only the tonemap reads: changing them re-tonemaps without restarting the samples.
pub const TONEMAP_ONLY: [usize; 3] = [P_EXPOSURE, P_SATURATION, P_TONEMAP];
/// Slots that change every launch.
pub const PER_LAUNCH: [usize; 3] = [P_SAMPLE_BEGIN, P_SPP, P_SEED];
/// Adaptive-sampling slots: they choose which tiles keep sampling, never the accumulated values,
/// so changing them must not restart the render (the `adapt` kernel re-decides every step).
pub const SCHEDULE_ONLY: [usize; 2] = [P_ADAPT_THRESHOLD, P_ADAPT_MIN];

/// Samples of a film in the kernels' 8x4 tile layout: every tile is whole, so partial tiles at
/// the right and bottom edges are padded. Both backends size their films with it.
pub(crate) fn padded_len(width: usize, height: usize) -> usize {
    width.div_ceil(8) * height.div_ceil(4) * 32
}

/// A film plane both backends accumulate (`Gpu::read_film`): the radiance sums with the sample
/// count in `w`, and the denoiser's albedo and normal guide sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Plane {
    Accum,
    Albedo,
    Normal,
}

/// Worker-local counters; uploads count only completed host-to-device copies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreparationStats {
    pub builds: u64,
    pub hits: u64,
    pub environment_decodes: u64,
    pub lut_uploads: u64,
    pub object_uploads: u64,
    pub light_uploads: u64,
    pub upload_bytes: u64,
}

/// Typed runtime inputs. Authoring documents, animation curves and display settings
/// are not packed by the GPU and never require a preparation-cache rebuild.
pub(crate) struct PreparationInputs {
    world_render: bool,
    camera_reference: Option<f32>,
    object_world: Option<[[f32; 4]; 4]>,
    formula: crate::scene::Formula,
    julia: [f32; 3],
    julia_mix: f32,
    orbit_offset: [f32; 3],
    inversion_strength: f32,
    inversion_center: [f32; 3],
    inversion_radius: f32,
    object: crate::scene::ObjectTransform,
    camera: crate::scene::Camera,
    sun: crate::scene::Sun,
    sky: crate::scene::Sky,
    material: crate::scene::Material,
    palette: PaletteScheme,
    coloring: crate::scene::Coloring,
    trap_point: [f32; 3],
    trap_axis: u32,
    trap_scale: f32,
    render: crate::scene::Render,
    direct: Option<crate::render_profiles::DirectRender>,
    environment: crate::environment::Environment,
    objects: Vec<PreparationInputs>,
    masks: Vec<crate::scene::MaskInstance>,
    lights: Vec<crate::scene::Sun>,
}

pub(crate) fn preparation_render(scene: &Scene, root: bool) -> crate::scene::Render {
    let mut render = scene.render;
    render.denoise = Default::default();
    render.backend = crate::render_profiles::Backend::Auto;
    if root {
        render.exposure_stops = 0.0;
        render.saturation = 0.0;
        render.reinhard = false;
    }
    render
}

fn same_material(a: &crate::scene::Material, b: &crate::scene::Material) -> bool {
    // The pattern has no `..`: a new material field does not compile until it is compared here
    // or named next to `preset` (a UI label) as not reaching the upload.
    macro_rules! equal {
        ($($field:ident),+ $(,)?) => {{
            let crate::scene::Material { $($field: _,)+ preset: _ } = a;
            true $(&& a.$field == b.$field)+
        }};
    }
    equal!(
        model,
        color_source,
        base_color,
        facing,
        base,
        base_tint,
        diffuse_roughness,
        metalness,
        specular,
        specular_color,
        specular_roughness,
        specular_ior,
        specular_anisotropy,
        specular_rotation,
        transmission,
        transmission_color,
        transmission_extra_roughness,
        transmission_depth,
        sheen,
        sheen_color,
        sheen_roughness,
        coat,
        coat_color,
        coat_roughness,
        coat_ior,
        coat_affect_color,
        coat_affect_roughness,
        thin_film_thickness,
        thin_film_ior,
        emission,
        emission_color
    )
}

impl PreparationInputs {
    // Both functions destructure `Scene` without `..`: a new scene field does not compile until
    // it is either part of the key or listed as not reaching the GPU upload, so the cache can
    // never silently reuse an upload that is missing it.
    pub(crate) fn new(scene: &Scene, root: bool) -> Self {
        let Scene {
            world_render,
            camera_reference,
            objects,
            lights,
            object_world,
            masks,
            formula,
            julia,
            julia_mix,
            orbit_offset,
            inversion_strength,
            inversion_center,
            inversion_radius,
            object,
            camera,
            sun,
            sky,
            material,
            palette,
            coloring,
            trap_point,
            trap_axis,
            trap_scale,
            render: _,
            direct,
            environment,
            // Authoring and display state: never packed into the upload (`mask` reaches it
            // resolved, as `masks`; `bounds` are drawn by the viewport, never traced; `node`
            // only maps a pick back to its node).
            mask: _,
            bounds: _,
            node: _,
            document: _,
            animation: _,
            name: _,
            colour: _,
        } = scene;
        Self {
            world_render: *world_render,
            camera_reference: *camera_reference,
            object_world: *object_world,
            formula: *formula,
            julia: *julia,
            julia_mix: *julia_mix,
            orbit_offset: *orbit_offset,
            inversion_strength: *inversion_strength,
            inversion_center: *inversion_center,
            inversion_radius: *inversion_radius,
            object: *object,
            camera: *camera,
            sun: *sun,
            sky: *sky,
            material: material.clone(),
            palette: *palette,
            coloring: *coloring,
            trap_point: *trap_point,
            trap_axis: *trap_axis,
            trap_scale: *trap_scale,
            render: preparation_render(scene, root),
            direct: *direct,
            environment: environment.clone(),
            objects: objects
                .iter()
                .map(|object| Self::new(object, false))
                .collect(),
            masks: masks.clone(),
            lights: lights.clone(),
        }
    }

    pub(crate) fn matches(&self, scene: &Scene, root: bool) -> bool {
        let Scene {
            world_render,
            camera_reference,
            objects,
            lights,
            object_world,
            masks,
            formula,
            julia,
            julia_mix,
            orbit_offset,
            inversion_strength,
            inversion_center,
            inversion_radius,
            object,
            camera,
            sun,
            sky,
            material,
            palette,
            coloring,
            trap_point,
            trap_axis,
            trap_scale,
            render: _,
            direct,
            environment,
            mask: _,
            bounds: _,
            node: _,
            document: _,
            animation: _,
            name: _,
            colour: _,
        } = scene;
        self.world_render == *world_render
            && self.camera_reference == *camera_reference
            && self.object_world == *object_world
            && self.formula == *formula
            && self.julia == *julia
            && self.julia_mix == *julia_mix
            && self.orbit_offset == *orbit_offset
            && self.inversion_strength == *inversion_strength
            && self.inversion_center == *inversion_center
            && self.inversion_radius == *inversion_radius
            && self.object == *object
            && self.camera == *camera
            && self.sun == *sun
            && self.sky == *sky
            && same_material(&self.material, material)
            && self.palette == *palette
            && self.coloring == *coloring
            && self.trap_point == *trap_point
            && self.trap_axis == *trap_axis
            && self.trap_scale == *trap_scale
            && self.render == preparation_render(scene, root)
            && self.direct == *direct
            && self.environment == *environment
            && self.lights == *lights
            && self.masks == *masks
            && self.objects.len() == objects.len()
            && self
                .objects
                .iter()
                .zip(objects)
                .all(|(input, object)| input.matches(object, false))
    }
}

/// Material specialization belongs to the evaluated world, not the root scene's snapshot
/// material. Cache it with the packed upload so launches do not rescan authoring nodes.
/// Fast and Full worlds compile out the unused BSDF; mixed material worlds choose per hit.
/// Family specialization is independent and uses `PreparedScene::homogeneous_family`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorldMaterial {
    Fast,
    Full,
    Mixed,
}
impl WorldMaterial {
    pub(crate) fn from_upload(upload: &WorldUpload) -> Self {
        let mut objects = upload.objects.as_chunks::<OBJECT_STRIDE>().0.iter();
        let Some(first) = objects.next() else {
            return Self::Mixed;
        };
        let full = first[P_MATERIAL_MODEL] != 0.0;
        if objects.any(|object| (object[P_MATERIAL_MODEL] != 0.0) != full) {
            Self::Mixed
        } else if full {
            Self::Full
        } else {
            Self::Fast
        }
    }
}

pub(crate) struct PreparedScene {
    pub(crate) upload: WorldUpload,
    pub(crate) world_material: WorldMaterial,
    /// When every world object uses the same formula family (`0..Formula::COUNT`), both backends
    /// dispatch a family-specialized world pipeline instead of the 28-way dynamic switch.
    pub(crate) homogeneous_family: Option<u32>,
    pub(crate) trace: Vec<f32>,
    pub(crate) palettes: Vec<PaletteScheme>,
    pub(crate) environment: Option<(String, u64)>,
}
impl PreparedScene {
    pub(crate) fn new(scene: &Scene, width: u32, height: u32) -> Result<Self, String> {
        let upload = WorldUpload::new(scene, width, height)?;
        let world_material = WorldMaterial::from_upload(&upload);
        let homogeneous_family = {
            let mut objects = upload.objects.as_chunks::<OBJECT_STRIDE>().0.iter();
            if let Some(first) = objects.next() {
                let fam = first[P_FAMILY] as u32;
                if fam < crate::scene::Formula::COUNT
                    && objects.all(|obj| obj[P_FAMILY] as u32 == fam)
                {
                    Some(fam)
                } else {
                    None
                }
            } else {
                None
            }
        };
        Ok(Self {
            upload,
            world_material,
            homogeneous_family,
            trace: scene_trace_data(scene)?,
            palettes: std::iter::once(scene.palette)
                .chain(
                    scene
                        .objects
                        .iter()
                        .filter(|_| scene.world_render)
                        .map(|object| object.palette),
                )
                .collect(),
            environment: scene.environment.key(),
        })
    }

    /// The packed parameter block with `scene`'s tonemap-only slots: the cached upload ignores
    /// exposure, saturation and Reinhard (`preparation_render`), so a look edit reuses it.
    pub(crate) fn params(&self, scene: &Scene) -> [f32; P_COUNT] {
        let mut params = self.upload.params;
        params[P_EXPOSURE] = 2f32.powf(scene.render.exposure_stops);
        params[P_SATURATION] = scene.render.saturation;
        params[P_TONEMAP] = scene.render.reinhard as u32 as f32;
        params
    }
}

pub(crate) struct PreparationEntry {
    pub(crate) inputs: PreparationInputs,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) result: Result<Arc<PreparedScene>, String>,
}

#[derive(Default)]
pub(crate) struct PreparationCache {
    pub(crate) entries: [Option<PreparationEntry>; 4],
    next: usize,
    pub(crate) builds: u64,
    pub(crate) hits: u64,
}
impl PreparationCache {
    pub(crate) fn get(
        &mut self,
        scene: &Scene,
        width: u32,
        height: u32,
    ) -> Result<Arc<PreparedScene>, String> {
        if let Some(entry) = self.entries.iter().flatten().find(|entry| {
            entry.width == width && entry.height == height && entry.inputs.matches(scene, true)
        }) {
            self.hits += 1;
            return entry.result.clone();
        }
        self.builds += 1;
        let result = PreparedScene::new(scene, width, height).map(Arc::new);
        self.entries[self.next] = Some(PreparationEntry {
            inputs: PreparationInputs::new(scene, true),
            width,
            height,
            result: result.clone(),
        });
        self.next = (self.next + 1) % self.entries.len();
        result
    }
}

/// This state belongs to the shared Gpu, never to an individual progressive target.
/// Failed uploads invalidate identity because a copy may have partially changed device data.
pub(crate) struct Resident<K> {
    pub(crate) key: Option<K>,
}
impl<K> Default for Resident<K> {
    fn default() -> Self {
        Self { key: None }
    }
}
impl<K: PartialEq> Resident<K> {
    pub(crate) fn update(
        &mut self,
        key: K,
        upload: impl FnOnce() -> Result<(), String>,
    ) -> Result<bool, String> {
        if self.key.as_ref() == Some(&key) {
            return Ok(false);
        }
        if let Err(error) = upload() {
            self.key = None;
            return Err(error);
        }
        self.key = Some(key);
        Ok(true)
    }
}

pub(crate) struct LutResidentKey(pub(crate) Arc<PreparedScene>);
impl PartialEq for LutResidentKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.palettes == other.0.palettes && self.0.environment == other.0.environment
    }
}
pub(crate) struct ObjectResidentKey(pub(crate) Arc<PreparedScene>);
impl PartialEq for ObjectResidentKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.upload.objects == other.0.upload.objects
    }
}
pub(crate) struct LightResidentKey(pub(crate) Arc<PreparedScene>);
impl PartialEq for LightResidentKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.upload.lights == other.0.upload.lights
    }
}

#[derive(Default)]
pub(crate) struct DenoiseScratch {
    pub(crate) tiled: Vec<[f32; 4]>,
    pub(crate) color: Vec<[f32; 4]>,
    pub(crate) albedo: Vec<[f32; 4]>,
    pub(crate) normal: Vec<[f32; 4]>,
}

pub(crate) fn untile_into(
    tiled: &[[f32; 4]],
    width: usize,
    height: usize,
    normalize: bool,
    output: &mut Vec<[f32; 4]>,
) {
    output.resize(width * height, [0.0; 4]);
    if width == 0 || height == 0 {
        return;
    }
    let tiles_per_row = width.div_ceil(8);
    let untile_row = |(y, row): (usize, &mut [[f32; 4]])| {
        let tile_row = (y / 4) * tiles_per_row;
        let in_tile_y = (y % 4) * 8;
        if normalize {
            for (tx, chunk) in row.chunks_mut(8).enumerate() {
                let base = (tile_row + tx) * 32 + in_tile_y;
                let src = &tiled[base..base + chunk.len()];
                for (dst, &pixel) in chunk.iter_mut().zip(src.iter()) {
                    let inv = if pixel[3] > 0.0 {
                        pixel[3].recip()
                    } else {
                        0.0
                    };
                    *dst = [pixel[0] * inv, pixel[1] * inv, pixel[2] * inv, 1.0];
                }
            }
        } else {
            for (tx, chunk) in row.chunks_mut(8).enumerate() {
                let base = (tile_row + tx) * 32 + in_tile_y;
                chunk.copy_from_slice(&tiled[base..base + chunk.len()]);
            }
        }
    };
    if width * height >= 16_384 {
        use rayon::prelude::*;
        output
            .par_chunks_mut(width)
            .enumerate()
            .for_each(untile_row);
    } else {
        output.chunks_mut(width).enumerate().for_each(untile_row);
    }
}

/// What the displayed pixels of a film were made with: the display settings and the OCIO config
/// load (`Ocio::serial`, None without OCIO). A config reloaded from disk is a new load, so the
/// same selection redisplays; the transforms themselves are cached by selection and load
/// (`color::ColorPipeline`), so a selection change never drops them.
#[derive(Clone, PartialEq)]
pub(crate) struct DisplayKey {
    exposure: f32,
    saturation: f32,
    reinhard: bool,
    config_load: Option<u64>,
    colour: crate::ocio::Sel,
}
impl DisplayKey {
    pub(crate) fn matches(&self, scene: &Scene, config_load: Option<u64>) -> bool {
        self.exposure == scene.render.exposure_stops
            && self.saturation == scene.render.saturation
            && self.reinhard == scene.render.reinhard
            && self.config_load == config_load
            && self.colour == scene.colour
    }
    pub(crate) fn new(scene: &Scene, config_load: Option<u64>) -> Self {
        Self {
            exposure: scene.render.exposure_stops,
            saturation: scene.render.saturation,
            reinhard: scene.render.reinhard,
            config_load,
            colour: scene.colour.clone(),
        }
    }
}

/// Render-relevant runtime data only; hierarchy names and UUIDs never enter this signature.
/// An unusable mask is the same error [`WorldUpload::new`] reports for it.
pub(crate) fn scene_trace_data(scene: &Scene) -> Result<Vec<f32>, String> {
    if !scene.world_render {
        return Ok(vec![
            scene.world_render as u8 as f32,
            scene.objects.len() as f32,
            scene.lights.len() as f32,
        ]);
    }
    let mut values = Vec::with_capacity(
        3 + scene.objects.len() * (P_COUNT + 18 + MAX_MASKS * MASK_STRIDE)
            + scene.lights.len() * (P_SKY_INTENSITY - P_LIGHT_DIR),
    );
    values.extend_from_slice(&[
        scene.world_render as u8 as f32,
        scene.objects.len() as f32,
        scene.lights.len() as f32,
    ]);
    for (index, object) in scene.objects.iter().enumerate() {
        let p = object.pack(1, 1);
        values.extend_from_slice(&p[P_FAMILY..P_LIGHT_DIR]);
        values.extend_from_slice(&p[P_BASE..P_EXPOSURE]);
        values.extend_from_slice(&p[P_TRANSMISSION..P_COUNT]);
        values.push(p[P_MATERIAL_MODEL]);
        if let Some(matrix) = object.object_world {
            values.push(1.0);
            values.extend(matrix.into_iter().flatten());
        } else {
            values.push(0.0);
        }
        values.push(object.palette as u8 as f32);
        for m in &object.masks {
            values.extend(mask_row(m).map_err(|e| format!("World object {index}: {e}"))?);
        }
    }
    let mut light_template = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
    for light in &scene.lights {
        light_template.sun = *light;
        let p = light_template.pack(1, 1);
        values.extend_from_slice(&p[P_LIGHT_DIR..P_SKY_INTENSITY]);
    }
    Ok(values)
}

pub(crate) struct WorldUpload {
    pub(crate) params: [f32; P_COUNT],
    pub(crate) objects: Vec<f32>,
    pub(crate) lights: Vec<f32>,
}

fn object_matrix(object: &Scene) -> glam::Mat4 {
    object
        .object_world
        .map(|columns| glam::Mat4::from_cols_array_2d(&columns))
        .unwrap_or_else(|| {
            // Legacy ObjectTransform uses the same XYZ Euler convention as Scene::pack.
            let p = object.pack(1, 1);
            let s = p[P_OBJ_SCALE];
            glam::Mat4::from_cols(
                glam::Vec4::new(
                    p[P_OBJ_AXES] * s,
                    p[P_OBJ_AXES + 1] * s,
                    p[P_OBJ_AXES + 2] * s,
                    0.0,
                ),
                glam::Vec4::new(
                    p[P_OBJ_AXES + 3] * s,
                    p[P_OBJ_AXES + 4] * s,
                    p[P_OBJ_AXES + 5] * s,
                    0.0,
                ),
                glam::Vec4::new(
                    p[P_OBJ_AXES + 6] * s,
                    p[P_OBJ_AXES + 7] * s,
                    p[P_OBJ_AXES + 8] * s,
                    0.0,
                ),
                glam::Vec4::from((glam::Vec3::from_array(object.object.offset), 1.0)),
            )
        })
}

fn inverse_object(object: &Scene, index: usize) -> Result<glam::Mat4, String> {
    let matrix = object_matrix(object);
    let columns = matrix.to_cols_array_2d();
    if !matrix.is_finite()
        || columns[0][3] != 0.0
        || columns[1][3] != 0.0
        || columns[2][3] != 0.0
        || columns[3][3] != 1.0
        || matrix.determinant() == 0.0
    {
        return Err(format!(
            "World object {index} has a nonfinite, nonaffine or singular transform"
        ));
    }
    let inverse = matrix.inverse();
    if !inverse.is_finite() {
        return Err(format!(
            "World object {index} transform has no finite inverse"
        ));
    }
    Ok(inverse)
}

/// The conservative distance scale of an affine map, from the columns `cols` of its inverse:
/// `||A^-1||2 <= sqrt(||A^-1||1 ||A^-1||inf)`, so `1 / sqrt(norm1 * norm_inf)` is a lower bound
/// of `sigma_min(A)`, parent shear included. World objects and masks scale their DE by it.
fn stretch_scale(cols: &[[f32; 4]; 4]) -> f32 {
    let norm1 = (0..3)
        .map(|c| (0..3).map(|r| cols[c][r].abs()).sum::<f32>())
        .fold(0.0f32, f32::max);
    let norm_inf = (0..3)
        .map(|r| (0..3).map(|c| cols[c][r].abs()).sum::<f32>())
        .fold(0.0f32, f32::max);
    1.0 / (norm1.sqrt() * norm_inf.sqrt())
}

/// A mask's kernel row (`fractal_core::params` `M_*`): its world-to-mask inverse affine, the
/// conservative distance scale (as for objects), shape, invert, size, density and seed.
pub(crate) fn mask_row(m: &crate::scene::MaskInstance) -> Result<[f32; MASK_STRIDE], String> {
    use crate::scene::MaskShape;
    let matrix = glam::Mat4::from_cols_array_2d(&m.world);
    let inverse = matrix.inverse();
    if !inverse.is_finite() || matrix.determinant().abs() <= f32::MIN_POSITIVE {
        return Err("a mask transform has no finite inverse".into());
    }
    let mask = m.mask;
    if mask.size.iter().any(|v| !(v.is_finite() && *v > 0.0))
        || !(0.0..=1.0).contains(&mask.density)
    {
        return Err("a mask needs a positive size and a density in 0..1".into());
    }
    let cols = inverse.to_cols_array_2d();
    let mut row = [0.0f32; MASK_STRIDE];
    for r in 0..3 {
        for (c, column) in cols.iter().enumerate() {
            row[M_INVERSE + r * 4 + c] = column[r];
        }
    }
    row[M_SCALE] = stretch_scale(&cols);
    row[M_SHAPE] = match mask.shape {
        MaskShape::Sphere => MASK_SPHERE,
        MaskShape::Box => MASK_BOX,
        MaskShape::Cells => MASK_CELLS,
    } as f32;
    row[M_INVERT] = f32::from(u8::from(mask.invert));
    row[M_SIZE..M_SIZE + 3].copy_from_slice(&mask.size);
    row[M_DENSITY] = mask.density;
    row[M_SEED] = mask.seed as f32;
    if mask.shape == MaskShape::Cells {
        row[M_ARRAY] = 0.0;
    } else {
        use fractal_core::params::{ARRAY_LINEAR, ARRAY_RADIAL};
        row[M_ARRAY] = match mask.array.kind {
            crate::scene::ArrayKind::Linear => ARRAY_LINEAR,
            crate::scene::ArrayKind::Radial => ARRAY_RADIAL,
        } as f32;
        row[M_ARRAY_COUNT] = fractal_core::mask_array::count(mask.array.count) as f32;
        row[M_ARRAY_OFFSET..M_ARRAY_OFFSET + 3].copy_from_slice(&mask.array.offset);
        row[M_ARRAY_RADIUS] = mask.array.radius;
        row[M_ARRAY_AXIS] = mask.array.axis.code() as f32;
        // 0 stays exactly 0: the kernel reads it as a full circle.
        row[M_ARRAY_SWEEP] = mask.array.sweep_degrees.to_radians();
        row[M_ARRAY_ALIGN] = f32::from(u8::from(mask.array.align));
    }
    if !row.iter().all(|v| v.is_finite()) || row[M_SCALE] <= 0.0 {
        return Err("a mask transform is unusable".into());
    }
    Ok(row)
}

#[cfg(test)]
fn packed_array(mask: crate::scene::Mask) -> [f32; MASK_STRIDE] {
    mask_row(&crate::scene::MaskInstance {
        mask,
        world: glam::Mat4::IDENTITY.to_cols_array_2d(),
    })
    .unwrap()
}

#[test]
fn a_radial_mask_packs_its_ring_and_cells_pack_no_array() {
    use crate::scene::{ArrayAxis, ArrayKind, Mask, MaskArray, MaskShape};
    let row = packed_array(Mask {
        array: MaskArray {
            kind: ArrayKind::Radial,
            count: 4,
            radius: 2.0,
            axis: ArrayAxis::Y,
            sweep_degrees: 180.0,
            ..MaskArray::default()
        },
        ..Mask::default()
    });
    assert_eq!(row[M_ARRAY], ARRAY_RADIAL as f32);
    assert_eq!(row[M_ARRAY_COUNT], 4.0);
    assert_eq!(row[M_ARRAY_RADIUS], 2.0);
    assert_eq!(row[M_ARRAY_AXIS], 1.0);
    assert!((row[M_ARRAY_SWEEP] - std::f32::consts::PI).abs() < 1e-5);
    assert_eq!(row[M_ARRAY_ALIGN], 1.0);
    let cells = packed_array(Mask {
        shape: MaskShape::Cells,
        array: MaskArray {
            count: 8,
            ..MaskArray::default()
        },
        ..Mask::default()
    });
    assert_eq!(cells[M_ARRAY], 0.0);
}

#[cfg(test)]
pub(crate) fn validate_world(scene: &Scene) -> Result<(), String> {
    if scene.world_render {
        WorldUpload::new(scene, 1, 1)?;
    }
    Ok(())
}

impl WorldUpload {
    pub(crate) fn new(scene: &Scene, width: u32, height: u32) -> Result<Self, String> {
        scene.camera.validate_lens()?;
        let mut params: [f32; P_COUNT] = scene
            .pack(width, height)
            .try_into()
            .map_err(|_| "The scene's parameter block does not have P_COUNT slots".to_string())?;
        let mut objects = Vec::new();
        let mut lights = Vec::new();
        if scene.world_render {
            params[P_WORLD] = WORLD_ABI_VERSION as f32;
            params[P_OBJECT_COUNT] = scene.objects.len() as f32;
            params[P_LIGHT_COUNT] = scene.lights.len() as f32;
            let mut extent = 0.0f32;
            for (index, object) in scene.objects.iter().enumerate() {
                let inverse = inverse_object(object, index)?;
                let matrix = object_matrix(object);
                let mut local = object.clone();
                local.object.offset = [0.0; 3];
                local.object.rotation_degrees = [0.0; 3];
                local.object.scale = 1.0;
                let p = local.pack(width, height);
                let local_radius = p[P_CLIP_RADIUS];
                objects.extend_from_slice(&p);
                let cols = inverse.to_cols_array_2d();
                for row in 0..3 {
                    for column in &cols {
                        objects.push(column[row]);
                    }
                }
                let scale = stretch_scale(&cols);
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(format!(
                        "World object {index} has an unusable distance scale"
                    ));
                }
                objects.push(scale);
                objects.push(local_radius);
                let tangent = matrix.transform_vector3(glam::Vec3::Y).normalize();
                if !tangent.is_finite() {
                    return Err(format!(
                        "World object {index} has an unusable material tangent"
                    ));
                }
                objects.extend(tangent.to_array());
                if object.masks.len() > MAX_MASKS {
                    return Err(format!(
                        "World object {index} has {} masks; at most {MAX_MASKS} reach one object",
                        object.masks.len()
                    ));
                }
                objects.push(object.masks.len() as f32);
                for slot in 0..MAX_MASKS {
                    match object.masks.get(slot) {
                        Some(m) => objects
                            .extend(mask_row(m).map_err(|e| format!("World object {index}: {e}"))?),
                        None => objects.extend([0.0; MASK_STRIDE]),
                    }
                }
                let cols = matrix.to_cols_array_2d();
                let upper = (0..3)
                    .map(|c| (0..3).map(|r| cols[c][r] * cols[c][r]).sum::<f32>())
                    .sum::<f32>()
                    .sqrt();
                let center = matrix.transform_point3(glam::Vec3::ZERO);
                extent = extent.max(center.length() + local_radius * upper);
            }
            if !extent.is_finite() {
                return Err("World geometry bounds exceed finite render range".into());
            }
            params[P_CLIP_CENTER..P_CLIP_CENTER + 3].fill(0.0);
            params[P_CLIP_RADIUS] = extent;
            let camera = glam::Vec3::from_slice(&params[P_CAM_ORIGIN..P_CAM_ORIGIN + 3]);
            params[P_MAX_DISTANCE] = params[P_MAX_DISTANCE].max(camera.length() + 2.0 * extent);
            for light in &scene.lights {
                let mut packed = Scene::preset(Formula::Mandelbulb(Bulb::PRESET));
                packed.sun = *light;
                let mut p = packed.pack(1, 1);
                // The world kernels have no delta light: a zero angle keeps a minimal cone.
                let (omc, pdf) = crate::scene::sun_cone(
                    (light.angle * 0.5)
                        .to_radians()
                        .clamp(1e-4, std::f32::consts::PI),
                );
                p[P_SUN_ONE_MINUS_COS] = omc;
                p[P_SUN_CONE_PDF] = pdf;
                if p[P_LIGHT_DIR..P_SKY_INTENSITY]
                    .iter()
                    .any(|v| !v.is_finite())
                {
                    return Err("World directional light has nonfinite parameters".into());
                }
                lights.extend_from_slice(&p[P_LIGHT_DIR..P_SKY_INTENSITY]);
            }
        }
        Ok(Self {
            params,
            objects,
            lights,
        })
    }
}
