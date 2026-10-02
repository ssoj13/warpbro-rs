//! The material library: the curated presets of usd-rs `usd-mat-lib` (data copied verbatim,
//! USD authoring dropped; `path` is now just the preset name) and their translation onto this
//! renderer's Standard Surface inputs, following usd-rs `usd-hd-pt` material.rs
//! (UsdPreviewSurface -> StandardSurfaceParams) and `pt-material-ext` (sheen, anisotropy,
//! facing mix).

use crate::scene::{ColorSource, Facing, Material, MaterialModel};

/// One curated preset (the six `UsdPreviewSurface` knobs + optional PT extensions).
///
/// All presets are authored as real `UsdShade.Material` prims that flow through
/// the `UsdPreviewSurface` → `StandardSurface` translator, so they render
/// identically in Storm and the path tracer. The two optional extension fields
/// are authored only by the path tracer (`pt-material-ext`); Storm ignores them.
pub struct MaterialPreset {
    /// Absolute USD path, e.g. `WoodOak`.
    pub path: &'static str,
    /// The UI category / tag bucket this preset belongs to — the SSOT for the
    /// Materials-panel grid section header (e.g. "Wood", "Metal", "Glass",
    /// "Plastic", "Paper", "Rubber", "Fabric", "Emissive", "CarPaint",
    /// "Stone", "Ceramic", "Leather"). This is an EXPLICIT tag (not derived from
    /// the path/name), so renames never re-bucket a preset and look-alike names
    /// (e.g. `Cardboard`, `RubberTyre`) land in the correct bucket.
    pub category: &'static str,
    /// Diffuse colour in Rec.709 (rotated to ACEScg at upload by the translator).
    /// For metals (`metallic = 1`) this is the conductor's bright F0 tint.
    pub diffuse: [f32; 3],
    /// Microfacet roughness `[0, 1]` (0 = mirror / glossy, 1 = fully matte).
    pub roughness: f32,
    /// Metallic workflow flag `0.0`/`1.0` (1 = conductor, uses `diffuse` as F0).
    pub metallic: f32,
    /// Opacity `[0, 1]` (< 1 = transmissive glass).
    pub opacity: f32,
    /// Index of refraction (1.5 default dielectric; ~1.45 for frosted glass).
    pub ior: f32,
    /// Emissive radiance (ACEScg energy units; `[0, 0, 0]` for non-emissive).
    pub emissive: [f32; 3],
    /// OPTIONAL Charlie **sheen** (velvet): `(color_rec709, roughness)`. When
    /// `Some`, [`author_preset`] calls [`pt_material_ext::author_sheen`] on the
    /// `/Surface` shader so the PT renders the retroreflective sheen lobe.
    pub sheen: Option<([f32; 3], f32)>,
    /// OPTIONAL **anisotropy** (brushed metal): `(amount, brush_dir_object_space)`.
    /// `amount` in `[-1, 1]`; when `Some`, [`author_preset`] calls
    /// [`pt_material_ext::author_anisotropy`] so the PT renders the UV-free
    /// brushed-GGX lobe oriented along `brush_dir`.
    pub anisotropy: Option<(f32, [f32; 3])>,
    /// OPTIONAL **facing-mix** (pearlescent / falloff): the material B (the
    /// grazing look) + grazing exponent, as `(B diffuse Rec.709, B roughness,
    /// B metallic, exponent)`. When `Some`, [`author_preset`] calls
    /// [`pt_material_ext::author_facing`] to author a SECOND `/SurfaceB`
    /// `UsdPreviewSurface` + `inputs:facingExponent` on `/Surface`, so the PT
    /// blends A→B by facing ratio (`pow(1 - |N·V|, exponent)`). Storm ignores it.
    pub facing: Option<([f32; 3], f32, f32, f32)>,
}

impl MaterialPreset {
    /// Construct a plain (no-extension) preset. Keeps the [`PRESETS`] table terse
    /// by defaulting `sheen` / `anisotropy` to `None`. `category` is the explicit
    /// UI tag bucket (2nd arg, right after `path`).
    const fn plain(
        path: &'static str,
        category: &'static str,
        diffuse: [f32; 3],
        roughness: f32,
        metallic: f32,
        opacity: f32,
        ior: f32,
        emissive: [f32; 3],
    ) -> Self {
        Self {
            path,
            category,
            diffuse,
            roughness,
            metallic,
            opacity,
            ior,
            emissive,
            sheen: None,
            anisotropy: None,
            facing: None,
        }
    }
}

/// The curated ~50-material preset set, grouped by class.
///
/// This is the single source of truth: a host iterates it to author USD prims
/// and enumerates it to build a material dropdown.
///
/// # Back-compat — the first five paths are STABLE
/// `PRESETS[0..5]` preserve the legacy viewer names/paths
/// (`Plastic` / `Metal` / `Paper` / `Glass` / `Emissive`) in that exact order so
/// persisted scatter configs that reference those paths by string keep resolving
/// and seed-stable scatter assignments do not shift. Append-only beyond index 5.
pub const PRESETS: &[MaterialPreset] = &[
    // --- Back-compat canonical 5 (legacy names/paths — DO NOT reorder/rename) ---
    // Plastic: glossy red dielectric.
    MaterialPreset::plain("Plastic",  "Plastic", [0.80, 0.10, 0.10], 0.35, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    // Metal: warm gold conductor (metallic workflow).
    MaterialPreset::plain("Metal",    "Metal", [0.94, 0.74, 0.34], 0.18, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    // Paper: warm cream rough dielectric.
    MaterialPreset::plain("Paper",    "Paper", [0.90, 0.85, 0.70], 0.90, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    // Glass: cyan-tinted low-opacity smooth dielectric.
    MaterialPreset::plain("Glass",    "Glass", [0.40, 0.70, 0.90], 0.05, 0.0, 0.1, 1.5, [0.0, 0.0, 0.0]),
    // Emissive: dark base with a bright warm emissive colour (the ONE emissive).
    MaterialPreset::plain("Emissive", "Emissive", [0.02, 0.02, 0.02], 0.50, 0.0, 1.0, 1.5, [3.0, 2.4, 1.2]),

    // --- Woods (warm diffuse + mid roughness; colour carries the species) ---
    MaterialPreset::plain("WoodOak",    "Wood", [0.62, 0.46, 0.28], 0.45, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("WoodWalnut", "Wood", [0.26, 0.16, 0.10], 0.50, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("WoodPine",   "Wood", [0.80, 0.67, 0.40], 0.50, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("WoodMahogany", "Wood", [0.36, 0.16, 0.12], 0.48, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),

    // --- Paper / card (very rough dielectric) ---
    MaterialPreset::plain("PaperMatte", "Paper", [0.92, 0.90, 0.84], 0.90, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("Cardboard",  "Paper", [0.66, 0.52, 0.34], 0.90, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),

    // --- Rubber (matte dielectric) ---
    MaterialPreset::plain("RubberTyre", "Rubber", [0.04, 0.04, 0.04], 0.70, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("RubberRed",  "Rubber", [0.55, 0.06, 0.06], 0.80, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),

    // --- Plastics (dielectric; roughness = gloss↔matte) ---
    MaterialPreset::plain("PlasticGlossyBlue",  "Plastic", [0.08, 0.22, 0.90], 0.10, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("PlasticGlossyGreen", "Plastic", [0.08, 0.66, 0.24], 0.18, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("PlasticMatteYellow", "Plastic", [0.93, 0.78, 0.12], 0.55, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("PlasticMatteOrange", "Plastic", [0.92, 0.38, 0.06], 0.55, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("PlasticIvory",       "Plastic", [0.92, 0.88, 0.78], 0.30, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("PlasticCharcoal",    "Plastic", [0.05, 0.05, 0.06], 0.33, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),

    // --- Metals (conductor, metallic=1; `diffuse` = bright F0 tint; roughness = polish) ---
    MaterialPreset::plain("MetalChrome",     "Metal", [0.95, 0.95, 0.96], 0.04, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("MetalGold",       "Metal", [0.94, 0.74, 0.34], 0.14, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("MetalCopper",     "Metal", [0.95, 0.64, 0.54], 0.12, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("MetalBrass",      "Metal", [0.86, 0.72, 0.38], 0.18, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("MetalAluminium", "Metal", [0.91, 0.92, 0.94], 0.30, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("MetalSteel",      "Metal", [0.62, 0.64, 0.66], 0.35, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("MetalTitanium",   "Metal", [0.62, 0.61, 0.63], 0.28, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("MetalIron",       "Metal", [0.56, 0.55, 0.54], 0.55, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("MetalAnodizedBlue", "Metal", [0.16, 0.42, 0.88], 0.16, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("MetalAnodizedRed",  "Metal", [0.86, 0.14, 0.16], 0.16, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    // Brushed metals: same conductor base, but carry anisotropy so the PT
    // renders the UV-free brushed-GGX lobe (amount 0.6, brush dir = object +X).
    MaterialPreset {
        path: "BrushedSteel",
        category: "Metal",
        diffuse: [0.62, 0.64, 0.66], roughness: 0.42, metallic: 1.0, opacity: 1.0, ior: 1.5,
        emissive: [0.0, 0.0, 0.0], sheen: None, anisotropy: Some((0.6, [1.0, 0.0, 0.0])), facing: None,
    },
    MaterialPreset {
        path: "BrushedAluminium",
        category: "Metal",
        diffuse: [0.91, 0.92, 0.94], roughness: 0.40, metallic: 1.0, opacity: 1.0, ior: 1.5,
        emissive: [0.0, 0.0, 0.0], sheen: None, anisotropy: Some((0.6, [1.0, 0.0, 0.0])), facing: None,
    },

    // --- Ceramic (glazed glossy / matte terracotta) ---
    MaterialPreset::plain("CeramicWhite",  "Ceramic", [0.92, 0.92, 0.90], 0.12, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("CeramicCobalt", "Ceramic", [0.10, 0.20, 0.66], 0.14, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("Terracotta",    "Ceramic", [0.70, 0.33, 0.20], 0.60, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),

    // --- Leather (saturated warm/black, mid-high roughness) ---
    MaterialPreset::plain("LeatherBlack", "Leather", [0.05, 0.04, 0.04], 0.45, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("LeatherBrown", "Leather", [0.32, 0.18, 0.10], 0.60, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("LeatherTan",   "Leather", [0.58, 0.40, 0.24], 0.55, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),

    // --- Stone / concrete ---
    MaterialPreset::plain("Concrete", "Stone", [0.55, 0.55, 0.53], 0.85, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("Slate",    "Stone", [0.18, 0.19, 0.21], 0.75, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("Marble",   "Stone", [0.90, 0.90, 0.88], 0.30, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),

    // --- Fabrics / velvet (Charlie sheen via pt-material-ext) ---
    // Velvets carry sheen so the PT renders the retroreflective sheen lobe; the
    // base diffuse stays deep/saturated. Cotton/Felt use a subtler sheen tint.
    MaterialPreset {
        path: "VelvetCrimson",
        category: "Fabric",
        diffuse: [0.45, 0.04, 0.10], roughness: 0.82, metallic: 0.0, opacity: 1.0, ior: 1.5,
        emissive: [0.0, 0.0, 0.0], sheen: Some(([0.85, 0.20, 0.30], 0.3)), anisotropy: None, facing: None,
    },
    MaterialPreset {
        path: "VelvetRoyalBlue",
        category: "Fabric",
        diffuse: [0.06, 0.10, 0.45], roughness: 0.82, metallic: 0.0, opacity: 1.0, ior: 1.5,
        emissive: [0.0, 0.0, 0.0], sheen: Some(([0.30, 0.40, 0.90], 0.3)), anisotropy: None, facing: None,
    },
    MaterialPreset {
        path: "CottonLinen",
        category: "Fabric",
        diffuse: [0.82, 0.76, 0.62], roughness: 0.90, metallic: 0.0, opacity: 1.0, ior: 1.5,
        emissive: [0.0, 0.0, 0.0], sheen: Some(([0.50, 0.48, 0.42], 0.5)), anisotropy: None, facing: None,
    },
    MaterialPreset {
        path: "FeltGrey",
        category: "Fabric",
        diffuse: [0.40, 0.42, 0.45], roughness: 0.95, metallic: 0.0, opacity: 1.0, ior: 1.5,
        emissive: [0.0, 0.0, 0.0], sheen: Some(([0.35, 0.36, 0.40], 0.6)), anisotropy: None, facing: None,
    },

    // --- Glass (exactly THREE: clear / amber tinted / frosted) ---
    MaterialPreset::plain("GlassClear",   "Glass", [0.92, 0.95, 0.97], 0.02, 0.0, 0.08, 1.5,  [0.0, 0.0, 0.0]),
    MaterialPreset::plain("GlassAmber",   "Glass", [0.85, 0.55, 0.20], 0.03, 0.0, 0.14, 1.5,  [0.0, 0.0, 0.0]),
    MaterialPreset::plain("GlassFrosted", "Glass", [0.90, 0.92, 0.95], 0.32, 0.0, 0.30, 1.45, [0.0, 0.0, 0.0]),

    // --- Car paint (high-gloss low-roughness dielectric, clearcoat-ish) ---
    MaterialPreset::plain("CarPaintCherry", "CarPaint", [0.55, 0.02, 0.06], 0.08, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("CarPaintSilver", "CarPaint", [0.62, 0.64, 0.68], 0.10, 1.0, 1.0, 1.5, [0.0, 0.0, 0.0]),
    MaterialPreset::plain("CarPaintMidnightBlue", "CarPaint", [0.04, 0.08, 0.28], 0.09, 0.0, 1.0, 1.5, [0.0, 0.0, 0.0]),

    // --- Facing-mix (pearlescent / falloff via pt-material-ext) ---
    // These carry a SECOND `/SurfaceB` surface; the PT blends the head-on look
    // (the six base knobs = material A) toward B by facing ratio. Storm shows
    // only material A. `facing = (B diffuse Rec.709, B roughness, B metallic, exponent)`.
    // Pearlescent: pale neutral head-on, saturated blue/purple grazing tint.
    MaterialPreset {
        path: "Pearlescent",
        category: "CarPaint",
        diffuse: [0.82, 0.82, 0.86], roughness: 0.18, metallic: 0.0, opacity: 1.0, ior: 1.5,
        emissive: [0.0, 0.0, 0.0], sheen: None, anisotropy: None,
        facing: Some(([0.18, 0.10, 0.65], 0.22, 0.0, 3.0)),
    },
    // Oil slick: dark head-on, vivid teal/green grazing tint (high contrast).
    MaterialPreset {
        path: "OilSlick",
        category: "CarPaint",
        diffuse: [0.04, 0.05, 0.07], roughness: 0.12, metallic: 0.0, opacity: 1.0, ior: 1.5,
        emissive: [0.0, 0.0, 0.0], sheen: None, anisotropy: None,
        facing: Some(([0.05, 0.75, 0.55], 0.16, 0.0, 4.0)),
    },
];

/// Category order of the library tab (usd-mat-lib's grouping).
pub const CATEGORIES: [&str; 12] = [
    "Metal", "Plastic", "CarPaint", "Ceramic", "Stone", "Wood", "Leather", "Fabric", "Rubber", "Paper",
    "Glass", "Emissive",
];

impl MaterialPreset {
    pub fn name(&self) -> &'static str {
        self.path
    }

    /// Whether the preset needs the Standard Surface kernels (sheen / anisotropy lobes).
    pub fn needs_standard_surface(&self) -> bool {
        self.sheen.is_some() || self.anisotropy.is_some()
    }

    /// UsdPreviewSurface -> Standard Surface (usd-hd-pt material.rs): diffuseColor -> base_color
    /// (weight 1), metallic -> metalness, roughness -> specular_roughness, ior -> specular_IOR,
    /// emissiveColor -> emission_color (weight 1 when non-zero). The fractal colour source
    /// switches to the material colour.
    ///
    /// opacity < 1 (glass) has no transmissive equivalent on a distance-estimated fractal (no
    /// interior to refract through), so it becomes a clear-coated, dark, smooth dielectric whose
    /// base weight is the opacity: the reflective look of glass without the refraction.
    pub fn apply(&self, m: &mut Material) {
        let defaults = Material::default();
        let model = m.model;
        *m = Material { model, ..defaults };
        m.color_source = ColorSource::Material;
        m.base_color = self.diffuse;
        m.metalness = self.metallic;
        m.specular_roughness = self.roughness;
        m.specular_ior = self.ior;
        if self.emissive.iter().any(|&c| c > 0.0) {
            m.emission = 1.0;
            m.emission_color = self.emissive;
        }
        if self.opacity < 1.0 {
            m.base = self.opacity;
            m.coat = 1.0;
            m.coat_roughness = self.roughness;
            m.coat_ior = self.ior;
        }
        if let Some((color, roughness)) = self.sheen {
            m.sheen = 1.0;
            m.sheen_color = color;
            m.sheen_roughness = roughness;
        }
        if let Some((amount, _brush_dir)) = self.anisotropy {
            // The brush direction follows the object's up axis (the shading tangent hint).
            m.specular_anisotropy = amount.abs();
        }
        m.facing = self.facing.map(|(color, roughness, metallic, exponent)| Facing { color, roughness, metallic, exponent });
        if self.needs_standard_surface() {
            m.model = MaterialModel::StandardSurface;
        }
        m.preset = Some(self.path.to_string());
    }
}
