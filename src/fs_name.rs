//! File names for everything WarpBro writes: re-exports [`video_io::naming`] + the scene suffix.

pub use video_io::naming::{check, frame_file, sequence_pattern, slug, stem};

/// The suffix of a scene (and template) file.
pub const SCENE_SUFFIX: &str = "frac.json";
