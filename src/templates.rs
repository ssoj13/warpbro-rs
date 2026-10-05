//! Scene templates: built into the binary, overridable by files in `~/.warpbro/templates`.
//! All disk work runs on the existing I/O worker; the UI retains the catalog and refreshes
//! it only on request or after a save.
use std::path::{Path, PathBuf};
pub fn directory() -> PathBuf {
    crate::warpbro_dir().join("templates")
}
/// Where a catalog template comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Built into the binary: `presets::scene(index)`.
    Builtin(usize),
    /// A scene file in the templates directory.
    File(PathBuf),
}
/// One entry of the Templates menu.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    /// The built-in's description, also kept when a file overrides it.
    pub description: Option<&'static str>,
    pub source: Source,
    /// A file replacing the built-in of the same name.
    pub overrides: bool,
}
/// The built-in templates, then the templates directory on top: a file whose name (slug)
/// matches a built-in replaces it, any other file is added. The binary needs no files; a
/// missing directory is simply empty and is never created here. Scene decoding is identical
/// to ordinary project files.
pub fn catalog(directory: &Path) -> Result<Vec<Entry>, String> {
    let mut entries: Vec<Entry> = crate::presets::ANIMATED
        .iter()
        .enumerate()
        .map(|(index, preset)| Entry {
            name: preset.name.to_owned(),
            description: Some(preset.description),
            source: Source::Builtin(index),
            overrides: false,
        })
        .collect();
    let builtins = entries.len();
    let read = match std::fs::read_dir(directory) {
        Ok(read) => read,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(entries),
        Err(e) => return Err(format!("{}: {e}", directory.display())),
    };
    let mut files = Vec::new();
    for entry in read {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let Some(stem) = name.strip_suffix(".frac.json").or_else(|| name.strip_suffix(".json")) else {
            continue;
        };
        if entry.file_type().map_err(|e| e.to_string())?.is_file() {
            files.push((stem.to_owned(), path));
        }
    }
    files.sort();
    for (stem, path) in files {
        let key = crate::slug(&stem);
        match entries[..builtins].iter_mut().find(|e| crate::slug(&e.name) == key) {
            Some(builtin) => {
                builtin.source = Source::File(path);
                builtin.overrides = true;
            }
            None => entries.push(Entry { name: stem, description: None, source: Source::File(path), overrides: false }),
        }
    }
    Ok(entries)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileAction {
    Open,
    Save,
    OpenTemplate,
    SaveTemplate,
}
impl FileAction {
    pub fn save(self) -> bool {
        matches!(self, Self::Save | Self::SaveTemplate)
    }
    pub fn template(self) -> bool {
        matches!(self, Self::OpenTemplate | Self::SaveTemplate)
    }
    pub fn history_key(self) -> &'static str {
        match self {
            Self::Open => crate::file_dialogs::SCENE_OPEN,
            Self::Save => crate::file_dialogs::SCENE_SAVE,
            Self::OpenTemplate => "template.open",
            Self::SaveTemplate => "template.save",
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn builtins_need_no_directory_and_files_override_by_name() {
        let dir = std::env::temp_dir().join(format!("warpbro-templates-test-{}", std::process::id()));
        let builtin = catalog(&dir).unwrap();
        assert_eq!(builtin.len(), crate::presets::ANIMATED.len());
        assert!(builtin.iter().all(|e| matches!(e.source, Source::Builtin(_)) && e.description.is_some()));
        assert!(!dir.exists(), "the catalog never creates the directory");
        std::fs::create_dir_all(&dir).unwrap();
        let first = crate::presets::ANIMATED[0].name;
        let text = serde_json::to_string(&crate::presets::scene(0).unwrap().document).unwrap();
        let replaced = dir.join(format!("{}.frac.json", crate::file_stem(first)));
        let user = dir.join("user-scene.json");
        std::fs::write(&replaced, &text).unwrap();
        std::fs::write(&user, &text).unwrap();
        std::fs::write(dir.join("notes.txt"), "ignored").unwrap();
        let merged = catalog(&dir).unwrap();
        assert_eq!(merged.len(), builtin.len() + 1);
        assert_eq!(merged[0].name, first);
        assert_eq!(merged[0].source, Source::File(replaced));
        assert!(merged[0].overrides && merged[0].description.is_some());
        let added = merged.last().unwrap();
        assert_eq!((added.name.as_str(), &added.source), ("user-scene", &Source::File(user.clone())));
        let scene = crate::io_service::decode_scene(&std::fs::read_to_string(&user).unwrap()).unwrap();
        assert_eq!(scene.document.unwrap().last, 249);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
