//! Editable scene templates. All disk work runs on the existing I/O worker;
//! the UI retains a catalog and refreshes it only on request or after a save.
use std::path::{Path, PathBuf};
pub fn directory() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".warpbro/templates")
}
/// Seed once, never replace user edits or resurrect templates deliberately deleted
/// after initialization. Scene decoding is identical to ordinary project files.
pub fn scan(directory: &Path) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let marker = directory.join(".initialized");
    if !marker.exists() {
        for (index, preset) in crate::presets::ANIMATED.iter().enumerate() {
            let path = directory.join(format!("{}.frac.json", crate::slug(preset.name)));
            if path.exists() {
                continue;
            }
            let scene = crate::presets::scene(index)?;
            let text = serde_json::to_string_pretty(&scene.document).map_err(|e| e.to_string())?;
            use std::io::Write;
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
            {
                Ok(mut file) => file.write_all(text.as_bytes()).map_err(|e| e.to_string())?,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
                Err(e) => return Err(e.to_string()),
            }
        }
        std::fs::write(marker, b"1").map_err(|e| e.to_string())?;
    }
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if entry.file_type().map_err(|e| e.to_string())?.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
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
    #[test]
    fn templates_are_normal_scenes_and_initialization_preserves_user_changes() {
        let path =
            std::env::temp_dir().join(format!("warpbro-templates-test-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        let initial = super::scan(&path).unwrap();
        assert_eq!(initial.len(), 5);
        for file in &initial {
            let scene =
                crate::io_service::decode_scene(&std::fs::read_to_string(file).unwrap()).unwrap();
            assert_eq!(scene.document.unwrap().last, 249);
        }
        let edited = initial[0].clone();
        let text = std::fs::read_to_string(&edited).unwrap();
        std::fs::write(&edited, format!("\n{text}\n")).unwrap();
        std::fs::remove_file(&initial[1]).unwrap();
        let user = path.join("user-scene.json");
        std::fs::write(&user, &text).unwrap();
        let refreshed = super::scan(&path).unwrap();
        assert_eq!(refreshed.len(), 5);
        assert!(refreshed.contains(&user));
        assert!(!initial[1].exists());
        assert!(std::fs::read_to_string(edited).unwrap().starts_with('\n'));
        for file in refreshed {
            std::fs::remove_file(file).unwrap();
        }
        std::fs::remove_file(path.join(".initialized")).unwrap();
        std::fs::remove_dir(path).unwrap();
    }
}
