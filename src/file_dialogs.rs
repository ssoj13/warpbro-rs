//! Per-control file-dialog preferences, independent of documents and undo.
//! Stable keys let unrelated controls retain separate directories and filters.
//! Observation only clones changed values; persistence uses the existing I/O worker.
use egui_file_dialog::{DialogState, FileDialog};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

pub const SCENE_OPEN: &str = "scene.open";
pub const SCENE_SAVE: &str = "scene.save";
pub const ENVIRONMENT: &str = "environment.path";
pub const OCIO: &str = "colour.config";

#[derive(Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct History {
    pub directories: HashMap<String, PathBuf>,
    /// Missing key uses the control's default; explicit None means All files.
    pub filters: HashMap<String, Option<String>>,
}

impl History {
    pub fn prepare(
        &self,
        mut dialog: FileDialog,
        key: &str,
        fallback: Option<&Path>,
        default_filter: &str,
    ) -> FileDialog {
        dialog = dialog.id(egui::Id::new(key));
        if let Some(directory) = self
            .directories
            .get(key)
            .map(PathBuf::as_path)
            .or(fallback)
            .filter(|p| !p.as_os_str().is_empty())
        {
            dialog = dialog.initial_directory(directory.to_path_buf());
        }
        let filter = self
            .filters
            .get(key)
            .map(|f| f.as_deref())
            .unwrap_or(Some(default_filter));
        dialog.config_mut().default_file_filter = filter.map(str::to_owned);
        dialog
    }

    /// Retain navigation and filter changes even when the user cancels.
    /// Closed dialogs must not overwrite a control's history.
    pub fn observe(&mut self, key: &str, dialog: &FileDialog) {
        if matches!(dialog.state(), DialogState::Closed) {
            return;
        }
        if let Some(directory) = dialog.directory()
            && self.directories.get(key).map(PathBuf::as_path) != Some(directory)
        {
            self.directories
                .insert(key.to_owned(), directory.to_path_buf());
        }
        let filter = dialog.selected_file_filter().map(|f| f.name.as_str());
        if self.filters.get(key).map(|f| f.as_deref()) != Some(filter) {
            self.filters
                .insert(key.to_owned(), filter.map(str::to_owned));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn active_hdr_filter_and_all_files_survive_reopening() {
        let factory =
            || FileDialog::new().add_file_filter_extensions("HDR / EXR", vec!["hdr", "exr"]);
        let mut history = History::default();
        let mut picker = history.prepare(factory(), ENVIRONMENT, None, "HDR / EXR");
        picker.pick_file();
        assert_eq!(
            picker.selected_file_filter().map(|f| f.name.as_str()),
            Some("HDR / EXR")
        );
        history.observe(ENVIRONMENT, &picker);
        assert_eq!(
            history.filters.get(ENVIRONMENT).unwrap().as_deref(),
            Some("HDR / EXR")
        );
        let directory = history.directories.get(ENVIRONMENT).unwrap().clone();
        picker.config_mut().default_file_filter = None;
        picker.pick_file();
        history.observe(ENVIRONMENT, &picker);
        let restored: History =
            serde_json::from_str(&serde_json::to_string(&history).unwrap()).unwrap();
        let mut next = restored.prepare(factory(), ENVIRONMENT, None, "HDR / EXR");
        next.pick_file();
        assert!(next.selected_file_filter().is_none());
        assert_eq!(next.directory(), Some(directory.as_path()));
        let before = history.clone();
        history.observe(SCENE_OPEN, &FileDialog::new());
        assert!(history == before);
    }
    #[test]
    fn history_roundtrip_keeps_controls_and_all_files_distinct() {
        let mut history = History::default();
        history
            .directories
            .insert(ENVIRONMENT.into(), PathBuf::from("C:/HDR"));
        history
            .directories
            .insert(SCENE_OPEN.into(), PathBuf::from("C:/Scenes"));
        history.filters.insert(ENVIRONMENT.into(), None);
        let restored: History =
            serde_json::from_str(&serde_json::to_string(&history).unwrap()).unwrap();
        assert!(history == restored);
        let mut dialog = restored.prepare(
            FileDialog::new().add_file_filter_extensions("HDR / EXR", vec!["hdr", "exr"]),
            ENVIRONMENT,
            None,
            "HDR / EXR",
        );
        assert_eq!(
            dialog.config_mut().initial_directory,
            PathBuf::from("C:/HDR")
        );
        assert_eq!(dialog.config_mut().default_file_filter, None);
        let mut first =
            History::default().prepare(FileDialog::new(), ENVIRONMENT, None, "HDR / EXR");
        assert_eq!(
            first.config_mut().default_file_filter.as_deref(),
            Some("HDR / EXR")
        );
        assert_eq!(
            serde_json::from_str::<History>("{}")
                .unwrap()
                .directories
                .len(),
            0
        );
    }
}
