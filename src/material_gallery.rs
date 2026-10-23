//! Cached projection of authored Material nodes for Gallery cards.
//! Selection, assignment, and rendering stay in their existing domain/UI adapters.
use crate::scene::Material;
use crate::world::{NodeId, WorldEditor, WorldKind};

/// Which card a thumbnail render belongs to. It travels through the render worker as an opaque
/// `u64` (`Command::Thumbnail::id`): the kind in the top two bits, the index or the material
/// ticket below. The one encoder and decoder of thumbnail ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ThumbId {
    /// A Gallery preset (index into the gallery cards).
    Gallery(usize),
    /// A saved bookmark (index into the bookmark cards).
    Bookmark(usize),
    /// A material-library swatch (index into `materials::PRESETS`).
    Swatch(usize),
    /// An authored Material node's card: a [`MaterialGallery`] ticket.
    Material(u64),
}
impl ThumbId {
    const SHIFT: u32 = 62;
    /// The largest index or ticket an id carries.
    pub(crate) const MAX_INDEX: u64 = (1 << Self::SHIFT) - 1;

    pub(crate) fn encode(self) -> u64 {
        let (kind, index) = match self {
            Self::Gallery(i) => (0, i as u64),
            Self::Bookmark(i) => (1, i as u64),
            Self::Swatch(i) => (2, i as u64),
            Self::Material(ticket) => (3, ticket),
        };
        (kind << Self::SHIFT) | (index & Self::MAX_INDEX)
    }

    pub(crate) fn decode(id: u64) -> Self {
        let index = id & Self::MAX_INDEX;
        match id >> Self::SHIFT {
            0 => Self::Gallery(index as usize),
            1 => Self::Bookmark(index as usize),
            2 => Self::Swatch(index as usize),
            _ => Self::Material(index),
        }
    }
}

/// One authored Material UUID and its current evaluated presentation.
pub(crate) struct MaterialEntry {
    pub id: NodeId,
    pub name: String,
    pub material: Material,
    pub thumb: Option<egui::TextureHandle>,
    pub error: Option<String>,
    pending: Option<u64>,
}

/// CPU-only intent for the existing asynchronous thumbnail worker.
pub(crate) struct MaterialThumbnailRequest {
    pub id: u64,
    pub material: Material,
}

struct ProjectedMaterial {
    id: NodeId,
    name: Option<String>,
    material: Material,
}

/// Idle refresh is an identity check. Dirty refresh preserves cards/textures for
/// unchanged evaluated materials and rejects results for changed/deleted nodes.
#[derive(Default)]
pub(crate) struct MaterialGallery {
    pub entries: Vec<MaterialEntry>,
    key: Option<(NodeId, u64, u64)>,
    document_id: Option<NodeId>,
    scratch: Vec<ProjectedMaterial>,
    sequence: u64,
}
impl MaterialGallery {
    /// Force evaluation after replacing an editor, even when loading the same
    /// document UUID/revision. Keep this cache instance to preserve ticket IDs.
    pub fn invalidate(&mut self) {
        self.key = None;
        for entry in &mut self.entries {
            entry.error = None;
        }
    }

    /// Refresh atomically through the document's authoritative material evaluator.
    /// Returns false without evaluation/allocation when doc/revision/frame is unchanged.
    pub fn refresh(&mut self, editor: &WorldEditor, frame: f64) -> Result<bool, String> {
        if !frame.is_finite() {
            return Err("Invalid material gallery frame".into());
        }
        let frame_bits = if frame == 0.0 { 0 } else { frame.to_bits() };
        let key = (
            NodeId(editor.document.graph.id),
            editor.revision(),
            frame_bits,
        );
        if self.key == Some(key) {
            return Ok(false);
        }
        let structure_changed = self.key.is_none_or(|old| old.0 != key.0 || old.1 != key.1);
        self.scratch.clear();
        if structure_changed {
            for node in editor
                .document
                .nodes()
                .into_iter()
                .filter(|node| node.kind == WorldKind::Material)
            {
                self.scratch.push(ProjectedMaterial {
                    id: node.id,
                    name: Some(node.name),
                    material: editor.document.material(node.id, frame)?,
                });
            }
        } else {
            // A clock tick changes values, not UUIDs/order/names. Reuse projection
            // identity instead of rebuilding/sorting the document's node list.
            for entry in &self.entries {
                self.scratch.push(ProjectedMaterial {
                    id: entry.id,
                    name: None,
                    material: editor.document.material(entry.id, frame)?,
                });
            }
        }
        // Identical node UUIDs in a different document must not accept old requests.
        if self.document_id != Some(key.0) {
            self.entries.clear();
        }
        if structure_changed {
            self.entries
                .retain(|entry| self.scratch.iter().any(|node| node.id == entry.id));
        }
        for (position, node) in self.scratch.drain(..).enumerate() {
            let index = if structure_changed {
                self.entries.iter().position(|entry| entry.id == node.id)
            } else {
                Some(position)
            };
            if let Some(index) = index {
                self.entries.swap(position, index);
                let entry = &mut self.entries[position];
                if let Some(name) = node.name {
                    entry.name.clone_from(&name);
                }
                if entry.material != node.material {
                    entry.material = node.material;
                    entry.thumb = None;
                    entry.pending = None;
                    entry.error = None;
                }
            } else {
                self.entries.insert(
                    position,
                    MaterialEntry {
                        id: node.id,
                        name: node.name.unwrap_or_default(),
                        material: node.material,
                        thumb: None,
                        error: None,
                        pending: None,
                    },
                );
            }
        }
        self.document_id = Some(key.0);
        self.key = Some(key);
        Ok(true)
    }

    /// Reserve one missing thumbnail. Retry rejected submissions, record render
    /// failures with `fail_thumbnail`, and refresh before accepting results.
    pub fn next_thumbnail(&mut self) -> Option<MaterialThumbnailRequest> {
        let entry = self.entries.iter_mut().find(|entry| {
            entry.thumb.is_none() && entry.pending.is_none() && entry.error.is_none()
        })?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .filter(|value| *value <= ThumbId::MAX_INDEX)?;
        let id = ThumbId::Material(self.sequence).encode();
        entry.pending = Some(id);
        Some(MaterialThumbnailRequest {
            id,
            material: entry.material.clone(),
        })
    }

    /// Check before allocating/uploading an egui texture for a worker completion.
    pub fn is_pending(&self, id: u64) -> bool {
        self.entries.iter().any(|entry| entry.pending == Some(id))
    }

    /// Attach a matching result; changed materials/documents discard stale tickets.
    pub fn accept_thumbnail(&mut self, id: u64, thumb: egui::TextureHandle) -> bool {
        let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.pending == Some(id))
        else {
            return false;
        };
        entry.thumb = Some(thumb);
        entry.pending = None;
        true
    }

    /// Record one terminal worker/colour error. Repaint must not retry a failed
    /// material until its value changes or the user explicitly refreshes it.
    pub fn fail_thumbnail(&mut self, id: u64, error: String) -> bool {
        let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.pending == Some(id))
        else {
            return false;
        };
        entry.pending = None;
        entry.error = Some(error);
        true
    }

    /// Explicitly refresh one node, rejecting any prior in-flight result.
    pub fn invalidate_thumbnail(&mut self, node: NodeId) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == node) else {
            return false;
        };
        entry.thumb = None;
        entry.pending = None;
        entry.error = None;
        true
    }

    /// Restore an unsent request to the existing asynchronous queue.
    pub fn retry_thumbnail(&mut self, id: u64) {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.pending == Some(id))
        {
            entry.pending = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Bulb, Formula, Scene};
    use crate::world::{WorldCommand, WorldDocument};
    use serde_json::json;

    /// The node whose card holds reserved ticket `id`.
    fn pending_node(gallery: &MaterialGallery, id: u64) -> Option<NodeId> {
        gallery
            .entries
            .iter()
            .find(|entry| entry.pending == Some(id))
            .map(|entry| entry.id)
    }

    fn world() -> (WorldEditor, NodeId, NodeId) {
        let mut editor = WorldEditor::new(WorldDocument::from_scene(&Scene::preset(
            Formula::Mandelbulb(Bulb::PRESET),
        )));
        let first = editor
            .document
            .nodes()
            .into_iter()
            .find(|node| node.kind == WorldKind::Material)
            .unwrap()
            .id;
        editor
            .execute(WorldCommand::Rename {
                id: first,
                name: "First".into(),
            })
            .unwrap();
        editor
            .execute(WorldCommand::Create {
                kind: WorldKind::Material,
                name: "Second".into(),
                parent: None,
            })
            .unwrap();
        let second = editor.selection.unwrap();
        (editor, first, second)
    }
    fn texture(ctx: &egui::Context) -> egui::TextureHandle {
        ctx.load_texture(
            "material-test",
            egui::ColorImage::new([1, 1], vec![egui::Color32::WHITE]),
            egui::TextureOptions::NEAREST,
        )
    }

    #[test]
    fn thumbnail_ids_round_trip_per_kind() {
        for id in [
            ThumbId::Gallery(0),
            ThumbId::Gallery(37),
            ThumbId::Bookmark(1_000_001),
            ThumbId::Swatch(5),
            ThumbId::Material(ThumbId::MAX_INDEX),
        ] {
            assert_eq!(ThumbId::decode(id.encode()), id);
        }
        assert_ne!(ThumbId::Gallery(3).encode(), ThumbId::Bookmark(3).encode());
    }

    #[test]
    fn terminal_thumbnail_errors_stop_repaint_retries_until_explicit_refresh_or_material_change() {
        let (mut editor, first, second) = world();
        let mut gallery = MaterialGallery::default();
        gallery.refresh(&editor, 0.0).unwrap();
        let old = gallery.next_thumbnail().unwrap();
        assert_eq!(pending_node(&gallery, old.id), Some(first));
        assert!(gallery.fail_thumbnail(old.id, "colour transform failed".into()));
        let other = gallery.next_thumbnail().unwrap();
        assert_eq!(pending_node(&gallery, other.id), Some(second));
        assert!(gallery.fail_thumbnail(other.id, "worker failed".into()));
        for _ in 0..32 {
            assert!(gallery.next_thumbnail().is_none());
        }
        gallery.refresh(&editor, 1.0).unwrap();
        assert!(
            gallery.next_thumbnail().is_none(),
            "unchanged material on a clock tick is not a retry"
        );
        assert!(gallery.invalidate_thumbnail(first));
        assert!(!gallery.fail_thumbnail(old.id, "stale error".into()));
        let current = gallery.next_thumbnail().unwrap();
        assert_ne!(current.id, old.id);
        assert_eq!(pending_node(&gallery, current.id), Some(first));
        assert!(gallery.fail_thumbnail(current.id, "another failure".into()));
        editor
            .execute(WorldCommand::SetAttribute {
                id: first,
                path: "/material/base".into(),
                value: json!(0.3),
                frame: 0.0,
            })
            .unwrap();
        gallery.refresh(&editor, 1.0).unwrap();
        assert!(
            gallery
                .entries
                .iter()
                .find(|entry| entry.id == first)
                .unwrap()
                .error
                .is_none()
        );
        let changed = gallery.next_thumbnail().unwrap();
        assert_eq!(pending_node(&gallery, changed.id), Some(first));
        gallery.fail_thumbnail(changed.id, "failed again".into());
        gallery.invalidate();
        gallery.refresh(&editor, 1.0).unwrap();
        assert!(
            gallery.entries.iter().all(|entry| entry.error.is_none()),
            "explicit reload clears failures"
        );
        assert!(gallery.next_thumbnail().is_some());
    }

    #[test]
    fn idle_projection_preserves_storage_selection_and_authoring_revision() {
        let (editor, _, _) = world();
        let selection = editor.selection;
        let revision = editor.revision();
        let mut gallery = MaterialGallery::default();
        assert!(gallery.refresh(&editor, 0.0).unwrap());
        assert_eq!(gallery.entries.len(), 2);
        let storage = (
            gallery.entries.as_ptr(),
            gallery.entries.capacity(),
            gallery.scratch.capacity(),
        );
        for _ in 0..64 {
            assert!(!gallery.refresh(&editor, -0.0).unwrap());
        }
        assert_eq!(
            storage,
            (
                gallery.entries.as_ptr(),
                gallery.entries.capacity(),
                gallery.scratch.capacity()
            )
        );
        let name_pointer = gallery.entries[0].name.as_ptr();
        assert!(gallery.refresh(&editor, 1.0).unwrap());
        assert_eq!(gallery.entries[0].name.as_ptr(), name_pointer);
        assert_eq!(gallery.entries.as_ptr(), storage.0);
        assert_eq!(editor.selection, selection);
        assert_eq!(editor.revision(), revision);
        assert!(gallery.refresh(&editor, f64::NAN).is_err());
        assert_eq!(gallery.entries.len(), 2);
    }

    #[test]
    fn rename_and_unrelated_frames_keep_textures_but_material_edits_reject_pending_results() {
        let (mut editor, first, second) = world();
        let mut gallery = MaterialGallery::default();
        gallery.refresh(&editor, 0.0).unwrap();
        let ctx = egui::Context::default();
        let first_request = gallery.next_thumbnail().unwrap();
        assert_eq!(pending_node(&gallery, first_request.id), Some(first));
        let thumb = texture(&ctx);
        let texture_id = thumb.id();
        assert!(gallery.accept_thumbnail(first_request.id, thumb));
        let old = gallery.next_thumbnail().unwrap();
        assert_eq!(pending_node(&gallery, old.id), Some(second));
        editor
            .execute(WorldCommand::Rename {
                id: first,
                name: "Renamed".into(),
            })
            .unwrap();
        gallery.refresh(&editor, 12.0).unwrap();
        assert_eq!(gallery.entries[0].name, "Renamed");
        assert_eq!(gallery.entries[0].thumb.as_ref().unwrap().id(), texture_id);
        assert!(gallery.is_pending(old.id));
        editor
            .execute(WorldCommand::SetAttribute {
                id: second,
                path: "/material/base".into(),
                value: json!(0.25),
                frame: 0.0,
            })
            .unwrap();
        gallery.refresh(&editor, 12.0).unwrap();
        assert!(!gallery.is_pending(old.id));
        assert!(!gallery.accept_thumbnail(old.id, texture(&ctx)));
        let current = gallery.next_thumbnail().unwrap();
        assert_ne!(current.id, old.id);
        assert_eq!(pending_node(&gallery, current.id), Some(second));
        assert_eq!(
            current.material,
            editor.document.material(second, 12.0).unwrap()
        );
        gallery.retry_thumbnail(current.id);
        assert!(!gallery.is_pending(current.id));
        let next = gallery.next_thumbnail().unwrap();
        assert_eq!(pending_node(&gallery, next.id), Some(second));
    }

    #[test]
    fn deletion_document_replacement_and_animation_invalidate_only_matching_materials() {
        let (mut editor, first, second) = world();
        let mut gallery = MaterialGallery::default();
        gallery.refresh(&editor, 0.0).unwrap();
        let old = gallery.next_thumbnail().unwrap();
        editor.execute(WorldCommand::Delete(first)).unwrap();
        gallery.refresh(&editor, 0.0).unwrap();
        assert!(!gallery.is_pending(old.id));
        assert_eq!(gallery.entries.len(), 1);
        let pending = gallery.next_thumbnail().unwrap();
        let mut replacement = editor.document.clone();
        replacement.graph.id = NodeId::new().into();
        editor = WorldEditor::new(replacement);
        gallery.refresh(&editor, 0.0).unwrap();
        assert!(!gallery.is_pending(pending.id));
        let ctx = egui::Context::default();
        let ticket = gallery.next_thumbnail().unwrap();
        assert!(gallery.accept_thumbnail(ticket.id, texture(&ctx)));
        editor
            .execute(WorldCommand::Key {
                id: second,
                path: "/material/base".into(),
                frame: 0.0,
            })
            .unwrap();
        editor
            .execute(WorldCommand::SetAttribute {
                id: second,
                path: "/material/base".into(),
                value: json!(0.25),
                frame: 10.0,
            })
            .unwrap();
        gallery.refresh(&editor, 0.0).unwrap();
        assert!(
            gallery.entries[0].thumb.is_some(),
            "unchanged keyed value preserves texture"
        );
        gallery.refresh(&editor, 10.0).unwrap();
        assert!(
            gallery.entries[0].thumb.is_none(),
            "changed evaluated key invalidates texture"
        );
    }
}
