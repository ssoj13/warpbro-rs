//! File names for everything WarpBro writes (snapshots, exports, CLI renders, fixtures, scenes,
//! templates): one rule for the stem, one for sequence numbering, one validation of a typed
//! name. Names keep their script - NTFS and every target filesystem store Unicode names; only
//! what a filesystem forbids changes.

/// Windows reserves these names (and `name.anything`) for devices in every directory; the
/// digit may also be a superscript ¹ ² ³, which `char::is_alphanumeric` keeps.
const DEVICES: [&str; 6] = ["con", "prn", "aux", "nul", "conin$", "conout$"];
const NUMBERED_DEVICES: [&str; 2] = ["com", "lpt"];
const DEVICE_DIGITS: &str = "0123456789¹²³";

/// The suffix of a scene (and template) file.
pub const SCENE_SUFFIX: &str = "frac.json";

/// The digits of a sequence frame number (`stem.000042.suffix`).
const FRAME_DIGITS: usize = 6;

/// The longest stem in UTF-16 units: NTFS allows 255 per name, which leaves room for a
/// sequence number (`.000000`) and the longest suffix (`.display.exr`).
const MAX_STEM_UNITS: usize = 200;

/// The comparison key of a name (templates) and the base of a file stem: lowercase letters
/// and digits of any script, every other run of characters one `-`. Everything a filesystem
/// forbids (`<>:"/\|?*`, control characters) is not alphanumeric, so it never survives.
pub fn slug(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_alphanumeric() {
            s.extend(c.to_lowercase());
        } else if !s.ends_with('-') {
            s.push('-');
        }
    }
    s.trim_matches('-').to_string()
}

/// A file stem for a scene / descriptor name: its slug ("Медная турбина" -> "медная-турбина"),
/// a device name with `_` appended, cut to `MAX_STEM_UNITS`; "untitled" only when the name has
/// no letter or digit at all.
pub fn stem(name: &str) -> String {
    let slug = slug(name);
    if slug.is_empty() {
        return "untitled".into();
    }
    // Truncation can end the stem on a separator.
    let stem = truncate(&slug, MAX_STEM_UNITS)
        .trim_end_matches('-')
        .to_string();
    if is_device(&stem) {
        format!("{stem}_")
    } else {
        stem
    }
}

/// Validate a typed file name (the export panel's Name) as it will be written: not empty, no
/// character a filesystem forbids, no trailing dot or space (Windows drops them), not a
/// device name, at most `MAX_STEM_UNITS` UTF-16 units.
pub fn check(name: &str) -> Result<(), String> {
    if name.trim().is_empty() {
        return Err("The name is empty".into());
    }
    if let Some(c) = name
        .chars()
        .find(|&c| c.is_control() || r#"<>:"/\|?*"#.contains(c))
    {
        return Err(format!(
            "The name contains {c:?}, which file names cannot hold"
        ));
    }
    if name.ends_with(['.', ' ']) {
        return Err("The name cannot end with a dot or a space".into());
    }
    if is_device(name) {
        return Err(format!("\"{name}\" is a Windows device name"));
    }
    if name.encode_utf16().count() > MAX_STEM_UNITS {
        return Err(format!(
            "The name is longer than {MAX_STEM_UNITS} characters"
        ));
    }
    Ok(())
}

/// The file name of `stem` with `suffix` (`png`, `pq.png`, `display.exr`, ...): one file, or
/// frame `number` of a sequence as `stem.000042.suffix` - for every sequence writer.
pub fn frame_file(stem: &str, number: Option<u32>, suffix: &str) -> String {
    match number {
        Some(n) => format!("{stem}.{n:0FRAME_DIGITS$}.{suffix}"),
        None => format!("{stem}.{suffix}"),
    }
}

/// The ffmpeg image-sequence pattern of `frame_file`'s numbering (`dir/stem.%06d.suffix`) for
/// the frames of `stem` in `dir`, every literal `%` of the path escaped.
pub fn sequence_pattern(dir: &std::path::Path, stem: &str, suffix: &str) -> String {
    let base = dir.join(stem).display().to_string().replace('%', "%%");
    format!("{base}.%0{FRAME_DIGITS}d.{}", suffix.replace('%', "%%"))
}

/// A device name, also with an extension (`con`, `CON.txt`, `com1`, `lpt¹`).
fn is_device(name: &str) -> bool {
    let base = name
        .split('.')
        .next()
        .unwrap_or(name)
        .trim_end()
        .to_lowercase();
    DEVICES.contains(&base.as_str())
        || NUMBERED_DEVICES.iter().any(|prefix| {
            base.strip_prefix(prefix).is_some_and(|rest| {
                let mut chars = rest.chars();
                matches!((chars.next(), chars.next()), (Some(d), None) if DEVICE_DIGITS.contains(d))
            })
        })
}

/// `s` cut to at most `units` UTF-16 units on a character boundary.
fn truncate(s: &str, units: usize) -> String {
    let mut used = 0;
    s.chars()
        .take_while(|c| {
            used += c.len_utf16();
            used <= units
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Names keep their script; only what a filesystem forbids or a device name blocks changes.
    #[test]
    fn stems_keep_unicode_and_avoid_device_names() {
        assert_eq!(slug("Медная  Турбина №2"), "медная-турбина-2");
        assert_eq!(slug(r#"A<b>:c/d\e|f?g*h"i"#), "a-b-c-d-e-f-g-h-i");
        assert_eq!(stem("Медная турбина"), "медная-турбина");
        assert_eq!(stem("日本語 シーン"), "日本語-シーン");
        assert_eq!(stem("CON"), "con_");
        assert_eq!(stem("COM¹"), "com¹_");
        assert_eq!(stem("lpt0"), "lpt0_");
        assert_eq!(stem("console"), "console");
        assert_eq!(stem("  ** "), "untitled");
        assert_eq!(stem("Copper Turbine (KIFS)"), "copper-turbine-kifs");
        assert_eq!(
            stem(&"я".repeat(300)).encode_utf16().count(),
            MAX_STEM_UNITS
        );
    }

    /// A typed name is refused for exactly what the filesystem would refuse.
    #[test]
    fn typed_names_are_checked_like_the_filesystem() {
        assert!(check("Медная турбина").is_ok());
        assert!(check("frame").is_ok());
        for bad in [
            "",
            "  ",
            "a/b",
            "a:b",
            "tab\tname",
            "dot.",
            "space ",
            "NUL",
            "con.txt",
            "COM¹",
            "CONIN$",
        ] {
            assert!(check(bad).is_err(), "{bad:?}");
        }
        assert!(check(&"x".repeat(MAX_STEM_UNITS + 1)).is_err());
    }

    /// One numbering for every sequence writer: the number before the whole suffix.
    #[test]
    fn sequences_number_before_the_suffix() {
        assert_eq!(frame_file("shot", None, "pq.png"), "shot.pq.png");
        assert_eq!(frame_file("shot", Some(42), "pq.png"), "shot.000042.pq.png");
        assert_eq!(
            frame_file("shot", Some(7), "display.exr"),
            "shot.000007.display.exr"
        );
        // Both the folder's and the name's `%` are literal; only the frame number is a field.
        let pattern = sequence_pattern(std::path::Path::new("50%"), "50% grey", "pq.png");
        assert_eq!(pattern.matches("%%").count(), 2, "{pattern}");
        assert!(pattern.ends_with("50%% grey.%06d.pq.png"), "{pattern}");
    }
}
