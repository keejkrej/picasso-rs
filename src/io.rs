//! File I/O for microscopy TIFF stacks, localizations (CSV/JSON), binding assay tables,
//! and `.yaml` metadata sidecars.
//!
//! Ported from `picasso.io`.

use crate::binding::{BindingSpot, FrameSpotCount, MergeEvent, TrackSummary, TrackedSpot};
use crate::types::{ClusterCenter, Localization, Metadata, PicassoError, Result};
use marray::np::{Array2, Array3};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{TiffEncoder, colortype};

/// Loads a 2D or 3D multi-page TIFF file into an `Array3<f64>` of shape `(n_frames, height, width)`
/// (`picasso.io.load_tif`).
pub fn load_tiff_stack(path: impl AsRef<Path>) -> Result<(Array3<f64>, Metadata)> {
    let file = File::open(path.as_ref())?;
    let mut decoder = Decoder::new(BufReader::new(file))
        .map_err(|e| PicassoError::InvalidParameter(format!("TIFF decode error: {e}")))?;

    let mut frames_data: Vec<Vec<f64>> = Vec::new();
    let mut width = 0usize;
    let mut height = 0usize;

    loop {
        let (w, h) = decoder
            .dimensions()
            .map_err(|e| PicassoError::InvalidParameter(format!("TIFF dimensions error: {e}")))?;
        let w = w as usize;
        let h = h as usize;
        if frames_data.is_empty() {
            width = w;
            height = h;
        } else if w != width || h != height {
            return Err(PicassoError::DimensionMismatch(
                "Inconsistent frame dimensions in multi-page TIFF".to_string(),
            ));
        }

        let res = decoder
            .read_image()
            .map_err(|e| PicassoError::InvalidParameter(format!("TIFF read error: {e}")))?;
        let page: Vec<f64> = match res {
            DecodingResult::U8(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::U16(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::U32(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::U64(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::I8(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::I16(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::I32(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::I64(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::F32(v) => v.into_iter().map(|x| x as f64).collect(),
            DecodingResult::F64(v) => v,
            DecodingResult::F16(v) => v.into_iter().map(f64::from).collect(),
        };
        if page.len() != width * height {
            return Err(PicassoError::DimensionMismatch(format!(
                "Expected {} grayscale pixels per frame, got {}",
                width * height,
                page.len()
            )));
        }
        frames_data.push(page);

        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .map_err(|e| PicassoError::InvalidParameter(format!("TIFF next_image error: {e}")))?;
    }

    let n_frames = frames_data.len();
    let flat: Vec<f64> = frames_data.into_iter().flatten().collect();
    let movie = Array3::from_shape_vec((n_frames, height, width), flat)
        .map_err(|e| PicassoError::DimensionMismatch(e.to_string()))?;

    let info = Metadata {
        frames: Some(n_frames),
        height: Some(height),
        width: Some(width),
        data_type: Some("f64".to_string()),
        generated_by: Some("picasso-rs".to_string()),
        ..Default::default()
    };

    Ok((movie, info))
}

/// Saves a 3D movie stack `(n_frames, height, width)` as a 16-bit grayscale multi-page TIFF file.
pub fn save_tiff_stack_u16(path: impl AsRef<Path>, movie: &Array3<f64>) -> Result<()> {
    let file = File::create(path.as_ref())?;
    let mut encoder = TiffEncoder::new(BufWriter::new(file))
        .map_err(|e| PicassoError::InvalidParameter(format!("TIFF encoder error: {e}")))?;
    let shape = movie.shape();
    let (n_frames, h, w) = (shape[0], shape[1], shape[2]);

    for t in 0..n_frames {
        let mut page = Vec::with_capacity(h * w);
        for iy in 0..h {
            for ix in 0..w {
                let v = movie[[t, iy, ix]].round().clamp(0.0, u16::MAX as f64) as u16;
                page.push(v);
            }
        }
        encoder
            .write_image::<colortype::Gray16>(w as u32, h as u32, &page)
            .map_err(|e| PicassoError::InvalidParameter(format!("TIFF write error: {e}")))?;
    }

    Ok(())
}

/// Saves a 2D image `(height, width)` as a 16-bit grayscale single-page TIFF file.
pub fn save_tiff_frame_u16(path: impl AsRef<Path>, frame: &Array2<f64>) -> Result<()> {
    let (h, w) = frame.dim();
    let mut stack = Array3::<f64>::zeros((1, h, w));
    stack.slice_mut(marray::np::s![0, .., ..]).assign(frame);
    save_tiff_stack_u16(path, &stack)
}

/// Generic CSV writer for any serializable record slice.
pub fn save_csv<T: Serialize>(path: impl AsRef<Path>, records: &[T]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    for rec in records {
        wtr.serialize(rec)?;
    }
    wtr.flush()?;
    Ok(())
}

/// Generic CSV reader for any deserializable record type.
pub fn load_csv<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<Vec<T>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut out = Vec::new();
    for row in rdr.deserialize() {
        out.push(row?);
    }
    Ok(out)
}

/// Saves [`Localization`] records to a CSV file (`picasso.io.save_locs`).
pub fn save_locs_csv(path: impl AsRef<Path>, locs: &[Localization]) -> Result<()> {
    save_csv(path, locs)
}

/// Loads [`Localization`] records from a CSV file (`picasso.io.load_locs`).
pub fn load_locs_csv(path: impl AsRef<Path>) -> Result<Vec<Localization>> {
    load_csv(path)
}

/// Saves [`BindingSpot`] records to a CSV file compatible with `lisca-binding-assay`.
pub fn save_binding_spots_csv(path: impl AsRef<Path>, spots: &[BindingSpot]) -> Result<()> {
    save_csv(path, spots)
}

/// Loads [`BindingSpot`] records from a CSV file.
pub fn load_binding_spots_csv(path: impl AsRef<Path>) -> Result<Vec<BindingSpot>> {
    load_csv(path)
}

/// Saves [`FrameSpotCount`] records to a CSV file.
pub fn save_spot_counts_csv(path: impl AsRef<Path>, counts: &[FrameSpotCount]) -> Result<()> {
    save_csv(path, counts)
}

/// Saves [`TrackedSpot`] records to a CSV file.
pub fn save_tracked_spots_csv(path: impl AsRef<Path>, tracked: &[TrackedSpot]) -> Result<()> {
    save_csv(path, tracked)
}

/// Saves [`MergeEvent`] records to a CSV file.
pub fn save_merge_events_csv(path: impl AsRef<Path>, merges: &[MergeEvent]) -> Result<()> {
    save_csv(path, merges)
}

/// Saves [`TrackSummary`] records to a CSV file.
pub fn save_track_summaries_csv(path: impl AsRef<Path>, summaries: &[TrackSummary]) -> Result<()> {
    save_csv(path, summaries)
}

/// Saves [`ClusterCenter`] records to a CSV file.
pub fn save_cluster_centers_csv(path: impl AsRef<Path>, centers: &[ClusterCenter]) -> Result<()> {
    save_csv(path, centers)
}

/// Saves multi-document Picasso `.yaml` metadata (`picasso.io.save_info`).
pub fn save_info(path: impl AsRef<Path>, info: &[Metadata]) -> Result<()> {
    let mut f = File::create(path)?;
    for (idx, doc) in info.iter().enumerate() {
        if idx > 0 {
            writeln!(f, "---")?;
        }
        let s = serde_yaml::to_string(doc)?;
        f.write_all(s.as_bytes())?;
    }
    Ok(())
}

/// Loads multi-document Picasso `.yaml` metadata (`picasso.io.load_info`).
pub fn load_info(path: impl AsRef<Path>) -> Result<Vec<Metadata>> {
    let mut f = File::open(path)?;
    let mut content = String::new();
    f.read_to_string(&mut content)?;
    let mut docs = Vec::new();
    for doc_str in content.split("\n---") {
        let trimmed = doc_str.trim();
        if !trimmed.is_empty() {
            let meta: Metadata = serde_yaml::from_str(trimmed)?;
            docs.push(meta);
        }
    }
    Ok(docs)
}
