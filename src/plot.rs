//! Publication-quality 2D scientific plotting using `mplot` (`https://github.com/keejkrej/mplot-rs`).
//!
//! Provides diagnostic and assay visualizations for Picasso SMLM workflows (`plot_nena`,
//! `plot_frc`, `plot_drift`, `plot_rendered_image`) and Lisca binding assays (`plot_lnp_binding`).

use crate::binding::{BindingSpot, FrameSpotCount};
use crate::frc::FrcResult;
use crate::postprocess::NenaResult;
use crate::types::{PicassoError, Result};
use marray::np::{Array1, Array2};
use mplot::prelude::*;
use std::path::Path;

/// Saves a NeNA (Nearest-Neighbor Analysis) localization precision diagnostic plot to `output_path`.
pub fn plot_nena(
    nena: &NenaResult,
    pixelsize_nm: f64,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let d_nm: Vec<f64> = nena
        .distances
        .iter()
        .map(|&d| d * pixelsize_nm)
        .collect();
    let pdf: Vec<f64> = nena
        .probability_density
        .iter()
        .map(|&p| p / pixelsize_nm.max(1e-6))
        .collect();
    let fit: Vec<f64> = nena
        .fitted_curve
        .iter()
        .map(|&f| f / pixelsize_nm.max(1e-6))
        .collect();
    let nena_nm = nena.nena_px * pixelsize_nm;

    let fig = Figure::builder()
        .size(Size::inches(6.5, 4.5))
        .panel(GridPos::new(1, 1, 1), |p| {
            p.scatter(
                &d_nm,
                &pdf,
                ScatterStyle::new()
                    .color(Color::hex("#1f77b4"))
                    .size(4.5)
                    .label("Empirical NND"),
            );
            p.line(
                &d_nm,
                &fit,
                LineStyle::new()
                    .color(Color::hex("#d62728"))
                    .width(2.0)
                    .label(format!("NeNA fit ({nena_nm:.2} nm)")),
            );
            p.axes(
                AxesStyle::new()
                    .title(format!("NeNA Localization Precision: {nena_nm:.2} nm"))
                    .x_label("Adjacent-frame distance (nm)")
                    .y_label("Probability density (1/nm)")
                    .legend(LegendStyle::show().loc(LegendLoc::UpperRight)),
            );
        })
        .build()
        .map_err(|e| PicassoError::Plot(e.to_string()))?;

    fig.save(output_path, SaveOptions::new().dpi(150))
        .map_err(|e| PicassoError::Plot(e.to_string()))
}

/// Saves a Fourier Ring Correlation (FRC) resolution curve plot to `output_path`.
pub fn plot_frc(frc: &FrcResult, output_path: impl AsRef<Path>) -> Result<()> {
    let thresh_line = vec![frc.threshold; frc.spatial_frequencies.len()];

    let fig = Figure::builder()
        .size(Size::inches(6.5, 4.5))
        .panel(GridPos::new(1, 1, 1), |p| {
            p.scatter(
                &frc.spatial_frequencies,
                &frc.frc_curve,
                ScatterStyle::new()
                    .color(Color::hex("#7f7f7f"))
                    .size(3.5)
                    .label("Raw FRC"),
            );
            p.line(
                &frc.spatial_frequencies,
                &frc.frc_smooth,
                LineStyle::new()
                    .color(Color::hex("#1f77b4"))
                    .width(2.0)
                    .label("Smoothed FRC"),
            );
            p.line(
                &frc.spatial_frequencies,
                &thresh_line,
                LineStyle::new()
                    .color(Color::hex("#d62728"))
                    .width(1.5)
                    .label("1/7 Threshold"),
            );
            p.axes(
                AxesStyle::new()
                    .title(format!("FRC Resolution: {:.1} nm", frc.resolution_nm))
                    .x_label("Spatial frequency (1/nm)")
                    .y_label("Fourier Ring Correlation")
                    .legend(LegendStyle::show().loc(LegendLoc::UpperRight)),
            );
        })
        .build()
        .map_err(|e| PicassoError::Plot(e.to_string()))?;

    fig.save(output_path, SaveOptions::new().dpi(150))
        .map_err(|e| PicassoError::Plot(e.to_string()))
}

/// Saves a 2D drift trajectory plot (`drift_x`, `drift_y` vs frame) to `output_path`.
pub fn plot_drift(
    drift_x: &Array1<f64>,
    drift_y: &Array1<f64>,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let n = drift_x.len().min(drift_y.len());
    let frames: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let dx: Vec<f64> = drift_x.iter().take(n).copied().collect();
    let dy: Vec<f64> = drift_y.iter().take(n).copied().collect();

    let fig = Figure::builder()
        .size(Size::inches(6.5, 4.5))
        .panel(GridPos::new(1, 1, 1), |p| {
            p.line(
                &frames,
                &dx,
                LineStyle::new()
                    .color(Color::hex("#1f77b4"))
                    .width(2.0)
                    .label("Drift X (px)"),
            );
            p.line(
                &frames,
                &dy,
                LineStyle::new()
                    .color(Color::hex("#ff7f0e"))
                    .width(2.0)
                    .label("Drift Y (px)"),
            );
            p.axes(
                AxesStyle::new()
                    .title("Estimated Sample Drift Trajectory")
                    .x_label("Frame")
                    .y_label("Drift (pixels)")
                    .legend(LegendStyle::show().loc(LegendLoc::UpperLeft)),
            );
        })
        .build()
        .map_err(|e| PicassoError::Plot(e.to_string()))?;

    fig.save(output_path, SaveOptions::new().dpi(150))
        .map_err(|e| PicassoError::Plot(e.to_string()))
}

/// Saves a rendered 2D super-resolution image or microscopy frame using `mplot`'s `Inferno` colormap.
pub fn plot_rendered_image(
    image: &Array2<f64>,
    title: impl Into<String>,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let (h, w) = image.dim();
    let flat: Vec<f64> = image.iter().copied().collect();
    let title_str = title.into();

    let fig = Figure::builder()
        .size(Size::inches(6.0, 6.0))
        .panel(GridPos::new(1, 1, 1), |p| {
            p.image(
                flat,
                w,
                h,
                ImageStyle::new().colormap(Colormap::Inferno).colorbar(true),
            );
            p.axes(
                AxesStyle::new()
                    .title(title_str)
                    .x_label("X (sub-pixels)")
                    .y_label("Y (sub-pixels)"),
            );
        })
        .build()
        .map_err(|e| PicassoError::Plot(e.to_string()))?;

    fig.save(output_path, SaveOptions::new().dpi(150))
        .map_err(|e| PicassoError::Plot(e.to_string()))
}

/// Saves a 2-panel Lisca binding assay summary plot (per-frame & cumulative binding counts +
/// single-particle intensity distribution) using `mplot` (`lisca-binding-assay/src/binding/plot_lnp.py`).
pub fn plot_lnp_binding(
    counts: &[FrameSpotCount],
    spots: &[BindingSpot],
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let times: Vec<f64> = counts.iter().map(|c| c.time as f64).collect();
    let per_frame: Vec<f64> = counts.iter().map(|c| c.spot_count as f64).collect();
    let cumulative: Vec<f64> = counts.iter().map(|c| c.cumulative_count as f64).collect();
    let mut intensities: Vec<f64> = spots
        .iter()
        .map(|s| s.intensity)
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if intensities.is_empty() {
        intensities.push(0.0);
    }

    let fig = Figure::builder()
        .size(Size::inches(11.0, 4.5))
        .panel(GridPos::new(1, 2, 1), |p| {
            if !times.is_empty() {
                p.line(
                    &times,
                    &per_frame,
                    LineStyle::new()
                        .color(Color::hex("#1f77b4"))
                        .width(2.0)
                        .label("Per-frame spots"),
                );
                p.line(
                    &times,
                    &cumulative,
                    LineStyle::new()
                        .color(Color::hex("#2ca02c"))
                        .width(2.0)
                        .label("Cumulative binding events"),
                );
            }
            p.axes(
                AxesStyle::new()
                    .title("Single-Particle Binding Kinetics")
                    .x_label("Frame / Timepoint")
                    .y_label("Spot Count")
                    .legend(LegendStyle::show().loc(LegendLoc::UpperLeft)),
            );
        })
        .panel(GridPos::new(1, 2, 2), |p| {
            p.histogram(
                &intensities,
                HistStyle::new().bins(25).color(Color::hex("#ff7f0e")),
            );
            p.axes(
                AxesStyle::new()
                    .title("Fitted Gaussian Amplitude (intens_A)")
                    .x_label("Spot Intensity (photons)")
                    .y_label("Count"),
            );
        })
        .build()
        .map_err(|e| PicassoError::Plot(e.to_string()))?;

    fig.save(output_path, SaveOptions::new().dpi(150))
        .map_err(|e| PicassoError::Plot(e.to_string()))
}
