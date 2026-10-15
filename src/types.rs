//! Core data structures, configuration types, and error definitions for `picasso`.

use marray::np::Array2;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

/// Unified error type for `picasso`.
#[derive(Debug, Error)]
pub enum PicassoError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Numerical error: {0}")]
    Numerical(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("Plot error: {0}")]
    Plot(String),
}

impl From<String> for PicassoError {
    fn from(s: String) -> Self {
        Self::InvalidParameter(s)
    }
}

impl From<&str> for PicassoError {
    fn from(s: &str) -> Self {
        Self::InvalidParameter(s.to_string())
    }
}

/// Alias for `std::result::Result<T, PicassoError>`.
pub type Result<T> = std::result::Result<T, PicassoError>;

/// Camera baseline (ADU offset): either a scalar value or a 2D per-pixel sCMOS offset map.
#[derive(Debug, Clone)]
pub enum Baseline {
    Scalar(f64),
    Map(Array2<f64>),
}

impl Default for Baseline {
    fn default() -> Self {
        Self::Scalar(100.0)
    }
}

impl From<f64> for Baseline {
    fn from(v: f64) -> Self {
        Self::Scalar(v)
    }
}

/// Camera sensitivity (photoelectrons per ADU): either scalar or 2D per-pixel sensitivity map (`1 / gain_map`).
#[derive(Debug, Clone)]
pub enum Sensitivity {
    Scalar(f64),
    Map(Array2<f64>),
}

impl Default for Sensitivity {
    fn default() -> Self {
        Self::Scalar(1.0)
    }
}

impl From<f64> for Sensitivity {
    fn from(v: f64) -> Self {
        Self::Scalar(v)
    }
}

/// Camera calibration and acquisition configuration for photon conversion and noise modeling.
#[derive(Debug, Clone)]
pub struct CameraInfo {
    /// Baseline offset in ADU (scalar or 2D sCMOS offset map).
    pub baseline: Baseline,
    /// Conversion factor from ADU to photoelectrons (`e- / ADU`), or 2D map.
    pub sensitivity: Sensitivity,
    /// EMCCD electron multiplication gain (`1` for standard sCMOS / CMOS cameras).
    pub gain: f64,
    /// Quantum efficiency (`0.0 < qe <= 1.0`, preserved for legacy compatibility; defaults to `1.0`).
    pub qe: f64,
    /// Optional per-pixel readout noise variance map in ADU^2 (for sCMOS MLE/LSQ fitting).
    pub variance_map: Option<Array2<f64>>,
}

impl Default for CameraInfo {
    fn default() -> Self {
        Self {
            baseline: Baseline::Scalar(0.0),
            sensitivity: Sensitivity::Scalar(1.0),
            gain: 1.0,
            qe: 1.0,
            variance_map: None,
        }
    }
}

impl CameraInfo {
    /// Creates a simple scalar camera configuration.
    pub fn new(baseline: f64, sensitivity: f64, gain: f64) -> Self {
        Self {
            baseline: Baseline::Scalar(baseline),
            sensitivity: Sensitivity::Scalar(sensitivity),
            gain: gain.max(1.0),
            qe: 1.0,
            variance_map: None,
        }
    }

    /// Returns whether electron multiplication noise excess factor (2x variance) is active.
    pub fn is_em(&self) -> bool {
        self.gain > 1.0
    }
}

/// Candidate spot identified by local-maximum + net-gradient thresholding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Identification {
    /// Frame index (0-based).
    pub frame: u32,
    /// Integer x (column) coordinate of local maximum.
    pub x: i32,
    /// Integer y (row) coordinate of local maximum.
    pub y: i32,
    /// Radial net gradient value inside the detection box.
    pub net_gradient: f32,
}

/// 2D Gaussian PSF model parameterization (matching `picasso.fitting.gaussfit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GaussModel {
    /// Symmetric 2D Gaussian: `[peak, x, y, s, bg]` (5 parameters).
    Spherical = 0,
    /// Axis-aligned elliptical 2D Gaussian: `[peak, x, y, sx, sy, bg]` (6 parameters).
    #[default]
    Elliptic = 1,
    /// Rotated elliptical 2D Gaussian: `[peak, x, y, sx, sy, bg, angle]` (7 parameters).
    Rotated = 2,
}

impl GaussModel {
    /// Number of fitted parameters in the model.
    pub const fn n_params(self) -> usize {
        match self {
            Self::Spherical => 5,
            Self::Elliptic => 6,
            Self::Rotated => 7,
        }
    }
}

/// Estimator objective for PSF fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FitEstimator {
    /// Unweighted least squares (LSE).
    #[default]
    LeastSquares,
    /// Poisson / sCMOS shifted-Poisson maximum likelihood estimation (MLE).
    Mle,
}

/// High-level fitting algorithm selection (matching `picasso.localize.fit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SpotFitMethod {
    /// Gpufit-ported Levenberg-Marquardt Least Squares 2D Gaussian (`gaussfit`, LSE).
    #[default]
    GaussLq,
    /// Gpufit-ported Levenberg-Marquardt Poisson MLE 2D Gaussian (`gaussfit`, MLE).
    GaussMle,
    /// Legacy Smith et al. (2010) pixel-integrated erf Newton-Raphson MLE (`gaussmle.py`).
    GaussMleIntegrated,
    /// Fast ROI-summed intensity at identified center (`avgroi.py`).
    AvgRoi,
}

/// Convergence status returned by iterative PSF fitters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
pub enum FitState {
    /// Converged within tolerance.
    Converged = 0,
    /// Reached maximum iterations without meeting tolerance.
    MaxIterationsExceeded = 1,
    /// Singular or non-finite Hessian / invalid model step.
    SingularHessian = 2,
}

/// Single-molecule / single-particle localization record (matching Picasso's `locs` recarray).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Localization {
    /// Frame number (0-based).
    pub frame: u32,
    /// Sub-pixel x coordinate in camera pixels (column).
    pub x: f64,
    /// Sub-pixel y coordinate in camera pixels (row).
    pub y: f64,
    /// Integrated spot signal in photons (above background).
    pub photons: f64,
    /// Gaussian PSF standard deviation along x (or major axis if rotated) in pixels.
    pub sx: f64,
    /// Gaussian PSF standard deviation along y (or minor axis if rotated) in pixels.
    pub sy: f64,
    /// Local background level in photons per pixel.
    pub bg: f64,
    /// Localization precision along x in pixels (Mortensen 2010 or CRLB).
    pub lpx: f64,
    /// Localization precision along y in pixels (Mortensen 2010 or CRLB).
    pub lpy: f64,
    /// Spot ellipticity `(max(sx, sy) - min(sx, sy)) / max(sx, sy)`.
    pub ellipticity: f64,
    /// Net gradient from spot identification.
    pub net_gradient: f64,
    /// Fitted peak amplitude in photons/pixel (optional, populated by `gaussfit`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak: Option<f64>,
    /// Fitted Gaussian rotation angle in radians (populated when `GaussModel::Rotated`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angle: Option<f64>,
    /// Axial coordinate `z` in nanometers (populated by 3D astigmatism or 3D spline fitting).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z: Option<f64>,
    /// Axial localization precision `lpz` in nanometers or pixels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lpz: Option<f64>,
    /// Goodness-of-fit chi-square or log-likelihood ratio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chi2: Option<f64>,
    /// Number of iterations used by the fitter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iterations: Option<u32>,
    /// Convergence state (`0 = Converged`, `1 = MaxIterations`, `2 = Singular`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fit_state: Option<i32>,
    /// Uncertainty in fitted `sx` (populated by `gaussfit`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sx_unc: Option<f64>,
    /// Uncertainty in fitted `sy` (populated by `gaussfit`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sy_unc: Option<f64>,
    /// Uncertainty in `photons` (populated by MLE CRLB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photons_unc: Option<f64>,
    /// Uncertainty in `bg` (populated by MLE CRLB).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bg_unc: Option<f64>,
    /// Trajectory / pick / cluster group ID (`-1` for ungrouped / noise).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<i32>,
    /// Cluster ID assigned by clustering algorithms (`-1` for noise).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster: Option<i32>,
    /// Binding event length in frames (populated by `link`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub len: Option<u32>,
    /// Number of constituent localizations merged into a linked binding event (`link`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Photon emission rate `photons / n` (populated by `link`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photon_rate: Option<f64>,
    /// Dark time in frames until the next binding event in the same pick (`compute_dark_times`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dark: Option<i32>,
    /// Local localization density within radius `r` (`compute_local_density`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub density: Option<u32>,
}

impl Default for Localization {
    fn default() -> Self {
        Self {
            frame: 0,
            x: 0.0,
            y: 0.0,
            photons: 0.0,
            sx: 1.0,
            sy: 1.0,
            bg: 0.0,
            lpx: 0.0,
            lpy: 0.0,
            ellipticity: 0.0,
            net_gradient: 0.0,
            peak: None,
            angle: None,
            z: None,
            lpz: None,
            chi2: None,
            iterations: None,
            fit_state: None,
            sx_unc: None,
            sy_unc: None,
            photons_unc: None,
            bg_unc: None,
            group: None,
            cluster: None,
            len: None,
            n: None,
            photon_rate: None,
            dark: None,
            density: None,
        }
    }
}

impl Localization {
    /// Computes the Full Width at Half Maximum (FWHM) in pixels from `sx` and `sy`:
    /// `FWHM = 2 * sqrt(2 * ln(2)) * sqrt(sx * sy)`.
    pub fn fwhm(&self) -> f64 {
        2.0 * (2.0 * std::f64::consts::LN_2).sqrt() * (self.sx.max(0.0) * self.sy.max(0.0)).sqrt()
    }

    /// Computes the fitted Gaussian peak amplitude above background in photons/pixel.
    pub fn amplitude(&self) -> f64 {
        self.peak.unwrap_or_else(|| {
            let denom = 2.0 * std::f64::consts::PI * self.sx.max(1e-6) * self.sy.max(1e-6);
            self.photons / denom
        })
    }

    /// Signal-to-noise-and-background ratio (`amplitude / max(bg, 1.0)`).
    pub fn snb(&self) -> f64 {
        self.amplitude() / self.bg.max(1.0)
    }
}

/// Aggregated cluster summary produced by `clusterer::find_cluster_centers`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterCenter {
    /// Cluster identifier.
    pub groups: i32,
    /// Mean frame index of localizations in cluster.
    pub frame: f64,
    /// Inverse-variance weighted center of mass x.
    pub x: f64,
    /// Inverse-variance weighted center of mass y.
    pub y: f64,
    /// Optional inverse-variance weighted center of mass z.
    pub z: Option<f64>,
    /// Standard deviation of x coordinates in the cluster.
    pub std_x: f64,
    /// Standard deviation of y coordinates in the cluster.
    pub std_y: f64,
    /// Optional standard deviation of z coordinates in the cluster.
    pub std_z: Option<f64>,
    /// Mean photons.
    pub photons: f64,
    /// Mean sx.
    pub sx: f64,
    /// Mean sy.
    pub sy: f64,
    /// Mean background.
    pub bg: f64,
    /// Combined cluster localization precision in x.
    pub lpx: f64,
    /// Combined cluster localization precision in y.
    pub lpy: f64,
    /// Optional combined cluster localization precision in z.
    pub lpz: Option<f64>,
    /// Mean ellipticity.
    pub ellipticity: f64,
    /// Mean net gradient.
    pub net_gradient: f64,
    /// Number of localizations in the cluster.
    pub n: u32,
    /// Number of distinct binding events (non-adjacent frame runs) in the cluster.
    pub n_events: u32,
    /// 2D Convex Hull area (in square pixels) or 3D volume.
    pub area: f64,
    /// Equivalent circle radius `sqrt(area / pi)` (2D) or sphere radius (3D).
    pub convexhull: f64,
}

/// ROI pick geometry for `postprocess::picked_locs` and `pick_similar`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PickShape {
    /// Circular pick regions with `(x, y)` centers and shared `diameter` (pixels).
    Circles {
        centers: Vec<(f64, f64)>,
        diameter: f64,
    },
    /// Rectangular pick regions defined by `(x_start, y_start)` -> `(x_end, y_end)` centerline and `width`.
    Rectangles {
        segments: Vec<((f64, f64), (f64, f64))>,
        width: f64,
    },
    /// Arbitrary closed polygon pick regions defined by vertex lists `Vec<(f64, f64)>`.
    Polygons { polygons: Vec<Vec<(f64, f64)>> },
}

/// Picasso `.yaml` metadata document entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(rename = "Byte Order", skip_serializing_if = "Option::is_none")]
    pub byte_order: Option<String>,
    #[serde(rename = "Data Type", skip_serializing_if = "Option::is_none")]
    pub data_type: Option<String>,
    #[serde(rename = "Frames", skip_serializing_if = "Option::is_none")]
    pub frames: Option<usize>,
    #[serde(rename = "Height", skip_serializing_if = "Option::is_none")]
    pub height: Option<usize>,
    #[serde(rename = "Width", skip_serializing_if = "Option::is_none")]
    pub width: Option<usize>,
    #[serde(rename = "Pixelsize", skip_serializing_if = "Option::is_none")]
    pub pixelsize: Option<f64>,
    #[serde(rename = "Generated by", skip_serializing_if = "Option::is_none")]
    pub generated_by: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}
