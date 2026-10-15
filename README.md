# `picasso-rs` (`picasso`)

Pure-Rust port of [**jungmannlab/picasso**](https://github.com/jungmannlab/picasso) (v0.11.3) for single-molecule localization microscopy (SMLM), single-particle detection, 2D/3D Gaussian & cubic spline PSF fitting, post-processing, clustering, super-resolution rendering, and non-deep-learning single-particle binding assays.

Built natively on:
- [**`marray-rs`** (`marray`)](https://github.com/keejkrej/marray-rs) for NumPy/SciPy/scikit-learn scientific computing (`np`, `sp::ndimage`, `sp::spatial::{KDTree, ConvexHull}`, `sp::optimize`, `sp::linalg`, `sp::fft`, `sp::special::erf`, `sklearn::cluster::{DBSCAN, KMeans}`).
- [**`mplot-rs`** (`mplot`)](https://github.com/keejkrej/mplot-rs) for Matplotlib-inspired publication-quality 2D scientific plotting.

Designed to be imported directly as a Rust library in [**`lisca`**](https://github.com/keejkrej/lisca) and [**`lisca-binding-assay`**](https://github.com/keejkrej/lisca-binding-assay) as a fast, zero-Python/zero-PyTorch non-deep-learning MVP for single-particle LNP membrane-binding assays.

---

## Installation

Add `picasso` to your `Cargo.toml`:

```toml
[dependencies]
picasso = { git = "https://github.com/keejkrej/picasso-rs" }
```

---

## Module Mapping (`jungmannlab/picasso` $\rightarrow$ `picasso-rs`)

| Python `picasso` Module | Rust `picasso` Module | Key Capabilities |
|---|---|---|
| `picasso.localize` | `picasso::localize` | Local maxima (`local_maxima`), radial net gradient (`net_gradient`, `identify_in_image`, `identify`), rolling temporal median & Gaussian pre-filtering, ADU-to-photon spot cutting (`get_spots`), and end-to-end `fit` / `localize`. |
| `picasso.fitting.seeds` | `picasso::fitting::seeds` | 1D central-profile & 2D second-moment tensor initial parameter estimation (`initial_parameters_gauss`, `initial_parameters_spline`). |
| `picasso.fitting.gaussfit` | `picasso::fitting::gaussfit` | Gpufit-ported Levenberg-Marquardt 2D Gaussian fitter for `Spherical` (5p), `Elliptic` (6p), and `Rotated` (7p) models under Least Squares (`LeastSquares`) and Poisson / sCMOS shifted-Poisson MLE (`Mle`). |
| `picasso.fitting.precision` | `picasso::fitting::precision` | Mortensen et al. (2010) closed-form localization precision (`localization_precision`), LSQ width uncertainty, and exact $(6\times 6)$ / $(7\times 7)$ Fisher Information Matrix inversion (`gauss_crlb_single`, `gauss_crlb_batch`). |
| `picasso.gaussmle` | `picasso::fitting::gaussmle` | Legacy Smith et al. (2010) pixel-integrated `erf` Newton-Raphson MLE (`MleMethod::{Fixed, Sigma, SigmaYx}`) with analytical CRLBs. |
| `picasso.gausslq` | `picasso::fitting::gausslq` | Direct 2D elliptical Gaussian least-squares fitting (`fit_spot`, `fit_spots`, `locs_from_fits`). |
| `picasso.avgroi` | `picasso::avgroi` | Fast ROI photon integration (`fit_spot`, `fit_spots`, `locs_from_fits`). |
| `picasso.zfit` | `picasso::zfit` | 6th-order polynomial 3D astigmatism calibration (`calibrate_z`), 1D axial minimization (`fit_z`), and tail filtering (`filter_tails`). |
| `picasso.spline` & `fitting.splinefit` | `picasso::spline`, `picasso::fitting::splinefit` | 2D bicubic & 3D tricubic Catmull-Rom PSF spline calibration (`fit_spline_coefficients_2d`, `fit_spline_coefficients_3d`) and 3D Levenberg-Marquardt spline fitting (`fit_spots_spline`). |
| `picasso.scmos` | `picasso::scmos` | Per-pixel sCMOS dark offset, readout variance, and conversion gain calibration (`calibrate_scmos`, `validate_calibration`). |
| `picasso.postprocess` & `imageprocess` | `picasso::postprocess`, `picasso::imageprocess` | Spatial block indexing (`SpatialBlocks`), ROI picking (`picked_locs`, `pick_similar`), trajectory linking across dark frames (`link`), dark-time kinetics (`compute_dark_times`), NeNA precision (`nena`), pair correlation (`pair_correlation`), RCC drift correction (`undrift`, `rcc`), multi-channel alignment (`align`), and group statistics (`groupprops`). |
| `picasso.aim` | `picasso::aim` | Adaptive Intersection Maximization (`aim`) point-cloud drift correction. |
| `picasso.frc` | `picasso::frc` | 2D Fourier Ring Correlation (`frc`, `frc_from_images`) super-resolution estimation at the $1/7$ threshold. |
| `picasso.clusterer` & `g5m` | `picasso::clusterer`, `picasso::g5m` | SMLM local-density maxima clusterer (`cluster`), DBSCAN (`dbscan`), blinking frame analysis (`frame_analysis`), cluster center & 2D `ConvexHull` aggregation (`find_cluster_centers`), and AIC-selected Gaussian Mixture Molecular Mapping (`g5m`). |
| `picasso.render` | `picasso::render` | 2D & 3D super-resolution rendering (`render`, `render_hist`, `render_gaussian`, `render_gaussian_iso`, `render_smooth`, `render_convolve`, `render_hist3d`, `render_gaussian3d`). |
| `picasso.simulate` | `picasso::simulate` | Synthetic SMLM / binding time-lapse movie generator (`simulate_movie`, `render_noiseless_frame`) with Poisson shot noise and readout noise. |
| *Lisca Binding Assay MVP* | `picasso::binding`, `picasso::plot` | Turnkey Spotiflow-compatible binding assay detection (`detect_spots_frame`, `detect_spots_u16_slice`, `detect_spots_stack`), cell-mask filtering (`filter_spots`), per-frame & cumulative binding counts (`spot_counts`), Hungarian multi-frame tracking & coalescence/merge event detection (`track_spots`), and `mplot` figures (`plot_lnp_binding`). |

---

## Quick Start: Non-Deep-Learning Binding Assay for `lisca` / `lisca-binding-assay`

```rust
use picasso::prelude::*;

fn run_assay(movie: &marray::np::Array3<f64>) -> Result<()> {
    // 1. Detect and fit 2D Gaussian PSFs across all frames (no deep learning / PyTorch needed)
    let detect_cfg = BindingDetectConfig {
        min_gradient: 250.0,
        box_size: 7,
        fit_method: SpotFitMethod::GaussLq,
        gauss_model: GaussModel::Elliptic,
        gaussian_sigma: Some((1.0, 1.0)),
        camera_info: CameraInfo::new(100.0, 1.0, 1.0),
        pos: "Pos0".to_string(),
        roi: "crop_0".to_string(),
        parallel: true,
    };
    let raw_spots = detect_spots_stack(movie, &detect_cfg)?;

    // 2. Filter by fitted amplitude, FWHM, ellipticity, and optional cell segmentation mask
    let filter_cfg = BindingFilterConfig {
        min_intensity: Some(50.0),
        min_fwhm: Some(1.2),
        max_fwhm: Some(5.0),
        max_ellipticity: Some(0.5),
        ..Default::default()
    };
    let clean_spots = filter_spots(&raw_spots, &filter_cfg, None);

    // 3. Compute per-frame and cumulative new-binding-event counts
    let counts = spot_counts(&clean_spots, None, 5.0);

    // 4. Track spots across frames (Hungarian matcher) and detect coalescence / merge events
    let (tracked, merges, summaries) = track_spots(&clean_spots, &SpotTrackConfig::default());

    // 5. Render binding kinetics & intensity histogram via mplot
    plot_lnp_binding(&counts, &clean_spots, "binding_summary.png")?;
    Ok(())
}
```
