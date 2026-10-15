//! # `picasso-rs` (`picasso`)
//!
//! Pure-Rust port of [`jungmannlab/picasso`](https://github.com/jungmannlab/picasso) (v0.11.3)
//! for single-molecule localization microscopy (SMLM), single-particle detection, 2D/3D Gaussian
//! and cubic spline PSF fitting, post-processing, clustering, super-resolution rendering, and
//! non-deep-learning single-particle binding assays for [`lisca`](https://github.com/keejkrej/lisca)
//! and [`lisca-binding-assay`](https://github.com/keejkrej/lisca-binding-assay).
//!
//! Built on [`marray`](https://github.com/keejkrej/marray-rs) and
//! [`mplot`](https://github.com/keejkrej/mplot-rs).

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

pub use marray;
pub use mplot;

pub mod aim;
pub mod average;
pub mod avgroi;
pub mod binding;
pub mod clusterer;
pub mod design;
pub mod fitting;
pub mod frc;
pub mod g5m;
pub mod imageprocess;
pub mod io;
pub mod lib_utils;
pub mod localize;
pub mod nanotron;
pub mod plot;
pub mod postprocess;
pub mod render;
pub mod scmos;
pub mod simulate;
pub mod spinna;
pub mod spline;
pub mod toroidal_unwrap;
pub mod types;
pub mod zfit;

// Top-level convenience re-exports matching `picasso.*`
pub use fitting::gaussfit;
pub use fitting::gausslq;
pub use fitting::gaussmle;
pub use fitting::precision;
pub use fitting::seeds;
pub use fitting::splinefit;

pub use types::{
    Baseline, CameraInfo, ClusterCenter, FitEstimator, FitState, GaussModel, Identification,
    Localization, Metadata, PicassoError, PickShape, Result, Sensitivity, SpotFitMethod,
};

/// Prelude re-exporting commonly used types and functions for SMLM and Lisca binding assays.
pub mod prelude {
    pub use crate::binding::{
        BindingDetectConfig, BindingFilterConfig, BindingSpot, FrameSpotCount, MergeEvent,
        SpotTrackConfig, TrackSummary, TrackedSpot, detect_spots_frame, detect_spots_stack,
        detect_spots_u16_slice, filter_spots, spot_counts, track_spots,
    };
    pub use crate::clusterer::{cluster, dbscan, find_cluster_centers, frame_analysis};
    pub use crate::fitting::gaussfit::{GaussFitResult, eval_gaussian_spot, fit_spots_gauss};
    pub use crate::localize::{
        FitConfig, IdentifyOptions, fit, get_spots, identify, identify_in_frame, identify_in_image,
        local_maxima, localize, net_gradient,
    };
    pub use crate::plot::{
        plot_drift, plot_frc, plot_lnp_binding, plot_nena, plot_rendered_image,
    };
    pub use crate::postprocess::{
        LinkCombineMode, NenaResult, compute_dark_times, compute_local_density, link, nena,
        picked_locs, undrift,
    };
    pub use crate::render::{RenderBlurMethod, render, render_gaussian, render_hist};
    pub use crate::types::{
        Baseline, CameraInfo, ClusterCenter, FitEstimator, FitState, GaussModel, Identification,
        Localization, Metadata, PicassoError, PickShape, Result, Sensitivity, SpotFitMethod,
    };
}
