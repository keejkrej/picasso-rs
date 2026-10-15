//! Synthetic SMLM / DNA-PAINT / Binding Assay Movie Simulation.
//!
//! Ported from `picasso.simulate`. Uses `marray::sp::special::erf` and `marray::np::random`.

use crate::types::Result;
use marray::np::random::{normal, poisson, seed};
use marray::np::{Array2, Array3};
use marray::sp::special::erf;
use std::f64::consts::SQRT_2;

/// Configuration for synthetic microscopy movie generation (`picasso.simulate`).
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Image height in pixels.
    pub height: usize,
    /// Image width in pixels.
    pub width: usize,
    /// Number of movie frames.
    pub n_frames: usize,
    /// Background photon level per pixel per frame.
    pub bg_photons: f64,
    /// Camera baseline offset in ADU.
    pub baseline_adu: f64,
    /// Camera sensitivity (`e- / ADU`).
    pub sensitivity: f64,
    /// Camera gain (`ADU / e-` or EM gain).
    pub gain: f64,
    /// Readout noise standard deviation in ADU.
    pub readout_noise_adu: f64,
    /// Whether to apply Poisson shot noise.
    pub shot_noise: bool,
    /// Optional RNG seed for reproducibility.
    pub random_seed: Option<u64>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            height: 64,
            width: 64,
            n_frames: 10,
            bg_photons: 10.0,
            baseline_adu: 100.0,
            sensitivity: 1.0,
            gain: 1.0,
            readout_noise_adu: 1.5,
            shot_noise: true,
            random_seed: Some(42),
        }
    }
}

/// Emitter specification for a single frame `(frame, x, y, photons, sx, sy)`.
#[derive(Debug, Clone, Copy)]
pub struct SimulatedEmitter {
    pub frame: u32,
    pub x: f64,
    pub y: f64,
    pub photons: f64,
    pub sx: f64,
    pub sy: f64,
}

#[inline]
fn pixel_erf_1d(idx: usize, pos: f64, sigma: f64) -> f64 {
    let s = (SQRT_2 * sigma.max(1e-4)).recip();
    let d = (idx as f64) - pos;
    0.5 * (erf((d + 0.5) * s) - erf((d - 0.5) * s))
}

/// Renders a noiseless 2D frame of pixel-integrated 2D Gaussian emitters (`picasso.simulate.psf`).
pub fn render_noiseless_frame(
    height: usize,
    width: usize,
    bg_photons: f64,
    emitters: &[SimulatedEmitter],
) -> Array2<f64> {
    let mut frame = Array2::<f64>::from_elem((height, width), bg_photons.max(0.0));

    for em in emitters {
        let rx = (4.0 * em.sx).ceil() as isize + 1;
        let ry = (4.0 * em.sy).ceil() as isize + 1;
        let xc = em.x.round() as isize;
        let yc = em.y.round() as isize;

        let x0 = (xc - rx).max(0) as usize;
        let x1 = ((xc + rx + 1) as usize).min(width);
        let y0 = (yc - ry).max(0) as usize;
        let y1 = ((yc + ry + 1) as usize).min(height);

        for iy in y0..y1 {
            let py = pixel_erf_1d(iy, em.y, em.sy);
            for ix in x0..x1 {
                let px = pixel_erf_1d(ix, em.x, em.sx);
                frame[[iy, ix]] += em.photons.max(0.0) * px * py;
            }
        }
    }

    frame
}

/// Simulates a 3D movie stack `(n_frames, height, width)` from a list of [`SimulatedEmitter`]s
/// (`picasso.simulate.simulate_movie`).
pub fn simulate_movie(
    emitters: &[SimulatedEmitter],
    config: &SimulationConfig,
) -> Result<Array3<f64>> {
    if let Some(s) = config.random_seed {
        seed(s);
    }

    let mut movie = Array3::<f64>::zeros((config.n_frames, config.height, config.width));
    let conv = config.gain.max(1.0) / config.sensitivity.max(1e-6);

    for t in 0..config.n_frames {
        let frame_emitters: Vec<SimulatedEmitter> = emitters
            .iter()
            .copied()
            .filter(|e| e.frame as usize == t)
            .collect();
        let clean = render_noiseless_frame(
            config.height,
            config.width,
            config.bg_photons,
            &frame_emitters,
        );

        let read_noise = if config.readout_noise_adu > 0.0 {
            Some(normal(
                0.0,
                config.readout_noise_adu,
                &[config.height, config.width],
            ))
        } else {
            None
        };

        for iy in 0..config.height {
            for ix in 0..config.width {
                let expected_photons = clean[[iy, ix]].max(0.0);
                let sampled_photons = if config.shot_noise && expected_photons > 0.0 {
                    poisson(expected_photons, &[1])[0]
                } else {
                    expected_photons
                };
                let rn = read_noise.as_ref().map_or(0.0, |n| n[[iy, ix]]);
                let adu = config.baseline_adu + sampled_photons * conv + rn;
                movie[[t, iy, ix]] = adu.max(0.0);
            }
        }
    }

    Ok(movie)
}
