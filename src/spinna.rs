//! SPINNA: Stoichiometric Protein Labeling and Nearest-Neighbor Analysis.
//!
//! Ported from `picasso.spinna`. Uses `marray::sp::spatial::KDTree` and `marray::sp::stats::ks_2samp`.

use crate::types::Result;
use marray::np::{Array1, Array2};
use marray::sp::spatial::KDTree;
use marray::sp::stats::ks_2samp;

/// Computes `1..=k` nearest-neighbor distance vectors for a 2D or 3D coordinate array `(N, D)`
/// (`picasso.spinna`).
pub fn nearest_neighbor_distances(coords: &Array2<f64>, k_neighbors: usize) -> Result<Vec<Array1<f64>>> {
    let n = coords.shape()[0];
    let dim = coords.shape()[1];
    if n <= k_neighbors || k_neighbors == 0 {
        return Err("Not enough points for k-nearest neighbor distance calculation".into());
    }
    let tree = KDTree::new(coords);
    let mut dists_by_order: Vec<Vec<f64>> = vec![Vec::with_capacity(n); k_neighbors];

    for i in 0..n {
        let pt: Vec<f64> = (0..dim).map(|d| coords[[i, d]]).collect();
        let (dists, _) = tree.query(&pt, k_neighbors + 1);
        for ord in 0..k_neighbors {
            dists_by_order[ord].push(dists[ord + 1]);
        }
    }

    Ok(dists_by_order
        .into_iter()
        .map(Array1::from_vec)
        .collect())
}

/// Evaluates the two-sample Kolmogorov-Smirnov test distance between observed and simulated
/// nearest-neighbor distance distributions (`picasso.spinna`) using `marray::sp::stats::ks_2samp`.
pub fn compare_nnd_distributions(
    observed_nnd: &Array1<f64>,
    simulated_nnd: &Array1<f64>,
) -> Result<(f64, f64)> {
    let obs_slice = observed_nnd.as_slice().unwrap_or(&[]);
    let sim_slice = simulated_nnd.as_slice().unwrap_or(&[]);
    if obs_slice.is_empty() || sim_slice.is_empty() {
        return Err("NND distributions must not be empty".into());
    }
    if obs_slice == sim_slice {
        return Ok((0.0, 1.0));
    }
    Ok(ks_2samp(obs_slice, sim_slice))
}
