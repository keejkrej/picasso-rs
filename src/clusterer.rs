//! SMLM Clustering Algorithms: Density-Maxima Clusterer, DBSCAN, HDBSCAN, Frame Analysis,
//! and Cluster Center Aggregation.
//!
//! Ported from `picasso.clusterer`. Uses `marray::sp::spatial::{KDTree, ConvexHull}` and
//! `marray::sklearn::cluster::DBSCAN`.

use crate::types::{ClusterCenter, Localization, Result};
use marray::np::Array2;
use marray::sklearn::cluster::DBSCAN;
use marray::sp::spatial::{ConvexHull, KDTree};
use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

/// Tests whether a cluster's frame distribution passes the SMLM dark-frame / blinking test
/// (`picasso.clusterer.frame_analysis`).
///
/// Splits the acquisition range `[0..=max_frame]` into `20` equal temporal bins; returns `1`
/// if no single bin contains `> 80%` of the cluster's localizations (ruling out transient
/// sticking artifacts), or `0` otherwise.
pub fn frame_analysis(cluster_ids: &[i32], frames: &[u32]) -> Vec<i32> {
    if cluster_ids.is_empty() || frames.is_empty() {
        return Vec::new();
    }
    let max_frame = frames.iter().copied().max().unwrap_or(0).max(1) as f64;
    let n_bins = 20usize;

    let mut groups: HashMap<i32, Vec<u32>> = HashMap::new();
    for (&cid, &f) in cluster_ids.iter().zip(frames.iter()) {
        if cid >= 0 {
            groups.entry(cid).or_default().push(f);
        }
    }

    let mut passed_set: HashMap<i32, bool> = HashMap::new();
    for (cid, f_list) in groups {
        let n = f_list.len();
        if n == 0 {
            passed_set.insert(cid, false);
            continue;
        }
        let mut hist = [0usize; 20];
        for f in f_list {
            let b = (((f as f64) / (max_frame + 1.0)) * (n_bins as f64)).floor() as usize;
            hist[b.min(n_bins - 1)] += 1;
        }
        let max_bin = hist.into_iter().max().unwrap_or(0);
        let passed = (max_bin as f64) <= 0.8 * (n as f64);
        passed_set.insert(cid, passed);
    }

    cluster_ids
        .iter()
        .map(|&cid| {
            if cid >= 0 && *passed_set.get(&cid).unwrap_or(&false) {
                cid
            } else {
                -1
            }
        })
        .collect()
}

/// Performs Picasso's SMLM density-maxima clustering in 2D or 3D (`picasso.clusterer._cluster`).
///
/// 1. Builds a `KDTree` (`marray::sp::spatial::KDTree`).
/// 2. Counts local density within `radius` for each point.
/// 3. Identifies local density maxima (points with `density >= min_locs` and `>=` all neighbors within `radius`).
/// 4. Propagates labels from local maxima to neighbors within `radius` over two hops, merging connected components.
/// 5. Filters out clusters with `< min_locs` or failing `frame_analysis` (if enabled).
pub fn cluster(
    locs: &[Localization],
    radius_xy: f64,
    min_locs: usize,
    do_frame_analysis: bool,
    radius_z: Option<f64>,
    pixelsize_nm: Option<f64>,
) -> Result<Vec<Localization>> {
    if locs.is_empty() {
        return Ok(Vec::new());
    }
    let n = locs.len();
    let is_3d = radius_z.is_some() && locs.iter().all(|l| l.z.is_some());
    let dim = if is_3d { 3 } else { 2 };

    let mut coords = Array2::<f64>::zeros((n, dim));
    for (i, l) in locs.iter().enumerate() {
        coords[[i, 0]] = l.x / radius_xy.max(1e-9);
        coords[[i, 1]] = l.y / radius_xy.max(1e-9);
        if is_3d {
            let rz = radius_z.unwrap_or(radius_xy).max(1e-9);
            let px_nm = pixelsize_nm.unwrap_or(1.0).max(1e-9);
            let z_px = l.z.unwrap_or(0.0) / px_nm;
            coords[[i, 2]] = z_px / rz;
        }
    }

    let tree = KDTree::new(&coords);
    let mut neighbors_list: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut n_locs_local = vec![0usize; n];

    for i in 0..n {
        let pt: Vec<f64> = (0..dim).map(|d| coords[[i, d]]).collect();
        let nb = tree.query_radius(&pt, 1.0);
        n_locs_local[i] = nb.len();
        neighbors_list.push(nb);
    }

    // Identify local maxima
    let mut is_max = vec![false; n];
    for i in 0..n {
        if n_locs_local[i] < min_locs {
            continue;
        }
        let d_i = n_locs_local[i];
        if neighbors_list[i].iter().all(|&j| d_i >= n_locs_local[j]) {
            is_max[i] = true;
        }
    }

    // Disjoint-set union for connected cluster assignments
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            if ra < rb {
                parent[rb] = ra;
            } else {
                parent[ra] = rb;
            }
        }
    }

    let mut assigned = vec![false; n];
    for i in 0..n {
        if is_max[i] {
            assigned[i] = true;
            for &j in &neighbors_list[i] {
                assigned[j] = true;
                union(&mut parent, i, j);
            }
        }
    }
    // Second hop: non-maxima assigned points pull in their neighbors within radius
    let first_hop_assigned = assigned.clone();
    for i in 0..n {
        if first_hop_assigned[i] && !is_max[i] {
            for &j in &neighbors_list[i] {
                if !assigned[j] {
                    assigned[j] = true;
                    union(&mut parent, i, j);
                }
            }
        }
    }

    let mut raw_labels = vec![-1i32; n];
    let mut root_counts: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        if assigned[i] {
            let r = find(&mut parent, i);
            *root_counts.entry(r).or_insert(0) += 1;
        }
    }

    let mut root_to_id: HashMap<usize, i32> = HashMap::new();
    let mut next_id = 0i32;
    for i in 0..n {
        if assigned[i] {
            let r = find(&mut parent, i);
            if *root_counts.get(&r).unwrap_or(&0) >= min_locs {
                let id = *root_to_id.entry(r).or_insert_with(|| {
                    let cid = next_id;
                    next_id += 1;
                    cid
                });
                raw_labels[i] = id;
            }
        }
    }

    if do_frame_analysis {
        let frames: Vec<u32> = locs.iter().map(|l| l.frame).collect();
        raw_labels = frame_analysis(&raw_labels, &frames);
    }

    // Re-index valid cluster IDs contiguously `0..k` and filter out `-1` (matching `extract_valid_labels`)
    let mut remap: BTreeMap<i32, i32> = BTreeMap::new();
    let mut next_clean = 0i32;
    for &cid in &raw_labels {
        if cid >= 0 && !remap.contains_key(&cid) {
            remap.insert(cid, next_clean);
            next_clean += 1;
        }
    }

    let mut out = Vec::new();
    for (loc, cid) in locs.iter().zip(raw_labels) {
        if let Some(&clean_id) = remap.get(&cid) {
            let mut c = loc.clone();
            c.group = Some(clean_id);
            c.cluster = Some(clean_id);
            out.push(c);
        }
    }

    Ok(out)
}

/// Runs DBSCAN clustering on `locs` using `marray::sklearn::cluster::DBSCAN` (`picasso.clusterer.dbscan`).
pub fn dbscan(
    locs: &[Localization],
    radius: f64,
    min_samples: usize,
    min_locs: usize,
    pixelsize_nm: Option<f64>,
) -> Result<Vec<Localization>> {
    if locs.is_empty() {
        return Ok(Vec::new());
    }
    let n = locs.len();
    let is_3d = pixelsize_nm.is_some() && locs.iter().all(|l| l.z.is_some());
    let dim = if is_3d { 3 } else { 2 };
    let mut coords = Array2::<f64>::zeros((n, dim));

    for (i, l) in locs.iter().enumerate() {
        coords[[i, 0]] = l.x;
        coords[[i, 1]] = l.y;
        if is_3d {
            let px_nm = pixelsize_nm.unwrap_or(1.0).max(1e-9);
            coords[[i, 2]] = l.z.unwrap_or(0.0) / px_nm;
        }
    }

    let labels = DBSCAN::new(radius, min_samples).fit_predict(&coords);
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &lbl in labels.iter() {
        let cid = lbl.round() as i32;
        if cid >= 0 {
            *counts.entry(cid).or_insert(0) += 1;
        }
    }

    let mut remap: BTreeMap<i32, i32> = BTreeMap::new();
    let mut next_id = 0i32;
    for &lbl in labels.iter() {
        let cid = lbl.round() as i32;
        if cid >= 0 && *counts.get(&cid).unwrap_or(&0) >= min_locs && !remap.contains_key(&cid) {
            remap.insert(cid, next_id);
            next_id += 1;
        }
    }

    let mut out = Vec::new();
    for (loc, &lbl) in locs.iter().zip(labels.iter()) {
        let cid = lbl.round() as i32;
        if let Some(&clean_id) = remap.get(&cid) {
            let mut c = loc.clone();
            c.group = Some(clean_id);
            c.cluster = Some(clean_id);
            out.push(c);
        }
    }

    Ok(out)
}

/// Computes inverse-variance weighted localization precision error (`picasso.clusterer.error_sums_wtd`).
#[inline]
pub fn error_sums_wtd(x: &[f64], w: &[f64]) -> f64 {
    let n = x.len();
    if n <= 1 {
        return 0.0;
    }
    let w_sum: f64 = w.iter().sum();
    if w_sum <= 0.0 {
        return 0.0;
    }
    let x_mean: f64 = x.iter().zip(w.iter()).map(|(&xi, &wi)| wi * xi).sum::<f64>() / w_sum;
    let num: f64 = x
        .iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| wi * (xi - x_mean).powi(2))
        .sum();
    (num / (((n - 1) as f64) * w_sum)).max(0.0).sqrt()
}

/// Calculates aggregated [`ClusterCenter`] records for all clusters in `locs`
/// (`picasso.clusterer.find_cluster_centers`), using `marray::sp::spatial::ConvexHull` for 2D hull areas.
pub fn find_cluster_centers(
    locs: &[Localization],
    pixelsize_nm: Option<f64>,
) -> Vec<ClusterCenter> {
    let mut groups: BTreeMap<i32, Vec<&Localization>> = BTreeMap::new();
    for l in locs {
        let cid = l.group.or(l.cluster).unwrap_or(-1);
        if cid >= 0 {
            groups.entry(cid).or_default().push(l);
        }
    }

    let mut centers = Vec::with_capacity(groups.len());
    for (gid, list) in groups {
        let n = list.len();
        if n == 0 {
            continue;
        }
        let nf = n as f64;
        let frame_mean = list.iter().map(|l| l.frame as f64).sum::<f64>() / nf;
        let photons_mean = list.iter().map(|l| l.photons).sum::<f64>() / nf;
        let sx_mean = list.iter().map(|l| l.sx).sum::<f64>() / nf;
        let sy_mean = list.iter().map(|l| l.sy).sum::<f64>() / nf;
        let bg_mean = list.iter().map(|l| l.bg).sum::<f64>() / nf;
        let ell_mean = list.iter().map(|l| l.ellipticity).sum::<f64>() / nf;
        let ng_mean = list.iter().map(|l| l.net_gradient).sum::<f64>() / nf;

        let x_vec: Vec<f64> = list.iter().map(|l| l.x).collect();
        let y_vec: Vec<f64> = list.iter().map(|l| l.y).collect();
        let wx_vec: Vec<f64> = list.iter().map(|l| 1.0 / l.lpx.max(1e-6).powi(2)).collect();
        let wy_vec: Vec<f64> = list.iter().map(|l| 1.0 / l.lpy.max(1e-6).powi(2)).collect();

        let wx_sum: f64 = wx_vec.iter().sum::<f64>().max(1e-12);
        let wy_sum: f64 = wy_vec.iter().sum::<f64>().max(1e-12);

        let x_wmean: f64 = x_vec.iter().zip(&wx_vec).map(|(x, w)| x * w).sum::<f64>() / wx_sum;
        let y_wmean: f64 = y_vec.iter().zip(&wy_vec).map(|(y, w)| y * w).sum::<f64>() / wy_sum;

        let std_x = (x_vec.iter().map(|x| (x - x_wmean).powi(2)).sum::<f64>() / nf).sqrt();
        let std_y = (y_vec.iter().map(|y| (y - y_wmean).powi(2)).sum::<f64>() / nf).sqrt();

        let lpx = if n > 1 {
            error_sums_wtd(&x_vec, &wx_vec)
        } else {
            list[0].lpx
        };
        let lpy = if n > 1 {
            error_sums_wtd(&y_vec, &wy_vec)
        } else {
            list[0].lpy
        };

        let (z_opt, std_z_opt, lpz_opt) = if list.iter().all(|l| l.z.is_some()) {
            let px_nm = pixelsize_nm.unwrap_or(1.0).max(1e-6);
            let z_vec: Vec<f64> = list.iter().map(|l| l.z.unwrap_or(0.0)).collect();
            let wz_vec: Vec<f64> = list
                .iter()
                .map(|l| {
                    let lpz_val = l.lpz.unwrap_or((l.lpx + l.lpy) * px_nm).max(1e-6);
                    1.0 / (lpz_val * lpz_val)
                })
                .collect();
            let wz_sum: f64 = wz_vec.iter().sum::<f64>().max(1e-12);
            let z_wmean: f64 = z_vec.iter().zip(&wz_vec).map(|(z, w)| z * w).sum::<f64>() / wz_sum;
            let sz = (z_vec.iter().map(|z| (z - z_wmean).powi(2)).sum::<f64>() / nf).sqrt();
            let lz = if n > 1 {
                error_sums_wtd(&z_vec, &wz_vec)
            } else {
                list[0].lpz.unwrap_or(sz)
            };
            (Some(z_wmean), Some(sz), Some(lz))
        } else {
            (None, None, None)
        };

        // Binding events: number of non-adjacent frame runs
        let mut frames: Vec<u32> = list.iter().map(|l| l.frame).collect();
        frames.sort_unstable();
        frames.dedup();
        let mut n_events = 1u32;
        for w in frames.windows(2) {
            if w[1] > w[0] + 1 {
                n_events += 1;
            }
        }

        // 2D ConvexHull area via `marray::sp::spatial::ConvexHull`
        let mut area = 0.0;
        let mut convexhull = 0.0;
        if n >= 3 {
            let mut pts2d = Array2::<f64>::zeros((n, 2));
            for (i, l) in list.iter().enumerate() {
                pts2d[[i, 0]] = l.x;
                pts2d[[i, 1]] = l.y;
            }
            if let Ok(hull) = ConvexHull::new(&pts2d) {
                area = hull.area;
                convexhull = (area / PI).max(0.0).sqrt();
            }
        }

        centers.push(ClusterCenter {
            groups: gid,
            frame: frame_mean,
            x: x_wmean,
            y: y_wmean,
            z: z_opt,
            std_x,
            std_y,
            std_z: std_z_opt,
            photons: photons_mean,
            sx: sx_mean,
            sy: sy_mean,
            bg: bg_mean,
            lpx,
            lpy,
            lpz: lpz_opt,
            ellipticity: ell_mean,
            net_gradient: ng_mean,
            n: n as u32,
            n_events,
            area,
            convexhull,
        });
    }

    centers
}
