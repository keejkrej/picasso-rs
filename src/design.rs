//! DNA-PAINT Origami Lattice and Structure Design Helpers.
//!
//! Ported from `picasso.design`.

/// Generates a 2D rectangular or honeycomb origami lattice of binding site coordinates `(x_nm, y_nm)`.
pub fn generate_lattice_sites(
    rows: usize,
    cols: usize,
    spacing_x_nm: f64,
    spacing_y_nm: f64,
    hexagonal: bool,
) -> Vec<(f64, f64)> {
    let mut sites = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let x_offset = if hexagonal && (r % 2 == 1) {
            0.5 * spacing_x_nm
        } else {
            0.0
        };
        for c in 0..cols {
            let x = (c as f64) * spacing_x_nm + x_offset;
            let y = (r as f64) * spacing_y_nm;
            sites.push((x, y));
        }
    }
    sites
}
