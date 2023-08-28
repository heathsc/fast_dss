/* use super::lsquares::{get_sizes, WeightedLeastSquares};

#[derive(Default)]
pub struct Dss {
    work: Vec<f32>,
    wls: WeightedLeastSquares,
}

impl Dss {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn estimate_phi(&mut self, mv: &[f32], x: &[f32], z: &[f32]) -> anyhow::Result<f32> {
        let (_, m, p) = get_sizes(x, mv);
        assert_eq!(z.len(), m, "Inconsistent vector lengths in estimate_phi()");
        self.work.resize(m, 0.0);
        for (z, w) in mv.iter().zip(self.work.iter_mut()) {
            *w = 1.0 / *z
        }

        let beta = self.wls.weighted_least_sqaures(x, z, &self.work)?.0;
        let chi2 = chisq(mv, x, beta, z, m);
        Ok(phi(p, chi2, mv))
    }
}

 */

fn chisq(mv: &[f32], x: &[f32], beta: &[f32], z: &[f32], ns: usize) -> f32 {
    mv.iter()
        .copied()
        .zip(z.iter().copied())
        .enumerate()
        .map(|(i, (m, zi))| {
            let pred: f32 = beta
                .iter()
                .copied()
                .zip(x[i..].iter().step_by(ns).copied())
                .map(|(b, xj)| b * xj)
                .sum();
            m * (zi - pred).powi(2)
        })
        .sum()
}

#[test]
fn chisq_works() {
    let m: Vec<f32> = vec![1.0, 1.0];
    let x: Vec<f32> = vec![1.0, 1.0, 0.0, 1.0];
    let beta: Vec<f32> = vec![1.0, 1.0];
    let z: Vec<f32> = vec![0.0, 0.0];
    let res = chisq(&m, &x, &beta, &z, 2);
    assert_eq!(res, 5.0);
}

fn phi(p: usize, chi2: f32, m: &[f32]) -> f32 {
    let ns = m.len();
    let sigmasq = chi2 / (ns - p) as f32;
    let sm: f32 = m.iter().map(|x| *x - 1.0).sum();
    ((ns as f32) * (sigmasq - 1.0) / sm).max(0.001).min(0.999)
}

#[test]
fn phi_works() {
    let p: usize = 2;
    let m: Vec<f32> = vec![26., 61., 19., 13.];
    let chi2: f32 = 10.;
    let res = phi(p, chi2, &m);
    assert!((res - 0.13913).abs() < 1e-6);
}
