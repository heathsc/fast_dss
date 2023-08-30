use anyhow::Context;

use super::{betabinomial::*, lsquares::*};

struct DssWork<'a> {
    z: &'a mut [f64],
    wt: &'a mut [f64],
    bb_dist: &'a mut [f64],
    pi: &'a mut [f64],
    lfact: &'a LogFactorial,
    n_samples: usize,
}

impl<'a> DssWork<'a> {
    fn from_work_slice(
        work: &'a mut [f64],
        n_samples: usize,
        max_depth: usize,
        lfact: &'a LogFactorial,
    ) -> Self {
        let (z, t) = work.split_at_mut(n_samples);
        let (wt, t) = t.split_at_mut(n_samples);
        let (bb_dist, pi) = t.split_at_mut(max_depth + 1);
        DssWork {
            z,
            wt,
            bb_dist,
            pi,
            lfact,
            n_samples,
        }
    }
}

pub struct Dss {
    work: Vec<f64>,
    ls: LeastSquares,
    lfact: LogFactorial,
    max_depth: usize,
}

impl Dss {
    pub fn new() -> Self {
        Self {
            work: Vec::new(),
            ls: LeastSquares::new(),
            lfact: LogFactorial::new(1000),
            max_depth: 0,
        }
    }

    pub fn fit(&mut self, y: &[f64], depth: &[f64], x: &[f64]) -> anyhow::Result<()> {
        let n_samples = y.len();
        assert_eq!(n_samples, depth.len(), "Unequal input vector sizes");

        // Setup work space
        let max_depth = match depth.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(x) => *x as usize,
            None => return Err(anyhow!("Dss::fit() called with empty observation vector")),
        };

        self.max_depth = max_depth;

        let work_size = 2 * (n_samples + max_depth + 1);
        self.work.resize(work_size, 0.0);

        let dsw = DssWork::from_work_slice(&mut self.work, n_samples, max_depth, &self.lfact);

        // Transform data
        for (z, (x, cov)) in dsw.z.iter_mut().zip(y.iter().zip(depth)) {
            *z = asin_transform(*x, *cov)
        }

        // Initial phi estimate
        let dss_fit = self
            ._fit(depth, x)
            .with_context(|| "Error in model fitting")?;

        Ok(())
    }

    fn _fit(&mut self, depth: &[f64], x: &[f64]) -> anyhow::Result<LeastSquaresResult> {
        let mut dsw =
            DssWork::from_work_slice(&mut self.work, depth.len(), self.max_depth, &self.lfact);

        // For the initial estimates we assume the variance is 1/depth
        // For the wls we use 1/variance as the weights, we can just use the depth
        // as the weights

        let ls_res = self
            .ls
            .wls(x, dsw.z, depth)
            .with_context(|| "Error from wls for initial phi estimate")?;

        let chi2 = ls_res
            .residuals()
            .unwrap()
            .iter()
            .zip(depth)
            .map(|(e, d)| e * d)
            .sum::<f64>();

        let phi = calc_phi(ls_res.n_effects(), chi2, depth);

        // Calculate the weights for the next round of wls

        let fval = ls_res.fitted_values().unwrap();
        for (ix, (f, d)) in fval.iter().zip(depth.iter()).enumerate() {
            let pi = (f.sin() + 1.0) * 0.5;
            set_weight(pi, phi, d.round() as usize, &mut dsw, ix);
        }

        self.ls
            .wls(x, dsw.z, dsw.wt)
            .with_context(|| "Error from wls for second fitting")
    }
}

fn set_weight(pi: f64, phi: f64, d: usize, dsw: &mut DssWork, ix: usize) {
    let alpha = pi * (1.0 - phi) / phi;
    let beta = 1.0 - alpha;
    let konst = lbeta(alpha, beta);

    let d1 = d as f64;

    // First calculate betabinomial pdf
    for y in 0..=d {
        dsw.pi[y] = (((2 * y) as f64) / d1 - 1.0).sin();
        dsw.bb_dist[y] = (dsw.lfact.dbetabinomial(alpha, beta, d, y) - konst).exp()
    }

    let zmean = dsw
        .bb_dist
        .iter()
        .zip(dsw.pi.iter())
        .map(|(p, y)| y * p)
        .sum::<f64>();

    let zvariance = dsw
        .bb_dist
        .iter()
        .zip(dsw.pi.iter())
        .map(|(p, y)| (*y - zmean).powi(2) * p)
        .sum::<f64>();

    dsw.wt[ix] = 1.0 / zvariance
}

/// a) arcsin returns z between -pi/2 and +pi/2 ~ 1.57
/// b) z = ArcSin(2*(y+c0)/(m+2*c0)-1) is inverted by
/// y = (Sin(z)+1)*(m+2*c0)/2 - c0
pub fn asin_transform(y: f64, cov: f64) -> f64 {
    let c0 = 0.1;
    (2.0 * (y + c0) / (cov + 2.0 * c0) - 1.0).asin()
}

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

pub fn calc_phi(p: usize, chi2: f64, depth: &[f64]) -> f64 {
    let n_samples = depth.len();
    let sigma_sq = chi2 / ((n_samples - p) as f64);

    let sm = depth.iter().sum::<f64>() - (n_samples as f64);

    ((n_samples as f64) * (sigma_sq - 1.0) / sm)
        .max(0.001)
        .min(0.999)
}

mod test {
    use super::*;

    #[test]
    fn test_asin_transform() {
        let y = 10.0;
        let n = 30.0;
        assert!((&asin_transform(y, n) + 0.337496).abs() < 1e-6);
    }

    #[test]
    fn test_calc_phi() {
        let p: usize = 2;
        let m: Vec<f64> = vec![26., 61., 19., 13.];
        let chi2: f64 = 10.;
        let res = calc_phi(p, chi2, &m);
        assert!((res - 0.13913).abs() < 1e-6);
    }

    #[test]
    fn test_chisq() {
        let m: Vec<f32> = vec![1.0, 1.0];
        let x: Vec<f32> = vec![1.0, 1.0, 0.0, 1.0];
        let beta: Vec<f32> = vec![1.0, 1.0];
        let z: Vec<f32> = vec![0.0, 0.0];
        let res = chisq(&m, &x, &beta, &z, 2);
        assert_eq!(res, 5.0);
    }
}
