use anyhow::Context;
use libm::{erf, erfc};
use std::f64::consts::SQRT_2;

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
        let (bb_dist, t) = t.split_at_mut(max_depth + 1);
        let pi = &mut t[..=max_depth];
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

pub struct DssResult<'a> {
    ls_fit: LeastSquaresResult<'a>,
    phi: f64,
}

impl<'a> std::ops::Deref for DssResult<'a> {
    type Target = LeastSquaresResult<'a>;

    fn deref(&self) -> &Self::Target {
        &self.ls_fit
    }
}

impl<'a> DssResult<'a> {
    pub fn phi(&self) -> f64 {
        self.phi
    }

    pub fn se(&self) -> Option<impl Iterator<Item = f64> + '_> {
        self.inverse().map(|inv| {
            (0..self.ls_fit.n_effects()).map(|i| inv[(((i + 1) * (i + 2)) >> 1) - 1].sqrt())
        })
    }
}

pub struct Dss {
    work: Vec<f64>,
    ls: LeastSquares,
    lfact: LogFactorial,
    max_depth: usize,
    fixed_phi: Option<f64>,
}

impl Default for Dss {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_DEPTH: usize = 100;

impl Dss {
    pub fn new() -> Self {
        Self {
            work: Vec::new(),
            ls: LeastSquares::new(),
            lfact: LogFactorial::new(1000),
            max_depth: 0,
            fixed_phi: None,
        }
    }

    pub fn set_phi(&mut self, phi: f64) {
        self.fixed_phi = Some(phi)
    }

    pub fn clear_phi(&mut self) {
        self.fixed_phi = None
    }

    pub fn fit(&mut self, y: &[f64], depth: &[f64], x: &[f64]) -> anyhow::Result<DssResult> {
        let n_samples = y.len();
        assert_eq!(n_samples, depth.len(), "Unequal input vector sizes");

        // Setup work space
        let max_depth = match depth.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            Some(x) => *x as usize,
            None => return Err(anyhow!("Dss::fit() called with empty observation vector")),
        }
        .min(MAX_DEPTH)
        .max(1);

        self.max_depth = max_depth;
        // eprintln!("Max depth = {max_depth}");
        let work_size = 2 * (n_samples + max_depth + 1);
        self.work.resize(work_size, 0.0);

        let dsw = DssWork::from_work_slice(&mut self.work, n_samples, max_depth, &self.lfact);

        // Transform data
        for (z, (x, cov)) in dsw.z.iter_mut().zip(y.iter().zip(depth)) {
            *z = asin_transform(*x, *cov)
        }

        // Fit model
        self._fit(depth, x)
            .with_context(|| "Error in model fitting")
    }

    fn _fit(&mut self, depth: &[f64], x: &[f64]) -> anyhow::Result<DssResult> {
        let mut dsw =
            DssWork::from_work_slice(&mut self.work, depth.len(), self.max_depth, &self.lfact);

        // For the initial estimates we assume the variance is 1/depth
        // For the wls we use 1/variance as the weights, we can just use the depth
        // as the weights

        let ls_fit = self
            .ls
            .wls(x, dsw.z, depth)
            .with_context(|| "Error from wls for initial phi estimate")?;

        let chi2 = ls_fit
            .residuals()
            .unwrap()
            .iter()
            .zip(depth)
            .map(|(e, d)| e * e * d)
            .sum::<f64>();

        let phi = self
            .fixed_phi
            .unwrap_or_else(|| calc_phi(ls_fit.n_effects(), chi2, depth));

        // Calculate the weights for the next round of wls

        let fval = ls_fit.fitted_values().unwrap();
        for (ix, (f, d)) in fval.iter().zip(depth.iter()).enumerate() {
            let depth = (*d as usize).min(MAX_DEPTH).max(1);
            let pi = (f.sin() + 1.0) * 0.5;
            set_weight(pi, phi, depth, &mut dsw, ix);
        }

        let ls_fit = self
            .ls
            .wls(x, dsw.z, dsw.wt)
            .with_context(|| "Error from wls for second fitting")?;

        Ok(DssResult { ls_fit, phi })
    }
}

pub fn dss_wald_test(fit: &LeastSquaresResult, contrasts: &[f64]) -> f64 {
    wald_test(fit.beta(), fit.inverse().unwrap(), contrasts)
}

pub fn wald_test(beta: &[f64], cov: &[f64], contrasts: &[f64]) -> f64 {
    let p = beta.len();
    assert_eq!(p, contrasts.len(), "Contrasts vector is the wrong size");

    // Calculae C'VC
    let mut z = 0.0;
    let mut ix = 0;
    for (i, ci) in contrasts.iter().enumerate() {
        for (x, cj) in cov[ix..ix + i].iter().zip(contrasts.iter()) {
            z += 2.0 * ci * cj * x
        }
        z += ci * ci * cov[ix + i];
        ix += i + 1
    }

    // Calculate test statistic
    contrasts
        .iter()
        .zip(beta.iter())
        .map(|(c, b)| b * c)
        .sum::<f64>()
        / z.sqrt()
}

fn zmean(p: &[f64], y: &[f64]) -> f64 {
    p.iter().zip(y.iter()).map(|(p, y)| y * p).sum::<f64>()
}

fn zvariance(p: &[f64], y: &[f64]) -> f64 {
    let mn = zmean(p, y);

    p.iter()
        .zip(y.iter())
        .map(|(p, y)| (*y - mn).powi(2) * p)
        .sum::<f64>()
}

fn mk_betabinomial_dist(
    alpha: f64,
    beta: f64,
    d: usize,
    p: &mut [f64],
    pi: &mut [f64],
    lfact: &LogFactorial,
) {
    let konst = lbeta(alpha, beta);
    let d1 = d as f64;
    for y in 0..=d {
        pi[y] = (((2 * y) as f64) / d1 - 1.0).asin();
        p[y] = (lfact.dbetabinomial(alpha, beta, d, y) - konst).exp();
    }
}

fn set_weight(pi: f64, phi: f64, d: usize, dsw: &mut DssWork, ix: usize) {
    let alpha = pi * (1.0 - phi) / phi;
    let beta = (1.0 - pi) * (1.0 - phi) / phi;

    // First calculate betabinomial pdf
    mk_betabinomial_dist(
        alpha,
        beta,
        d,
        &mut dsw.bb_dist[..=d],
        &mut dsw.pi[..=d],
        dsw.lfact,
    );

    // Calculate variance
    let zvar = zvariance(&dsw.bb_dist[..=d], &dsw.pi[..=d]);
    if zvar.is_nan() {
        println!("{ix} zvar = {zvar} d: {d}, pi: {pi}, phi: {phi}");
    }
    dsw.wt[ix] = 1.0 / zvar
}

/// a) arcsin returns z between -pi/2 and +pi/2 ~ 1.57
/// b) z = ArcSin(2*(y+c0)/(m+2*c0)-1) is inverted by
/// y = (Sin(z)+1)*(m+2*c0)/2 - c0
pub fn asin_transform(y: f64, cov: f64) -> f64 {
    let c0 = 0.1;
    (2.0 * (y + c0) / (cov + 2.0 * c0) - 1.0).asin()
}

/// Lower tail of standard normal
pub fn pnorm(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / SQRT_2))
}

/// Upper tail of standard normal
pub fn pnormc(z: f64) -> f64 {
    0.5 * erfc(z / SQRT_2)
}

pub fn pvalue(ts: f32) -> f64 {
    if ts >= 0.0 {
        2.0 * pnormc(ts as f64)
    } else {
        2.0 * pnorm(ts as f64)
    }
}

fn calc_phi(p: usize, chi2: f64, depth: &[f64]) -> f64 {
    let n_samples = depth.len();
    let sigma_sq = chi2 / ((n_samples - p) as f64);

    // println!("{n_samples} {chi2} {sigma_sq}");
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
    fn test_wald_test() {
        let beta = vec![2.0, 4.0 / 3.0, 2.0, 2.0];
        let cov = vec![
            5.0 / 7.0,
            -2.0 / 7.0,
            22.0 / 21.0,
            -4.0 / 7.0,
            3.0 / 7.0,
            6.0 / 7.0,
            -3.0 / 7.0,
            -3.0 / 7.0,
            1.0 / 7.0,
            6.0 / 7.0,
        ];

        let contrasts = vec![0.0, 1.0, 0.0, 1.0];

        let t = wald_test(&beta, &cov, &contrasts);
        assert!((t - 3.256694736394648).abs() < f64::EPSILON.sqrt())
    }
    /*
       #[test]
       fn test_dss_fit() {
           let d = 18.0;
           let depth = vec![d; 6];
           let y = vec![0.25 * d, 0.25 * d, 0.25 * d, 0.75 * d, 0.75 * d, 0.75 * d];
           let x = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

           let mut dss = Dss::new();
           let fit = dss.fit(&y, &depth, &x).expect("Error in Dss::fit()");

           println!("beta: {:?}", fit.beta());
           println!("phi: {:?}", fit.phi());
           println!("res_var: {:?}", fit.res_var());
           println!("inverse: {:?}", fit.inverse());
           for (x, b) in fit.se().unwrap().zip(fit.beta()) {
               println!("se: {}, t: {}", x, *b / x);
           }
           panic!("ooook!");
       }

    */

    #[test]
    fn zmean_works() {
        let a = 4.0;
        let b = 3.0;
        let depth = 10;

        let lf = LogFactorial::new(10);
        let mut p = vec![0.0; 11];
        let mut pi = vec![0.0; 11];

        mk_betabinomial_dist(a, b, depth, &mut p, &mut pi, &lf);

        let res = zmean(&p, &pi);
        assert!((res - 0.1692037950).abs() < 1e-6);
    }

    #[test]
    fn zvariance_works() {
        let phi = 0.1;
        let pi = 0.3;
        let depth = 30;
        let lf = LogFactorial::new(30);
        let alpha = pi * (1.0 - phi) / phi;
        let beta = (1.0 - pi) * (1.0 - phi) / phi;

        let mut p = vec![0.0; depth + 1];
        let mut pi = vec![0.0; depth + 1];

        mk_betabinomial_dist(alpha, beta, depth, &mut p, &mut pi, &lf);

        let res = zvariance(&p, &pi);
        assert!((res - 0.153663).abs() < 1e-6);
    }
    /*
       #[test]
       fn zvariance_test() {
           let phis = [0.001, 0.01, 0.1, 0.25, 0.5, 0.999];
           let depth = 1;
           let lf = LogFactorial::new(depth + 1);
           let mut p_st = vec![0.0; depth + 1];
           let mut pi_st = vec![0.0; depth + 1];
           for i in 1..1000 {
               let pi = (i as f64) / 1000.0;
               print!("{pi}\t{depth}");
               for phi in phis {
                   let alpha = pi * (1.0 - phi) / phi;
                   let beta = (1.0 - pi) * (1.0 - phi) / phi;
                   mk_betabinomial_dist(alpha, beta, depth, &mut p_st, &mut pi_st, &lf);
                   let res = zvariance(&p_st, &pi_st);
                   let d = depth as f64;
                   let v1 = 1.0 / d;
                   let v2 = (1.0 + (d - 1.0) * phi) / d;
                   print!("\t{res}\t{v1}\t{v2}");
               }
               println!()
           }
           panic!("Oooook!")
       }

    */
}
