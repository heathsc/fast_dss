use crate::cholesky::*;
use anyhow::Context;

pub const LS_NO_RES: u16 = 1; // Do not calculate residuals (implies LS_NO_VAR)
pub const LS_NO_VAR: u16 = 2; // Do not calculate Variance matrix of effects
pub const LS_FILTER: u16 = 4; // Filter columns of X
pub const LS_FILTER_SET: u16 = 8; // Filter has been set
pub const LS_NO_RECHECK: u16 = 16; // Do not recheck filter on new input
pub const LS_MASK: u16 = 31;

struct LeastSquaresParams<'a> {
    xx: &'a mut [f64],
    l: &'a mut [f64],
    xy: &'a mut [f64],
    beta: &'a mut [f64],
    residuals: &'a mut [f64],
    fitted_values: &'a mut [f64],
    skip: &'a mut [bool],
}

#[derive(Default)]
pub struct LeastSquares {
    // general workspace
    // X'X lower diagonal - p * (p + 1) / 2
    // L (Cholesky decomposition of X'X) lower diagonal - p * (p + 1) / 2
    // X'Y - p
    // Beta (effects) - p
    // Residuals - m
    // Fitted Values - m
    work: Vec<f64>,
    skip: Vec<bool>,
    n_effects: usize, // p
    n_samples: usize, // m
    flags: u16,
}

impl LeastSquares {
    fn slices_mut(&mut self) -> LeastSquaresParams {
        assert!(!self.work.is_empty());
        let p = self.n_effects;
        let k = (p * (p + 1)) >> 1;
        let (xx, t) = self.work.split_at_mut(k);
        let (l, t) = t.split_at_mut(k);
        let (xy, t) = t.split_at_mut(p);
        let (beta, t) = t.split_at_mut(p);
        let (residuals, fitted_values) = t.split_at_mut(self.n_samples);
        LeastSquaresParams {
            xx,
            l,
            xy,
            beta,
            residuals,
            fitted_values,
            skip: &mut self.skip,
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_flags(&mut self, x: u16) {
        self.flags = x & LS_MASK
    }

    pub fn flags(&self) -> u16 {
        self.flags
    }

    pub fn skip(&self) -> &[bool] {
        &self.skip
    }

    pub fn set_skip(&mut self, s: &[bool]) {
        assert_eq!(s.len(), self.skip.len());
        self.skip.copy_from_slice(s);
        self.flags |= LS_FILTER | LS_FILTER_SET | LS_NO_RECHECK;
    }

    fn setup_workspace(&mut self, x: &[f64], y: &[f64]) {
        let m = y.len();
        let n = x.len();
        let p = n / m;
        assert_eq!(p * m, n, "Incorrect size for design matrix");

        // Reallocate workspace if required
        if self.skip.len() != p {
            let work_size = p * (p + 3) + 2 * m;
            self.work.resize(work_size, 0.0);
            self.skip.resize(p, false);
            self.flags &= !LS_FILTER_SET;
        }

        self.n_samples = m;
        self.n_effects = p;
    }

    /// m - number of samples
    /// p - number of effects
    /// Design matrix x has m rows and p columns stored in a m * p vector ordered by columns
    /// (i.e., column 1, column 2 etc.)
    /// y is the vector of observations and has m elements
    pub fn least_squares(&mut self, x: &[f64], y: &[f64]) -> anyhow::Result<LeastSquaresResult> {
        if x.is_empty() {
            Err(anyhow!("Empty Design matrix"))
        } else {
            self.setup_workspace(x, y);
            if (self.flags & LS_FILTER) == 0 {
                self.least_squares_unfiltered(x, y)
            } else {
                self.least_squares_filtered(x, y)
            }
        }
    }

    fn check_filter(&mut self, x: &[f64], y: &[f64]) -> anyhow::Result<()> {
        let (m, p) = (self.n_samples, self.n_effects);

        let LeastSquaresParams {
            xx,
            l,
            xy,
            beta: _,
            residuals: _,
            fitted_values: _,
            skip,
        } = self.slices_mut();

        make_xx_xy(m, p, x, y, xx, xy);
        find_dependencies(xx, l, skip, p)
            .with_context(|| "Error returned from find dependencies step")?;
        self.flags |= LS_FILTER_SET;
        Ok(())
    }

    fn least_squares_filtered(
        &mut self,
        x: &[f64],
        y: &[f64],
    ) -> anyhow::Result<LeastSquaresResult> {
        if (self.flags & (LS_FILTER_SET | LS_NO_RECHECK)) != (LS_FILTER_SET | LS_NO_RECHECK) {
            self.check_filter(x, y)?;
        }
        let flags = self.flags;
        let check_flags = |x: u16| (flags & x) != 0;

        let (m, p) = (self.n_samples, self.n_effects);
        let p_used: usize = self.skip.iter().filter(|x| !**x).count();
        if p_used == 0 {
            return Err(anyhow!("Design matrix is empty"));
        }

        let mut lsp = self.slices_mut();

        //       if p_used < p {
        //           lsp = lsp.trim(p_used)
        //       }

        let sz = (p_used * (p_used + 1)) >> 1;

        make_filtered_xx_xy(
            m,
            p,
            x,
            y,
            &mut lsp.xx[..sz],
            &mut lsp.xy[..p_used],
            lsp.skip,
        );

        cholesky(&lsp.xx[..sz], &mut lsp.l[..sz], p_used)
            .with_context(|| "Error returned from Cholesy Decomposition")?;
        cholesky_solve(&lsp.l[..sz], &lsp.xy[..p_used], &mut lsp.beta[..p_used]);

        let (fit, res) = if check_flags(LS_NO_RES) {
            (None, None)
        } else {
            calc_filtered_residuals(
                x,
                y,
                &lsp.beta[..p_used],
                lsp.fitted_values,
                lsp.residuals,
                lsp.skip,
            );
            (
                Some(lsp.fitted_values as &[f64]),
                Some(lsp.residuals as &[f64]),
            )
        };

        // If necessary, expand beta vector to account for skipped columns
        if p_used < p {
            let mut ix = p_used;
            for (i, s) in lsp.skip.iter().enumerate().rev() {
                lsp.beta[i] = if *s {
                    0.0
                } else {
                    ix -= 1;
                    lsp.beta[ix]
                }
            }
        }

        let rss = res.map(|r| r.iter().map(|e| e * e).sum());

        let var = if !check_flags(LS_NO_RES | LS_NO_VAR) && m > p_used {
            let res_var = rss.unwrap() / ((m - p_used) as f64);
            let v = lsp.xx;
            cholesky_inverse(&lsp.l[..sz], &mut v[..sz], p_used);
            for z in v.iter_mut() {
                *z *= res_var;
            }
            // If necessary, expand var vector to account for skipped columns
            if p_used < p {
                let mut ix = sz;
                let mut ix1 = (p * (p + 1)) >> 1;
                for (i, s) in lsp.skip.iter().enumerate().rev() {
                    ix1 -= i + 1;
                    if *s {
                        for x in v[ix1..ix1 + i + 1].iter_mut() {
                            *x = 0.0;
                        }
                    } else {
                        for j in (0..=i).rev() {
                            v[ix1 + j] = if lsp.skip[j] {
                                0.0
                            } else {
                                ix -= 1;
                                v[ix]
                            }
                        }
                    }
                }
            }

            Some(v as &[f64])
        } else {
            None
        };

        Ok(LeastSquaresResult {
            beta: lsp.beta,
            chol: lsp.l,
            fit,
            res,
            var,
            rss,
            df: m - p,
        })
    }

    fn least_squares_unfiltered(
        &mut self,
        x: &[f64],
        y: &[f64],
    ) -> anyhow::Result<LeastSquaresResult> {
        let flags = self.flags;
        let check_flags = |x| (flags & x) != 0;

        let (m, p) = (self.n_samples, self.n_effects);

        let LeastSquaresParams {
            xx,
            l,
            xy,
            beta,
            residuals: res_store,
            fitted_values: fit_store,
            skip: _,
        } = self.slices_mut();

        make_xx_xy(m, p, x, y, xx, xy);
        cholesky(xx, l, p).with_context(|| "Error returned from Cholesy Decomposition")?;
        cholesky_solve(l, xy, beta);

        let (fit, res) = if check_flags(LS_NO_RES) {
            (None, None)
        } else {
            calc_residuals(x, y, beta, fit_store, res_store);
            (Some(fit_store as &[f64]), Some(res_store as &[f64]))
        };

        let rss = res.map(|r| r.iter().map(|e| e * e).sum());

        let var = if !check_flags(LS_NO_RES | LS_NO_VAR) && m > p {
            let res_var = rss.unwrap() / ((m - p) as f64);
            let v = xx;
            cholesky_inverse(l, v, p);
            for z in v.iter_mut() {
                *z *= res_var;
            }
            Some(v as &[f64])
        } else {
            None
        };

        Ok(LeastSquaresResult {
            beta,
            chol: l,
            fit,
            res,
            var,
            rss,
            df: m - p,
        })
    }
}

#[test]
fn least_squares_works() {
    let x = vec![1.0, 1.0, 1.0, 1.0, 0.2, -0.3, 2.5, -1.6];
    let y = vec![23.1, 22.8, 27.3, 20.5];

    let mut ls = LeastSquares::new();
    let r = ls.least_squares(&x, &y).expect("Error in least squares");

    let tst = |x: &[f64], y: &[f64], s: &str| {
        assert_eq!(
            x.len(),
            y.len(),
            "Unequal vector lengths when comparing for {s}"
        );
        let z: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(
            z < 1.0e-16,
            "Unexpected result from least_squares_works for {s} (diff = {})",
            z
        );
    };

    println!("Beta: {:?}", r.beta());
    println!("Res: {:?}", r.residuals());
    println!("RSS: {:?}, Res_var: {:?}", r.rss(), r.res_var());
    println!("Var_Matrix {:?}", r.var_matrix());

    let beta_exp = &[23.09493166287016, 1.6503416856492028];
    let res_exp = &[
        -0.3249999999999993,
        0.20017084282460118,
        0.07921412300683528,
        0.04561503416856638,
    ];
    let var_exp = &[
        0.019607030694631076,
        -0.001754544133747747,
        0.008772720668738737,
    ];

    tst(r.beta(), beta_exp, "beta");
    tst(
        r.residuals().expect("Missing residuals"),
        res_exp,
        "residuals",
    );
    tst(
        r.var_matrix().expect("Missing covariance matrix"),
        var_exp,
        "covariance matrix",
    );
}

#[test]
fn least_squares_works2() {
    let x = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 0.4, -0.6, 5.0, -3.2, 0.3, 0.2, -0.3, 2.5, -1.6, 0.15, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -2.0, 7.2, 3.4, -1.1, -0.9,
    ];

    /*  let x = vec![
         1.0, 1.0, 1.0, 1.0, 1.0, 0.4, -0.6, 5.0, -3.2, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0, -2.0, 7.2,
         3.4, -1.1, -0.9,
     ];

    */

    let y = vec![23.1, 22.8, 27.3, 20.5, 22.2];

    let mut ls = LeastSquares::new();
    ls.set_flags(LS_FILTER);
    let r = ls.least_squares(&x, &y).expect("Error in least squares");

    let tst = |x: &[f64], y: &[f64], s: &str| {
        assert_eq!(
            x.len(),
            y.len(),
            "Unequal vector lengths when comparing for {s}"
        );
        let z: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(
            z < 1.0e-16,
            "Unexpected result from least_squares_works2 for {s} (diff = {})",
            z
        );
    };

    println!("Beta: {:?}", r.beta());
    println!("Res: {:?}", r.residuals());
    println!("RSS: {:?}, Res_var: {:?}", r.rss(), r.res_var());
    println!("Var_Matrix {:?}", r.var_matrix());

    let beta_exp = &[
        22.734784668944663,
        0.8008251881976463,
        0.0,
        0.083744953680487,
        0.0,
        0.0813664985363918,
    ];
    let res_exp = &[
        0.12387329916857581,
        -0.12387329916857936,
        0.2844432950433742,
        0.4173590816778372,
        -0.7018023767212043,
    ];
    let var_exp = &[
        0.27098418670466096,
        -0.01471783017703672,
        0.025548106132614847,
        0.0,
        0.0,
        0.0,
        -0.2655236990658112,
        0.034910915149720105,
        0.0,
        0.7604144459724388,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0026662579448282725,
        -0.006783951676700732,
        0.0,
        -0.039324414454589354,
        0.0,
        0.01588933739682598,
    ];

    tst(r.beta(), beta_exp, "beta");
    tst(
        r.residuals().expect("Missing residuals"),
        res_exp,
        "residuals",
    );
    tst(
        r.var_matrix().expect("Missing covariance matrix"),
        var_exp,
        "covariance matrix",
    );
    assert_eq!(
        ls.skip(),
        &[false, false, true, false, true, false],
        "Mismatch in skip vector"
    );
}

/// Make X'X and X'Y matrices from design matrix X and observation vector Y
/// X'X is symmetric and we store only the lower triangle.
/// X has m rows and p columns, and is stored by columns in a m * p vector.
fn make_xx_xy(m: usize, p: usize, x: &[f64], y: &[f64], xx: &mut [f64], xy: &mut [f64]) {
    let mut ix = 0;
    for i in 0..p {
        let x1 = &x[i * m..(i + 1) * m];
        for j in 0..i {
            xx[ix] = x[j * m..(j + 1) * m]
                .iter()
                .zip(x1)
                .map(|(a, b)| a * b)
                .sum();
            ix += 1;
        }
        let (z1, z2) = y
            .iter()
            .zip(x1)
            .fold((0.0, 0.0), |(s1, s2), (a, b)| (s1 + (b * b), s2 + (a * b)));
        xx[ix] = z1;
        ix += 1;
        xy[i] = z2;
    }
}

#[test]
fn make_xx_xy_works() {
    let x = vec![1.0, 1.0, 1.0, 1.0, 0.2, -0.3, 2.5, -1.6];
    let y = vec![23.1, 22.8, 27.3, 20.5];
    let mut xx = vec![0.0; 3];
    let mut xy = vec![0.0; 2];
    make_xx_xy(4, 2, &x, &y, &mut xx, &mut xy);

    let expected_xx = vec![4.0, 0.8, 8.94];
    let expected_xy = vec![93.7, 33.23];

    let tst = |x: &[f64], y: &[f64]| {
        let z: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(
            z < 1.0e-16,
            "Unexpected result from make_xx_xy_works (diff = {})",
            z
        );
    };
    tst(&xx, &expected_xx);
    tst(&xy, &expected_xy);
}

/// Make X'X and X'Y matrices from design matrix X and observation vector Y
/// X'X is symmetric and we store only the lower triangle.
/// Only columns where the corresponding element of skip are false will be used
/// X has m rows and p columns, and is stored by columns in a m * p vector.
/// Only columns where the corresponding element of skip are false will be used
/// If q is the number of false entries in skip, then the X'Y will be a q vector and
/// X'X will be a q * (q + 1) / 2 vector.
fn make_filtered_xx_xy(
    m: usize,
    p: usize,
    x: &[f64],
    y: &[f64],
    xx: &mut [f64],
    xy: &mut [f64],
    skip: &[bool],
) {
    let mut ix = 0;
    let mut i1 = 0;
    for i in 0..p {
        if !skip[i] {
            let x1 = &x[i * m..(i + 1) * m];
            for j in 0..i {
                if !skip[j] {
                    xx[ix] = x[j * m..(j + 1) * m]
                        .iter()
                        .zip(x1)
                        .map(|(a, b)| a * b)
                        .sum();
                    ix += 1;
                }
            }
            let (z1, z2) = y
                .iter()
                .zip(x1)
                .fold((0.0, 0.0), |(s1, s2), (a, b)| (s1 + (b * b), s2 + (a * b)));
            xx[ix] = z1;
            ix += 1;
            xy[i1] = z2;
            i1 += 1;
        }
    }
}

#[test]
fn make_filtered_xx_xy_works() {
    let x = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,
    ];

    let y = vec![2.0, 4.0, 3.0, 7.0, 6.0, 1.0, 8.0, 5.0];
    let mut xx = vec![0.0; 21];
    let mut xy = vec![0.0; 6];
    let skip = vec![false, true, false, false, true, false];

    make_filtered_xx_xy(8, 6, &x, &y, &mut xx, &mut xy, &skip);

    let expected_xx = vec![8.0, 3.0, 3.0, 3.0, 0.0, 3.0, 5.0, 3.0, 1.0, 5.0];
    let expected_xy = vec![36.0, 16.0, 14.0, 26.0];

    let tst = |x: &[f64], y: &[f64]| {
        let z: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(
            z < 1.0e-16,
            "Unexpected result from make_xx_xy_works (diff = {})",
            z
        );
    };
    tst(&xx, &expected_xx);
    tst(&xy, &expected_xy);
}

fn calc_residuals(x: &[f64], y: &[f64], beta: &[f64], fit: &mut [f64], res: &mut [f64]) {
    let m = y.len();
    for (ix, ((f, r), y)) in fit.iter_mut().zip(res.iter_mut()).zip(y).enumerate() {
        let z: f64 = x[ix..]
            .iter()
            .step_by(m)
            .zip(beta)
            .map(|(x, b)| x * b)
            .sum();
        *f = z;
        *r = *y - z;
    }
}

fn calc_filtered_residuals(
    x: &[f64],
    y: &[f64],
    beta: &[f64],
    fit: &mut [f64],
    res: &mut [f64],
    skip: &[bool],
) {
    let m = y.len();
    for (ix, ((f, r), y)) in fit.iter_mut().zip(res.iter_mut()).zip(y).enumerate() {
        let z: f64 = x[ix..]
            .iter()
            .step_by(m)
            .enumerate()
            .filter(|(j, _)| !skip[*j])
            .zip(beta)
            .map(|((_, x), b)| x * b)
            .sum();
        *f = z;
        *r = *y - z;
    }
}

pub struct LeastSquaresResult<'a> {
    beta: &'a [f64],
    chol: &'a [f64],
    fit: Option<&'a [f64]>,
    res: Option<&'a [f64]>,
    var: Option<&'a [f64]>,
    rss: Option<f64>,
    df: usize,
}

impl<'a> LeastSquaresResult<'a> {
    pub fn beta(&self) -> &'a [f64] {
        self.beta
    }
    pub fn residuals(&self) -> Option<&'a [f64]> {
        self.res
    }
    pub fn fitted_values(&self) -> Option<&'a [f64]> {
        self.fit
    }
    pub fn cholesky(&self) -> &'a [f64] {
        self.chol
    }
    pub fn df(&self) -> usize {
        self.df
    }
    pub fn rss(&self) -> Option<f64> {
        self.rss
    }
    pub fn res_var(&self) -> Option<f64> {
        if self.df > 0 {
            self.rss.map(|s| s / (self.df as f64))
        } else {
            None
        }
    }
    pub fn var_matrix(&self) -> Option<&[f64]> {
        self.var
    }
}

/*
#[derive(Default)]
pub struct WeightedLeastSquares {
    work: Vec<f32>,
    lsq: LeastSquares,
}

impl WeightedLeastSquares {
    pub fn new() -> Self {
        Self::default()
    }

    /// Weighted least squares - v contains the variance of each sample and is used to
    /// calculate the weights.  x and y are the same as for [LeastSquares::least_squares()]
    pub fn weighted_least_sqaures(
        &mut self,
        x: &[f32],
        y: &[f32],
        v: &[f32],
    ) -> anyhow::Result<(&[f32], &[f32])> {
        let (n, m, _) = get_sizes(x, y);
        assert_eq!(v.len(), m, "y and v unequal size");

        let work_size = n + m;
        self.work.resize(work_size, 0.0);

        let (wv, x2) = self.work.split_at_mut(m);
        x2.copy_from_slice(x);

        for (i, (yi, vi)) in y.iter().copied().zip(v.iter().copied()).enumerate() {
            let wt = (1.0 / vi).sqrt();
            wv[i] = yi * wt;
            for xj in x2[i..].iter_mut().step_by(m) {
                *xj *= wt
            }
        }

        self.lsq
            .least_squares(x2, wv)
            .with_context(|| "Weighted least squares - error from least squares")
    }
}

 */
