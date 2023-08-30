use crate::cholesky::*;
use anyhow::Context;

pub const LS_NO_RES: u16 = 1; // Do not calculate residuals (implies LS_NO_VAR)
pub const LS_NO_VAR: u16 = 2; // Do not calculate Variance matrix of effects
pub const LS_FILTER: u16 = 4; // Filter columns of X
pub const LS_FILTER_SET: u16 = 8; // Filter has been set
pub const LS_NO_RECHECK: u16 = 16; // Do not recheck filter on new input
pub const LS_MASK: u16 = 31;

struct LeastSquaresWork<'a> {
    xx: &'a mut [f64],
    l: &'a mut [f64],
    xy: &'a mut [f64],
    beta: &'a mut [f64],
    residuals: &'a mut [f64],
    fitted_values: &'a mut [f64],
    skip: &'a mut [bool],
    m: usize, // n_samples
    p: usize, // n_effects
}

impl<'a> LeastSquaresWork<'a> {
    fn from_work_slice(work: &'a mut [f64], skip: &'a mut [bool], m: usize, p: usize) -> Self {
        let k = (p * (p + 1)) >> 1;
        let (xx, t) = work.split_at_mut(k);
        let (l, t) = t.split_at_mut(k);
        let (xy, t) = t.split_at_mut(p);
        let (beta, t) = t.split_at_mut(p);
        let (residuals, fitted_values) = t.split_at_mut(m);
        LeastSquaresWork {
            xx,
            l,
            xy,
            beta,
            residuals,
            fitted_values,
            skip,
            m,
            p,
        }
    }
}

fn least_squares_work_size(n_samples: usize, n_effects: usize) -> usize {
    n_effects * (n_effects + 3) + 2 * n_samples
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
    fn slices_mut(&mut self) -> LeastSquaresWork {
        assert!(!self.work.is_empty());
        LeastSquaresWork::from_work_slice(
            &mut self.work,
            &mut self.skip,
            self.n_samples,
            self.n_effects,
        )
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_size(n_samples: usize, n_effects: usize) -> Self {
        let mut ls = Self::new();
        ls.alloc_workspace(n_samples, n_effects);
        ls
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

        self.alloc_workspace(m, p)
    }

    fn alloc_workspace(&mut self, n_samples: usize, n_effects: usize) {
        let work_size = least_squares_work_size(n_samples, n_effects);
        // Reallocate workspace if required
        if self.work.len() != work_size {
            self.work.resize(work_size, 0.0);
            self.skip.resize(n_effects, false);
            self.flags &= !LS_FILTER_SET;
        }

        self.n_samples = n_samples;
        self.n_effects = n_effects;
    }

    /// Regular (unweighted) least squares)
    ///
    /// m - number of samples
    /// p - number of effects
    /// Design matrix x has m rows and p columns stored in a m * p vector ordered by columns
    /// (i.e., column 1, column 2 etc.)
    /// y is the vector of observations and has m elements
    pub fn ls(&mut self, x: &[f64], y: &[f64]) -> anyhow::Result<LeastSquaresResult> {
        self._ls(x, y, None)
    }

    /// Weighted least squares)
    ///
    /// m - number of samples
    /// p - number of effects
    /// wt = vector of sample weights
    /// Design matrix x has m rows and p columns stored in a m * p vector ordered by columns
    /// (i.e., column 1, column 2 etc.)
    /// y is the vector of observations and has m elements
    pub fn wls(&mut self, x: &[f64], y: &[f64], wt: &[f64]) -> anyhow::Result<LeastSquaresResult> {
        self._ls(x, y, Some(wt))
    }

    fn _ls(
        &mut self,
        x: &[f64],
        y: &[f64],
        wt: Option<&[f64]>,
    ) -> anyhow::Result<LeastSquaresResult> {
        if x.is_empty() {
            Err(anyhow!("Empty Design matrix"))
        } else {
            self.setup_workspace(x, y);
            if (self.flags & LS_FILTER) == 0 {
                self.least_squares_unfiltered(x, y, wt)
            } else {
                self.least_squares_filtered(x, y, wt)
            }
        }
    }

    fn check_filter(&mut self, x: &[f64], y: &[f64], wt: Option<&[f64]>) -> anyhow::Result<()> {
        let mut lsw = self.slices_mut();

        make_xx_xy(x, y, wt, &mut lsw);
        find_dependencies(lsw.xx, lsw.l, lsw.skip, lsw.p)
            .with_context(|| "Error returned from find dependencies step")?;
        self.flags |= LS_FILTER_SET;
        Ok(())
    }

    fn least_squares_filtered(
        &mut self,
        x: &[f64],
        y: &[f64],
        wt: Option<&[f64]>,
    ) -> anyhow::Result<LeastSquaresResult> {
        if (self.flags & (LS_FILTER_SET | LS_NO_RECHECK)) != (LS_FILTER_SET | LS_NO_RECHECK) {
            self.check_filter(x, y, wt)?;
        }
        let flags = self.flags;
        let check_flags = |x: u16| (flags & x) != 0;

        let (m, p) = (self.n_samples, self.n_effects);
        let p_used: usize = self.skip.iter().filter(|x| !**x).count();
        if p_used == 0 {
            return Err(anyhow!("Design matrix is empty"));
        }

        let lsw = self.slices_mut();

        let sz = (p_used * (p_used + 1)) >> 1;

        make_filtered_xx_xy(
            m,
            p,
            x,
            y,
            &mut lsw.xx[..sz],
            &mut lsw.xy[..p_used],
            lsw.skip,
        );

        cholesky(&lsw.xx[..sz], &mut lsw.l[..sz], p_used)
            .with_context(|| "Error returned from Cholesky Decomposition")?;
        cholesky_solve(&lsw.l[..sz], &lsw.xy[..p_used], &mut lsw.beta[..p_used]);

        let (fit, res) = if check_flags(LS_NO_RES) {
            (None, None)
        } else {
            calc_filtered_residuals(
                x,
                y,
                &lsw.beta[..p_used],
                lsw.fitted_values,
                lsw.residuals,
                lsw.skip,
            );
            (
                Some(lsw.fitted_values as &[f64]),
                Some(lsw.residuals as &[f64]),
            )
        };

        // If necessary, expand beta vector to account for skipped columns
        if p_used < p {
            let mut ix = p_used;
            for (i, s) in lsw.skip.iter().enumerate().rev() {
                lsw.beta[i] = if *s {
                    0.0
                } else {
                    ix -= 1;
                    lsw.beta[ix]
                }
            }
        }

        let rss = res.map(|r| r.iter().map(|e| e * e).sum());

        let var = if !check_flags(LS_NO_RES | LS_NO_VAR) && m > p_used {
            let res_var = rss.unwrap() / ((m - p_used) as f64);
            let v = lsw.xx;
            cholesky_inverse(&lsw.l[..sz], &mut v[..sz], p_used);
            for z in v.iter_mut() {
                *z *= res_var;
            }
            // If necessary, expand var vector to account for skipped columns
            if p_used < p {
                let mut ix = sz;
                let mut ix1 = (p * (p + 1)) >> 1;
                for (i, s) in lsw.skip.iter().enumerate().rev() {
                    ix1 -= i + 1;
                    if *s {
                        for x in v[ix1..ix1 + i + 1].iter_mut() {
                            *x = 0.0;
                        }
                    } else {
                        for j in (0..=i).rev() {
                            v[ix1 + j] = if lsw.skip[j] {
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
            beta: lsw.beta,
            chol: lsw.l,
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
        wt: Option<&[f64]>,
    ) -> anyhow::Result<LeastSquaresResult> {
        let flags = self.flags;
        let check_flags = |x| (flags & x) != 0;

        let (m, p) = (self.n_samples, self.n_effects);

        let mut lsw = self.slices_mut();

        make_xx_xy(x, y, wt, &mut lsw);
        cholesky(lsw.xx, lsw.l, p).with_context(|| "Error returned from Cholesky Decomposition")?;
        cholesky_solve(lsw.l, lsw.xy, lsw.beta);

        let (fit, res) = if check_flags(LS_NO_RES) {
            (None, None)
        } else {
            calc_residuals(x, y, lsw.beta, lsw.fitted_values, lsw.residuals);
            (
                Some(lsw.fitted_values as &[f64]),
                Some(lsw.residuals as &[f64]),
            )
        };

        let rss = res.map(|r| r.iter().map(|e| e * e).sum());

        let var = if !check_flags(LS_NO_RES | LS_NO_VAR) && m > p {
            let res_var = rss.unwrap() / ((m - p) as f64);
            let v = lsw.xx;
            cholesky_inverse(lsw.l, v, p);
            for z in v.iter_mut() {
                *z *= res_var;
            }
            Some(v as &[f64])
        } else {
            None
        };

        Ok(LeastSquaresResult {
            beta: lsw.beta,
            chol: lsw.l,
            fit,
            res,
            var,
            rss,
            df: m - p,
        })
    }
}

/// Make X'X and X'Y matrices from design matrix X and observation vector Y
/// X'X is symmetric and we store only the lower triangle.
/// X has m rows and p columns, and is stored by columns in a m * p vector.
fn make_xx_xy(x: &[f64], y: &[f64], wt: Option<&[f64]>, lsw: &mut LeastSquaresWork) {
    let m = lsw.m;

    let mut ix = 0;

    if let Some(wt) = wt {
        // Weighted
        for i in 0..lsw.p {
            let x1 = &x[i * m..(i + 1) * m];
            for j in 0..i {
                lsw.xx[ix] = x[j * m..(j + 1) * m]
                    .iter()
                    .zip(x1)
                    .zip(wt)
                    .map(|((a, b), w)| a * b * w)
                    .sum();
                ix += 1;
            }
            let (z1, z2) = y
                .iter()
                .zip(x1)
                .zip(wt)
                .fold((0.0, 0.0), |(s1, s2), ((a, b), w)| {
                    (s1 + (b * b * w), s2 + (a * b * w))
                });
            lsw.xx[ix] = z1;
            ix += 1;
            lsw.xy[i] = z2;
        }
    } else {
        // Unweighted
        for i in 0..lsw.p {
            let x1 = &x[i * m..(i + 1) * m];
            for j in 0..i {
                lsw.xx[ix] = x[j * m..(j + 1) * m]
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
            lsw.xx[ix] = z1;
            ix += 1;
            lsw.xy[i] = z2;
        }
    }
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

#[derive(Default)]
pub struct WeightedLeastSquares {
    work: Vec<f64>,
    lsq: LeastSquares,
}

impl WeightedLeastSquares {
    pub fn new() -> Self {
        Self::default()
    }

    /// Weighted least squares - w contains the weight for each sample and is used to
    /// calculate the weights.  x and y are the same as for [LeastSquares::ls()]
    pub fn weighted_least_squares(
        &mut self,
        x: &[f64],
        y: &[f64],
        w: &[f64],
    ) -> anyhow::Result<LeastSquaresResult> {
        let m = y.len();
        assert_eq!(w.len(), m, "y and w unequal size");

        let work_size = x.len() + y.len();
        self.work.resize(work_size, 0.0);

        let (wv, x2) = self.work.split_at_mut(m);
        x2.copy_from_slice(x);

        for (i, (yi, wi)) in y.iter().copied().zip(w.iter().copied()).enumerate() {
            let wt = wi.sqrt();
            wv[i] = yi * wt;
            for xj in x2[i..].iter_mut().step_by(m) {
                *xj *= wt
            }
        }

        self.lsq
            .ls(x2, wv)
            .with_context(|| "Weighted least squares - error from least squares")
    }
}

mod test {
    use super::*;

    fn tst(x: &[f64], y: &[f64], s: &str) {
        assert_eq!(x.len(), y.len(), "{s}: Input vectors not equal length");
        let z: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(
            z < f64::EPSILON,
            "Unexpected result from make_xx_xy_works\n{:?}\n{:?}\n",
            x,
            y
        );
    }

    #[test]
    fn test_make_xx_xy() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.2, -0.3, 2.5, -1.6];
        let y = vec![23.1, 22.8, 27.3, 20.5];

        let p = 2;
        let m = y.len();
        let work_size = least_squares_work_size(m, p);
        let mut work = vec![0.0; work_size];
        let mut skip = Vec::new();
        let mut lsw = LeastSquaresWork::from_work_slice(&mut work, &mut skip, m, p);

        make_xx_xy(&x, &y, None, &mut lsw);

        let expected_xx = vec![4.0, 0.8, 8.94];
        let expected_xy = vec![93.7, 33.23];

        tst(lsw.xx, &expected_xx, "X'X");
        tst(lsw.xy, &expected_xy, "X'Y");
    }

    #[test]
    fn test_make_xx_xy_weighted() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.2, -0.3, 2.5, -1.6];
        let y = vec![23.1, 22.8, 27.3, 20.5];
        let wt = vec![0.8, 1.2, 0.75, 2.0];

        let p = 2;
        let m = y.len();
        let work_size = least_squares_work_size(m, p);
        let mut work = vec![0.0; work_size];
        let mut skip = Vec::new();
        let mut lsw = LeastSquaresWork::from_work_slice(&mut work, &mut skip, m, p);

        make_xx_xy(&x, &y, Some(&wt), &mut lsw);

        let expected_xx = vec![4.75, -1.525, 9.9475];
        let expected_xy = vec![107.315, -18.9245];

        tst(lsw.xx, &expected_xx, "X'WX");
        tst(lsw.xy, &expected_xy, "X'WY");
    }

    #[test]
    fn test_make_filtered_xx_xy() {
        let x = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,
        ];

        let y = vec![2.0, 4.0, 3.0, 7.0, 6.0, 1.0, 8.0, 5.0];
        let mut xx = vec![0.0; 21];
        let mut xy = vec![0.0; 6];
        let skip = vec![false, true, false, false, true, false];

        make_filtered_xx_xy(8, 6, &x, &y, &mut xx, &mut xy, &skip);

        let expected_xx = vec![8.0, 3.0, 3.0, 3.0, 0.0, 3.0, 5.0, 3.0, 1.0, 5.0];
        let expected_xy = vec![36.0, 16.0, 14.0, 26.0];

        tst(&xx[..10], &expected_xx, "X'X");
        tst(&xy[..4], &expected_xy, "X'Y");
    }
    #[test]
    fn test_weighted_least_squares() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.2, -0.3, 2.5, -1.6];
        let y = vec![23.1, 22.8, 27.3, 20.5];
        let w = vec![1.5, 0.7, 1.1, 0.8];

        let mut wls = WeightedLeastSquares::new();
        let r = wls
            .weighted_least_squares(&x, &y, &w)
            .expect("Error in least squares");

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
    fn least_squares_works() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.2, -0.3, 2.5, -1.6];
        let y = vec![23.1, 22.8, 27.3, 20.5];

        let mut ls = LeastSquares::new();
        let r = ls.ls(&x, &y).expect("Error in least squares");

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
            1.0, 1.0, 1.0, 1.0, 1.0, 0.4, -0.6, 5.0, -3.2, 0.3, 0.2, -0.3, 2.5, -1.6, 0.15, 1.0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -2.0, 7.2, 3.4, -1.1, -0.9,
        ];

        let y = vec![23.1, 22.8, 27.3, 20.5, 22.2];

        let mut ls = LeastSquares::new();
        ls.set_flags(LS_FILTER);
        let r = ls.ls(&x, &y).expect("Error in least squares");

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
}
