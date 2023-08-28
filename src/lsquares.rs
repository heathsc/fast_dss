use crate::cholesky::*;
use anyhow::Context;

#[derive(Default, Copy, Clone)]
pub enum LeastSquaresOptions {
    #[default]
    All,
    NoVar,
    NoRes,
}

#[derive(Default)]
pub struct LeastSquares {
    // general workspace
    // X'X lower diagonal - p * (p + 1) / 2
    // L (Cholesky decomposition of X'X) lower diagonal - p * (p + 1) / 2
    // X'Y - p
    // Beta (effects) - p
    // Residuals - m
    work: Vec<f64>,
    n_effects: usize, // p
    n_samples: usize, // m
    options: LeastSquaresOptions,
}

impl LeastSquares {
    fn slices_mut(&mut self) -> (&mut [f64], &mut [f64], &mut [f64], &mut [f64], &mut [f64]) {
        assert!(!self.work.is_empty());
        let p = self.n_effects;
        let k = (p * (p + 1)) >> 1;
        let (xx, t) = self.work.split_at_mut(k);
        let (l, t) = t.split_at_mut(k);
        let (xy, t) = t.split_at_mut(p);
        let (beta, residuals) = t.split_at_mut(p);
        (xx, l, xy, beta, residuals)
    }

    fn slices(&self) -> (&[f64], &[f64], &[f64], &[f64], &[f64]) {
        assert!(!self.work.is_empty());
        let p = self.n_effects;
        let k = (p * (p + 1)) >> 1;
        let (xx, t) = self.work.split_at(k);
        let (l, t) = t.split_at(k);
        let (xy, t) = t.split_at(p);
        let (beta, residuals) = t.split_at(p);
        (xx, l, xy, beta, residuals)
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn no_var(&mut self) {
        self.options = LeastSquaresOptions::NoVar
    }
    pub fn no_res(&mut self) {
        self.options = LeastSquaresOptions::NoRes
    }

    fn setup_workspace(&mut self, x: &[f64], y: &[f64]) {
        let m = y.len();
        let n = x.len();
        let p = n / m;
        assert_eq!(p * m, n, "Incorrect size for design matrix");

        // Reallocate workspace if required
        let work_size = p * (p + 3) + m;
        self.work.resize(work_size, 0.0);
        self.n_samples = m;
        self.n_effects = p;
    }

    /// m - number of samples
    /// p - number of effects
    /// Design matrix x has m rows and p columns stored in a m * p vector ordered by columns
    /// (i.e., column 1, column 2 etc.)
    /// y is the vector of observations and has m elements
    pub fn least_squares(&mut self, x: &[f64], y: &[f64]) -> anyhow::Result<LeastSquaresResult> {
        self.setup_workspace(x, y);
        let options = self.options;
        let (m, p) = (self.n_samples, self.n_effects);

        let (xx, l, xy, beta, res_store) = self.slices_mut();
        make_xx_xy(m, p, x, y, xx, xy);
        cholesky(xx, l, p).with_context(|| "Error returned from Cholesy Decomposition")?;
        cholesky_solve(l, xy, beta);

        let res = match options {
            LeastSquaresOptions::NoRes => None,
            _ => {
                calc_residuals(x, y, beta, res_store);
                Some(res_store as &[f64])
            }
        };

        let rss = res.map(|r| r.iter().map(|e| e * e).sum());

        let var = match options {
            LeastSquaresOptions::All if m > p => {
                let res_var = rss.unwrap() / ((m - p) as f64);
                let v = xx;
                cholesky_inverse(l, v, p);
                for z in v.iter_mut() {
                    *z *= res_var;
                }
                Some(v as &[f64])
            }
            _ => None,
        };

        Ok(LeastSquaresResult {
            beta,
            chol: l,
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
        let z: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(
            z < 1.0e-16,
            "Unexpected result from make_xx_xy_works for {s} (diff = {})",
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

fn calc_residuals(x: &[f64], y: &[f64], beta: &[f64], res: &mut [f64]) {
    let m = y.len();
    for (ix, (r, y)) in res.iter_mut().zip(y).enumerate() {
        let z: f64 = x[ix..]
            .iter()
            .step_by(m)
            .zip(beta)
            .map(|(x, b)| x * b)
            .sum();
        *r = *y - z;
    }
}

pub struct LeastSquaresResult<'a> {
    beta: &'a [f64],
    chol: &'a [f64],
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
