use anyhow::Context;

/// Cholesky decomposition.  
/// Input: Lower triangle of input matrix stored as a n * (n + 1) / 2 vector
/// Output: Lower triangle of decomposition stored as a n * (n + 1) / 2 vector
/// n: matrix dimension
pub fn cholesky(a: &[f64], l: &mut [f64], n: usize) -> anyhow::Result<()> {
    assert_eq!(
        a.len(),
        l.len(),
        "Input and output vectors are different sizes"
    );
    assert_eq!(
        a.len(),
        (n * (n + 1)) >> 1,
        "Vector size does not correspond to matrix dimension"
    );

    let mut ix = 0;
    for i in 0..n {
        let ix1 = ix + i + 1;
        let mut ix2 = 0;
        for (j, a1) in a[ix..ix1].iter().enumerate() {
            let sum: f64 = (0..j).map(|k| l[ix + k] * l[ix2 + k]).sum();
            l[ix + j] = if i == j {
                check_sqrt(*a1 - sum).with_context(|| {
                    format!("cholesky(): Matrix not positive definite at index {}", i)
                })?
            } else {
                (*a1 - sum) / l[ix2 + j]
            };
            ix2 += j + 1;
        }
        ix = ix1;
    }

    Ok(())
}

#[test]
fn cholesky_works() {
    let a = vec![4.0, 12.0, 37.0, -16.0, -43.0, 98.0, 8.0, 27.0, 10.0, 110.0];
    let mut l = vec![0.0; 10];
    cholesky(&a, &mut l, 4).expect("Cholesky Error");
    let expected_res = vec![2.0, 6.0, 1.0, -8.0, 5.0, 3.0, 4.0, 3.0, 9.0, 2.0];
    let z: f64 = l
        .iter()
        .zip(expected_res.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    assert!(
        z < 1.0e-16,
        "Unexpected result from cholesky (diff = {})",
        z
    );
}

/// Solve A.x = y with solutions being returned in x.  l is the lower triangle of the Cholesky
/// decomposition of A.
pub fn cholesky_solve(l: &[f64], y: &[f64], x: &mut [f64]) {
    let n = y.len();

    assert_eq!(n, x.len(), "Input and output vectors are different sizes");

    assert_eq!(
        l.len(),
        (n * (n + 1)) >> 1,
        "Vector size does not correspond to matrix dimension"
    );

    let mut ix = 0;
    for i in 0..n {
        let ix1 = ix + i + 1;
        let sum: f64 = l[ix..ix1].iter().zip(x.iter()).map(|(a, b)| *a * *b).sum();
        x[i] = (y[i] - sum) / l[ix1 - 1];
        ix = ix1;
    }

    for i in (0..n).rev() {
        let mut sum = x[i];
        let mut ix2 = ix + i;
        for (k, b) in x[i + 1..n].iter().enumerate() {
            sum -= *b * l[ix2];
            ix2 += k + i + 2;
        }
        x[i] = sum / l[ix - 1];
        ix -= i + 1;
    }
}

#[test]
fn cholesky_solve_works() {
    let l = vec![2.0, 6.0, 1.0, -8.0, 5.0, 3.0, 4.0, 3.0, 9.0, 2.0];
    let y = vec![4.0, -2.0, 8.0, 13.0];
    let mut x = vec![0.0; 4];
    cholesky_solve(&l, &y, &mut x);

    let exp_res = vec![114442.0, -27764.0, 6721.0, -2115.0];
    let z: f64 = x
        .iter()
        .zip(exp_res.iter())
        .map(|(a, b)| (*a - (*b) / 36.0).powi(2))
        .sum();
    assert!(
        z < 1.0e-16,
        "Unexpected result from cholesky_solve (diff = {})",
        z
    );
}

pub fn cholesky_inverse(l: &[f64], z: &mut [f64], n: usize) {
    assert_eq!(
        l.len(),
        z.len(),
        "Input and output vectors are different sizes"
    );
    assert_eq!(
        l.len(),
        (n * (n + 1)) >> 1,
        "Vector size does not correspond to matrix dimension"
    );

    // First calculate L^(-1)
    let mut ix = 0;
    for i in 0..n {
        z[ix + i] = 1.0 / l[ix + i];
        let mut ix1 = ix;
        for j in i + 1..n {
            ix1 += j;
            let mut ix2 = ix + i;
            let sum: f64 = (i..j)
                .map(|k| {
                    let t = l[ix1 + k] * z[ix2];
                    ix2 += k + 1;
                    t
                })
                .sum();
            z[ix1 + i] = -sum / l[ix1 + j];
        }
        ix += i + 1;
    }

    // The inverse is (L^(-1))' , L^(-1)
    ix = 0;
    for i in 0..n {
        for j in 0..i {
            let mut ix1 = ix;
            let sum: f64 = (i..n)
                .map(|k| {
                    let t = z[ix1 + j] * z[ix1 + i];
                    ix1 += k + 1;
                    t
                })
                .sum();
            z[ix + j] = sum;
        }
        let mut ix1 = ix;
        let sum: f64 = (i..n)
            .map(|k| {
                let t = z[ix1 + i].powi(2);
                ix1 += k + 1;
                t
            })
            .sum();
        z[ix + i] = sum;
        ix += i + 1;
    }
}

#[test]
fn cholesky_inverse_works() {
    let a = vec![4.0, 12.0, 37.0, -16.0, -43.0, 98.0, 8.0, 27.0, 10.0, 110.0];
    let mut l = vec![0.0; 10];
    cholesky(&a, &mut l, 4).expect("Cholesky Error");
    let mut inv = vec![0.0; 10];
    cholesky_inverse(&l, &mut inv, 4);

    let expected_res = vec![
        24277.0 / 36.0,
        -5888.0 / 36.0,
        1432.0 / 36.0,
        1426.0 / 36.0,
        -344.0 / 36.0,
        85.0 / 36.0,
        -12.5,
        3.0,
        -0.75,
        0.25,
    ];

    let z: f64 = inv
        .iter()
        .zip(expected_res.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    assert!(
        z < 1.0e-16,
        "Unexpected result from cholesky_inverse (diff = {})",
        z
    );
}

fn check_sqrt(x: f64) -> anyhow::Result<f64> {
    if x > 0.0 {
        Ok(x.sqrt())
    } else {
        Err(anyhow!("Pivot is not > 0.0 ({})", x))
    }
}
