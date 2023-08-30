use libm::lgamma;

pub struct LogFactorial {
    cache_size: usize,
    cache: Vec<f64>,
}

impl LogFactorial {
    pub fn new(cache_size: usize) -> Self {
        let mut cache = Vec::with_capacity(cache_size + 1);
        cache.push(0.0);
        let mut z = 0.0;
        for i in 1..=cache_size {
            z += (i as f64).ln();
            cache.push(z)
        }
        Self { cache_size, cache }
    }

    pub fn lfact(&self, a: usize) -> f64 {
        self.cache
            .get(a)
            .copied()
            .unwrap_or_else(|| lgamma((a + 1) as f64))
    }

    pub fn lchoose(&self, n: usize, k: usize) -> f64 {
        assert!(k <= n);
        self.lfact(n) - self.lfact(k) - self.lfact(n - k)
    }

    /// Missing the constant part: -lbeta(alpha, beta)
    pub(super) fn dbetabinomial(&self, alpha: f64, beta: f64, depth: usize, y: usize) -> f64 {
        self.lchoose(depth, y) + lbeta(alpha + y as f64, beta + (depth - y) as f64)
    }
}

pub fn lbeta(alpha: f64, beta: f64) -> f64 {
    lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
}

mod test {
    use super::*;

    #[test]
    fn lfact_test() {
        let lf = LogFactorial::new(100);
        assert_eq!(lf.lfact(5), lgamma(6.0));
        assert_eq!(lf.lfact(150), lgamma(151.0));
    }

    #[test]
    fn lchoose_test() {
        let lf = LogFactorial::new(10);
        let z = lf.lchoose(10, 3) - 4.787_491_742_782_046;
        assert!(z.abs() < f64::EPSILON.sqrt())
    }

    #[test]
    fn lbeta_test() {
        let z = lbeta(8.0, 4.0) + 7.185_387_015_580_416_5;
        assert!(z.abs() < f64::EPSILON.sqrt())
    }
}
