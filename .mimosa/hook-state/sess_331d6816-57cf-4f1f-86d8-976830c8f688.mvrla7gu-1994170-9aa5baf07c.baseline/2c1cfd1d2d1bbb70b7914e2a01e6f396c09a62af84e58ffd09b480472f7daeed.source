//! CPU token sampling on a logits vector: greedy at temperature 0, else temperature + top-p with a seeded RNG.

use super::engine::GenParams;

/// splitmix64: tiny, seedable, good enough for sampling
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    rng: Rng,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            t.as_nanos() as u64 ^ (std::process::id() as u64) << 32
        });
        Self { temperature, top_p, rng: Rng::new(seed) }
    }

    /// Whether an accelerator can sample the untruncated distribution directly.
    pub fn untruncated(&self) -> bool {
        self.temperature > 0.0 && self.temperature.is_finite() && !(self.top_p < 1.0)
    }

    /// A categorical draw for an accelerator that implements this sampler's distribution.
    pub fn uniform_draw(&mut self) -> f64 {
        self.rng.next_f64()
    }

    pub fn from_params(p: &GenParams) -> Self {
        Self::new(p.temperature, p.top_p, p.seed)
    }

    /// The sampling distribution: (token, weight) pairs and their total weight, or None for greedy. Only tokens within
    /// 30 temperature-scaled logits of the max are candidates (each other token weighs < e^-30 of the top one: under
    /// 1.2e-8 of the mass over the whole vocabulary). With top_p < 1 the support is the nucleus, heaviest first.
    fn dist(&self, logits: &[f32]) -> Option<(Vec<(u32, f64)>, f64)> {
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return None;
        }
        let mx = logits.iter().copied().filter(|v| !v.is_nan()).fold(f32::NEG_INFINITY, f32::max);
        if !mx.is_finite() {
            return None;
        }
        let th = mx - 30.0 * self.temperature;
        let inv_t = 1.0 / self.temperature as f64;
        let mut c: Vec<(u32, f64)> = logits.iter().enumerate().filter(|&(_, &l)| l >= th).map(|(i, &l)| (i as u32, ((l - mx) as f64 * inv_t).exp())).collect();
        let sum: f64 = c.iter().map(|x| x.1).sum();
        let top_p = self.top_p as f64;
        if !(top_p < 1.0) {
            return Some((c, sum));
        }
        // nucleus: grow a sorted top-k until it covers top_p of the mass (usually a few dozen tokens)
        let target = top_p.max(0.0) * sum;
        let mut k = 64.min(c.len());
        loop {
            if k < c.len() {
                c.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            }
            c[..k].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut acc = 0.0;
            for j in 0..k {
                acc += c[j].1;
                if acc >= target {
                    c.truncate(j + 1);
                    return Some((c, acc));
                }
            }
            if k == c.len() {
                return Some((c, acc));
            }
            k = (k * 4).min(c.len());
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let Some((c, mass)) = self.dist(logits) else { return argmax(logits) };
        let mut r = self.rng.next_f64() * mass;
        for &(i, w) in &c {
            r -= w;
            if r < 0.0 {
                return i;
            }
        }
        c.last().map(|x| x.0).unwrap_or_else(|| argmax(logits))
    }
}

impl Sampler {
    /// Speculative sampling against a deterministic draft token `d` (prompt lookup or an argmax draft): accept `d`
    /// with its probability under this sampler's distribution (temperature, top-p); otherwise draw from that
    /// distribution with `d` removed and renormalized. The result is distributed exactly as `sample(logits)`.
    /// Returns (accepted, token).
    pub fn accept_or_resample(&mut self, logits: &[f32], d: u32) -> (bool, u32) {
        let Some((c, mass)) = self.dist(logits) else {
            let a = argmax(logits);
            return (a == d, a);
        };
        let wd = c.iter().find(|x| x.0 == d).map(|x| x.1).unwrap_or(0.0);
        if wd > 0.0 && self.rng.next_f64() < wd / mass {
            return (true, d);
        }
        // residual: the distribution without d
        let rest = mass - wd;
        if rest <= 0.0 {
            return (true, d);
        }
        let mut r = self.rng.next_f64() * rest;
        let mut last = d;
        for &(i, w) in &c {
            if i == d {
                continue;
            }
            last = i;
            r -= w;
            if r < 0.0 {
                return (false, i);
            }
        }
        (false, last)
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0 as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_matches_sampling() {
        // accept_or_resample's token distribution equals sample's, whatever the draft
        let l = logits();
        for (t, p) in [(1.0f32, 1.0f32), (0.7, 0.9)] {
            let mut a = Sampler::new(t, p, Some(7));
            let mut b = Sampler::new(t, p, Some(9));
            let n = 200_000;
            let ha = hist(&mut a, &l, n);
            let mut hb = std::collections::HashMap::new();
            for k in 0..n {
                let d = [17u32, 400, 3, 5][k % 4];
                *hb.entry(b.accept_or_resample(&l, d).1).or_insert(0usize) += 1;
            }
            for tok in [17u32, 400, 3, 999] {
                let (x, y) = (*ha.get(&tok).unwrap_or(&0) as f64 / n as f64, *hb.get(&tok).unwrap_or(&0) as f64 / n as f64);
                assert!((x - y).abs() < 0.006, "t {t} p {p} token {tok}: {x} vs {y}");
            }
        }
    }

    fn logits() -> Vec<f32> {
        // probabilities at T=1 ~ softmax([3, 2, 1, 0, -1, ...])
        let mut l = vec![-10.0f32; 1000];
        l[17] = 3.0;
        l[400] = 2.0;
        l[3] = 1.0;
        l[999] = 0.0;
        l
    }

    fn hist(s: &mut Sampler, l: &[f32], n: usize) -> std::collections::HashMap<u32, usize> {
        let mut h = std::collections::HashMap::new();
        for _ in 0..n {
            *h.entry(s.sample(l)).or_insert(0) += 1;
        }
        h
    }

    #[test]
    fn greedy_is_argmax() {
        let l = logits();
        let mut s = Sampler::new(0.0, 0.9, Some(1));
        for _ in 0..10 {
            assert_eq!(s.sample(&l), 17);
        }
        assert_eq!(argmax(&[f32::NAN, 1.0, 2.0, -1.0]), 2);
    }

    #[test]
    fn seeded_is_deterministic() {
        let l = logits();
        let a: Vec<u32> = { let mut s = Sampler::new(1.0, 1.0, Some(42)); (0..200).map(|_| s.sample(&l)).collect() };
        let b: Vec<u32> = { let mut s = Sampler::new(1.0, 1.0, Some(42)); (0..200).map(|_| s.sample(&l)).collect() };
        let c: Vec<u32> = { let mut s = Sampler::new(1.0, 1.0, Some(43)); (0..200).map(|_| s.sample(&l)).collect() };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn temperature_matches_softmax() {
        let l = logits();
        let n = 200_000;
        let h = hist(&mut Sampler::new(1.0, 1.0, Some(7)), &l, n);
        let z: f64 = l.iter().map(|&x| (x as f64).exp()).sum();
        for i in [17u32, 400, 3, 999] {
            let p = (l[i as usize] as f64).exp() / z;
            let got = *h.get(&i).unwrap_or(&0) as f64 / n as f64;
            assert!((got - p).abs() < 0.01, "token {i}: {got} vs {p}");
        }
        // T=0.5 sharpens: the top token gets more mass than at T=1
        let h2 = hist(&mut Sampler::new(0.5, 1.0, Some(7)), &l, 20_000);
        assert!(h2[&17] as f64 / 20_000.0 > 0.85);
    }

    #[test]
    fn top_p_truncates() {
        let l = logits();
        // p(17) ~ 0.64, p(400) ~ 0.24: top_p 0.5 keeps only 17, 0.8 keeps {17, 400}
        let h = hist(&mut Sampler::new(1.0, 0.5, Some(3)), &l, 5_000);
        assert_eq!(h.len(), 1);
        assert!(h.contains_key(&17));
        let h = hist(&mut Sampler::new(1.0, 0.8, Some(3)), &l, 20_000);
        assert_eq!(h.keys().copied().collect::<std::collections::BTreeSet<_>>(), [17, 400].into());
        let r = h[&17] as f64 / h[&400] as f64; // renormalized ratio stays e^1
        assert!((r - std::f64::consts::E).abs() < 0.2, "{r}");
    }

    #[test]
    fn top_p_grows_past_initial_k() {
        // flat distribution over 129k tokens: the nucleus is far larger than the first top-k window
        let l = vec![0.0f32; 129_280];
        let mut s = Sampler::new(1.0, 0.95, Some(9));
        let mut seen = std::collections::HashSet::new();
        for _ in 0..300 {
            let t = s.sample(&l);
            assert!((t as usize) < l.len());
            seen.insert(t);
        }
        assert!(seen.len() > 290);
    }
}
