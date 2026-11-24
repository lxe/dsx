//! Speculative decoding drafts that cost nothing to make: continuations looked up in the context itself.

/// Up to `k` tokens that followed the most recent earlier occurrence of the context's last 3 (else 2) tokens.
pub fn lookup(ctx: &[u32], k: usize) -> Vec<u32> {
    for n in [3usize, 2] {
        if ctx.len() <= n {
            continue;
        }
        let tail = &ctx[ctx.len() - n..];
        // newest match first; the match must leave at least one token after it
        for s in (0..ctx.len() - n).rev() {
            if &ctx[s..s + n] == tail {
                let from = s + n;
                let to = (from + k).min(ctx.len());
                if to > from {
                    return ctx[from..to].to_vec();
                }
            }
        }
    }
    vec![]
}


/// Live control of speculative decoding, the same on any hardware: it measures what a plain step, a draft and a
/// verify pass of T tokens cost here, and how often each draft position hits, and before every token picks the draft
/// length with the most tokens per millisecond (0 = decode plainly). While plain decoding wins it drafts again (a
/// probe, 2 drafts) after a gap that doubles while probes do not pay (16 .. 256 tokens), and early when the main
/// model has grown as confident as it was when drafting last paid (text turning into code).
#[derive(Clone, Debug)]
pub struct SpecControl {
    /// ms per plain step, per draft, per verify pass of T tokens (index T, 1..=max+1)
    pub step_ms: f64,
    pub draft_ms: f64,
    pub verify_ms: Vec<f64>,
    /// hit rate of draft position i given the positions before it hit
    pub hit: Vec<f64>,
    pub max: usize,
    since_probe: usize,
    pub probe_gap: usize,
    probing: bool,
    /// the main model's top probability (EMA over tokens) and its level when drafting last paid
    pub conf: f64,
    pub conf_paid: f64,
    /// which verify sizes have been measured; unmeasured ones are extrapolated from them, so a size is never
    /// written off on its prior alone; every EXPLORE_EVERY passes the next longer draft is tried when close
    measured: Vec<bool>,
    passes: usize,
    /// tokens decoded since a pass last paid (learned across requests and restarts, see `Learned`)
    pub since_paid: u64,
    /// end to end, like the tiering: windows of WINDOW tokens alternate between drafting allowed (on) and not (off);
    /// the wall time per token of each (sampling, steps, passes, catch-ups, everything) decides which mode runs
    on: bool,
    win_tok: usize,
    win_ms: f64,
    windows: usize,
    /// ms per token, EMA over windows
    pub ms_on: Option<f64>,
    pub ms_off: Option<f64>,
    /// tokens decoded while drafting measured slower than not drafting
    pub unprofitable: u64,
    /// copy signal: trigrams of the prompt and of what was generated; consecutive tokens whose trigram was seen before
    seen3: std::collections::HashSet<u64>,
    last: [u32; 3],
    filled: usize,
    copy_run: usize,
    /// per draft position and confidence bin: [drafts that reached it, accepted]; calibrates DSpark's confidence head
    /// (its raw scores run overconfident) on this machine's own verify outcomes
    pub calib: Vec<Vec<[u32; 2]>>,
    /// the answer was copying when this decode turn was chosen: only such turns count for the windows and the unpaid
    /// tokens (new text never drafts, so it says nothing about DSpark and must not switch it off)
    turn_copying: bool,
}

/// consecutive repeated trigrams (a 4-token run copied from the prompt or the output so far) that count as copying
pub const COPY_RUN: usize = 2;

/// tokens per measuring window
pub const WINDOW: usize = 256;
/// every this many windows the mode that lost runs once more (it may have become the better one)
pub const RECHECK_EVERY: usize = 6;
/// while drafting is off but the answer copies, a 3-token probe at least this often (a fully accepted one that pays
/// turns drafting on)
pub const OFF_PROBE: usize = 128;

/// What the controller has learned, kept across restarts (DSX_STATE_DIR/dspark-control.json): like the expert tiering,
/// DSpark decides from measurements on this machine and this workload whether it is worth its memory.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Learned {
    pub probe_gap: usize,
    pub conf_paid: f64,
    pub hit: Vec<f64>,
    pub since_paid: u64,
    /// unix seconds until which the server leaves DSpark unloaded (its memory goes to experts)
    #[serde(default)]
    pub skip_until: u64,
    #[serde(default)]
    pub ms_on: Option<f64>,
    #[serde(default)]
    pub ms_off: Option<f64>,
    #[serde(default)]
    pub unprofitable: u64,
    #[serde(default)]
    pub calib: Vec<Vec<[u32; 2]>>,
}

/// confidence-head logits are binned per unit from -CONF_BINS/2 to CONF_BINS/2
pub const CONF_BINS: usize = 12;
/// weight of the raw sigmoid(logit) prior in a bin's acceptance estimate, in drafts
const CONF_PRIOR: f64 = 8.0;

/// decoded tokens without one paying pass after which DSpark gives its memory back at the next start
pub const UNPAID_SKIP_TOKENS: u64 = 30_000;
/// how long it then stays unloaded before it measures again
pub const SKIP_SECS: u64 = 7 * 24 * 3600;

pub const EXPLORE_EVERY: usize = 24;

pub const PROBE_MIN: usize = 16;
/// a probe that does not pay doubles the gap up to this (prose: about one probe per 2k tokens, ~1 % cost)
pub const PROBE_MAX: usize = 2048;

impl SpecControl {
    pub fn new(max: usize) -> SpecControl {
        // priors: a verify pass of T tokens ~ 1 + 0.5 (T - 1) steps, drafts at 0.4 steps, positions hit 80 %
        SpecControl {
            step_ms: 25.0,
            draft_ms: 10.0,
            verify_ms: (0..=max + 1).map(|t| 25.0 * (1.0 + 0.5 * (t.max(1) - 1) as f64)).collect(),
            hit: vec![0.8; max],
            max,
            since_probe: 0,
            probe_gap: PROBE_MIN,
            probing: false,
            conf: 0.0,
            conf_paid: 1.0,
            measured: vec![false; max + 2],
            passes: 0,
            since_paid: 0,
            on: true,
            win_tok: 0,
            win_ms: 0.0,
            windows: 0,
            ms_on: None,
            ms_off: None,
            unprofitable: 0,
            seen3: std::collections::HashSet::new(),
            last: [0; 3],
            filled: 0,
            copy_run: 0,
            turn_copying: false,
            calib: vec![vec![[0; 2]; CONF_BINS]; max],
        }
    }

    fn tri(a: u32, b: u32, c: u32) -> u64 {
        ((a as u64) << 42) ^ ((b as u64) << 21) ^ c as u64
    }

    /// A request starts: its prompt's trigrams are what the answer may copy
    pub fn begin(&mut self, prompt: &[u32]) {
        self.seen3.clear();
        for w in prompt.windows(3) {
            self.seen3.insert(Self::tri(w[0], w[1], w[2]));
        }
        self.filled = 0;
        self.copy_run = 0;
        self.turn_copying = false;
        for &t in prompt.iter().rev().take(3).rev() {
            self.push_last(t);
        }
    }

    fn push_last(&mut self, t: u32) {
        self.last = [self.last[1], self.last[2], t];
        self.filled = (self.filled + 1).min(3);
    }

    /// A generated token (sampled or an accepted draft)
    pub fn note_token(&mut self, t: u32) {
        self.push_last(t);
        if self.filled < 3 {
            return;
        }
        let k = Self::tri(self.last[0], self.last[1], self.last[2]);
        self.copy_run = if self.seen3.contains(&k) { self.copy_run + 1 } else { 0 };
        self.seen3.insert(k);
    }

    /// The answer is repeating its prompt or itself: where the draft's guesses come true (DSpark only pays there on a
    /// GPU whose verify pass costs several steps; R208/R209: code tests and refactors +4..30 %, new text -2..4 %)
    pub fn copying(&self) -> bool {
        self.copy_run >= COPY_RUN
    }

    /// End-to-end accounting: `n` tokens in `ms` of wall time in the current window. At the window's end its mode's
    /// ms per token is folded in and the next window's mode chosen: an unmeasured mode first, then the faster one, the
    /// slower once every RECHECK_EVERY windows.
    pub fn note_wall(&mut self, n: usize, ms: f64) {
        if n == 0 || !(ms > 0.0) || !self.turn_copying {
            return;
        }
        self.win_tok += n;
        self.win_ms += ms;
        if self.win_tok < WINDOW {
            return;
        }
        let per = self.win_ms / self.win_tok as f64;
        let slot = if self.on { &mut self.ms_on } else { &mut self.ms_off };
        *slot = Some(slot.map_or(per, |v| 0.5 * v + 0.5 * per));
        if let (Some(a), Some(b)) = (self.ms_on, self.ms_off) {
            if a > 1.01 * b {
                self.unprofitable += self.win_tok as u64;
            } else {
                self.unprofitable = 0;
            }
        }
        self.windows += 1;
        self.win_tok = 0;
        self.win_ms = 0.0;
        self.on = match (self.ms_on, self.ms_off) {
            (None, _) => true,
            (_, None) => false,
            (Some(a), Some(b)) => {
                let better_on = a <= b;
                if self.windows % RECHECK_EVERY == 0 { !better_on } else { better_on }
            }
        };
    }

    /// drafting allowed in the current window
    pub fn drafting(&self) -> bool {
        self.on
    }

    pub fn learned(&self) -> Learned {
        Learned { probe_gap: self.probe_gap, conf_paid: self.conf_paid, hit: self.hit.clone(), since_paid: self.since_paid, skip_until: 0, ms_on: self.ms_on, ms_off: self.ms_off, unprofitable: self.unprofitable, calib: self.calib.clone() }
    }

    /// continue from what an earlier run learned (sizes that do not fit are ignored)
    pub fn resume(&mut self, l: &Learned) {
        if (PROBE_MIN..=PROBE_MAX).contains(&l.probe_gap) {
            self.probe_gap = l.probe_gap;
        }
        if (0.0..=1.0).contains(&l.conf_paid) {
            self.conf_paid = l.conf_paid;
        }
        if l.hit.len() == self.hit.len() && l.hit.iter().all(|h| (0.0..=1.0).contains(h)) {
            self.hit = l.hit.clone();
        }
        self.since_paid = l.since_paid;
        let ok = |v: Option<f64>| v.filter(|x| x.is_finite() && *x > 0.0 && *x < 10_000.0);
        self.ms_on = ok(l.ms_on);
        self.ms_off = ok(l.ms_off);
        self.unprofitable = l.unprofitable;
        if l.calib.len() == self.calib.len() && l.calib.iter().all(|v| v.len() == CONF_BINS) {
            self.calib = l.calib.clone();
        }
        if let (Some(a), Some(b)) = (self.ms_on, self.ms_off) {
            self.on = a <= b;
        }
    }

    fn ema(x: &mut f64, v: f64, a: f64) {
        *x = (1.0 - a) * *x + a * v;
    }

    /// expected tokens per ms when drafting `l` (0 = plain step)
    pub fn rate(&self, l: usize) -> f64 {
        if l == 0 {
            return 1.0 / self.step_ms;
        }
        let (mut reach, mut exp) = (1.0, 1.0);
        for i in 0..l {
            reach *= self.hit[i];
            exp += reach;
        }
        exp / (self.draft_ms + self.verify_cost(l + 1))
    }

    /// ms for a verify pass of `t` tokens: measured, else on the line through the plain step (t = 1) and the
    /// measured sizes (least squares), else the prior
    pub fn verify_cost(&self, t: usize) -> f64 {
        if self.measured[t] {
            return self.verify_ms[t];
        }
        let pts: Vec<(f64, f64)> = std::iter::once((1.0, self.step_ms)).chain((2..self.verify_ms.len()).filter(|&u| self.measured[u]).map(|u| (u as f64, self.verify_ms[u]))).collect();
        if pts.len() < 2 {
            return self.verify_ms[t];
        }
        let n = pts.len() as f64;
        let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
        let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let slope = if sxx > 0.0 { pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / sxx } else { 0.0 };
        (my + slope.max(0.0) * (t as f64 - mx)).max(self.step_ms)
    }

    /// draft length for the next token (0 = plain), at most `room`
    pub fn choose(&mut self, room: usize) -> usize {
        // drafts only while the answer copies (no probes either: on new text they cost more than they find)
        self.turn_copying = self.copying();
        if !self.turn_copying {
            self.probing = false;
            return 0;
        }
        if !self.on {
            // drafting measured slower overall, but now copying: a short probe every OFF_PROBE tokens; one that is fully
            // accepted and pays switches drafting back on, and the windows measure it again
            self.since_probe += 1;
            if self.since_probe >= self.probe_gap.min(OFF_PROBE) {
                self.since_probe = 0;
                self.probing = true;
                return room.min(3);
            }
            self.probing = false;
            return 0;
        }
        let room = room.min(self.max);
        let rates: Vec<f64> = (0..=room).map(|l| self.rate(l)).collect();
        let top = rates.iter().cloned().fold(0.0, f64::max);
        // within 2 % the longer draft: its estimate rests on more evidence and the verify cost is flat per token
        let mut best = (0..=room).rev().find(|&l| rates[l] >= 0.98 * top).unwrap_or(0);
        self.probing = false;
        if best > 0 || room == 0 {
            // now and then one longer, if it is not clearly worse: the estimate for it then rests on a measurement
            if best < room && self.passes % EXPLORE_EVERY == EXPLORE_EVERY - 1 && rates[best + 1] >= 0.9 * rates[best] {
                best += 1;
            }
            self.since_probe = 0;
            return best;
        }
        self.since_probe += 1;
        // early, but still within the back-off: at most every quarter gap
        let confident = self.since_probe >= (self.probe_gap / 4).max(4) && self.conf >= self.conf_paid;
        if self.since_probe >= self.probe_gap || confident {
            self.since_probe = 0;
            self.probing = true;
            return room.min(2);
        }
        0
    }

    pub fn note_step(&mut self, ms: f64) {
        Self::ema(&mut self.step_ms, ms, 0.1);
        if self.turn_copying {
            self.since_paid += 1;
        }
    }

    fn bin(c: f32) -> usize {
        ((c.floor() as i64) + CONF_BINS as i64 / 2).clamp(0, CONF_BINS as i64 - 1) as usize
    }

    /// estimated probability that draft position `i` with confidence logit `c` is accepted, given the ones before it were
    pub fn accept_p(&self, i: usize, c: f32) -> f64 {
        let prior = 1.0 / (1.0 + (-c as f64).exp());
        let [n, h] = self.calib.get(i).map(|v| v[Self::bin(c)]).unwrap_or([0, 0]);
        (h as f64 + CONF_PRIOR * prior) / (n as f64 + CONF_PRIOR)
    }

    /// How many of `l` drafted tokens to verify, by DSpark's confidence per position: the length with the most expected
    /// tokens per verify ms (the draft itself is already spent; 0 = a plain step). Cuts drafts that would mostly be
    /// rejected, so a pass verifies fewer wasted tokens.
    pub fn cut(&self, conf: &[f32], l: usize) -> usize {
        // DSX_DSPARK_CUT=0: verify every chosen draft (for A/B comparisons)
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("DSX_DSPARK_CUT").is_ok_and(|v| v == "0")) {
            return l;
        }
        let l = l.min(conf.len());
        let (mut reach, mut exp) = (1.0, 1.0);
        let mut best = (0, 1.0 / self.step_ms);
        for i in 0..l {
            reach *= self.accept_p(i, conf[i]);
            exp += reach;
            let r = exp / self.verify_cost(i + 2);
            if r > best.1 {
                best = (i + 1, r);
            }
        }
        // a probe still verifies its first draft: a bin learned bad gets the chance to be learned good again
        if self.probing && l > 0 { best.0.max(1) } else { best.0 }
    }

    /// a verified pass: positions before `j` were accepted, position `j` (if verified) rejected
    pub fn note_cut(&mut self, conf: &[f32], l: usize, j: usize) {
        for i in 0..l.min(conf.len()).min(j + 1) {
            if let Some(v) = self.calib.get_mut(i) {
                let c = &mut v[Self::bin(conf[i])];
                c[0] += 1;
                c[1] += (i < j) as u32;
                if c[0] > 4000 {
                    c[0] /= 2;
                    c[1] /= 2;
                }
            }
        }
    }

    /// the main model's top probability for the token just decided (any path)
    pub fn note_conf(&mut self, pmax: f64) {
        Self::ema(&mut self.conf, pmax, 0.2);
    }

    /// a pass drafted `l` (taking `draft_ms`), verified them in `verify_ms`, accepted `j`
    pub fn note_pass(&mut self, l: usize, j: usize, draft_ms: f64, verify_ms: f64) {
        Self::ema(&mut self.draft_ms, draft_ms, 0.2);
        if self.measured[l + 1] {
            Self::ema(&mut self.verify_ms[l + 1], verify_ms, 0.2);
        } else {
            self.verify_ms[l + 1] = verify_ms;
            self.measured[l + 1] = true;
        }
        self.passes += 1;
        for i in 0..l.min(j + 1) {
            Self::ema(&mut self.hit[i], if i < j { 1.0 } else { 0.0 }, 0.15);
        }
        let paid = (1 + j) as f64 / (draft_ms + verify_ms) > 1.0 / self.step_ms;
        if paid {
            Self::ema(&mut self.conf_paid, self.conf, 0.3);
            self.probe_gap = PROBE_MIN;
            self.since_paid = 0;
        } else {
            self.since_paid += (1 + j) as u64;
        }
        if !paid && self.probing {
            self.probe_gap = (self.probe_gap * 2).min(PROBE_MAX);
        }
        if paid && self.probing && !self.on && j == l {
            self.on = true;
            self.win_tok = 0;
            self.win_ms = 0.0;
        }
        self.probing = false;
    }
}

/// The top probability of a logit vector (softmax max), for SpecControl::note_conf
/// (every token: only logits within 16 of the max are exponentiated, the rest weigh < 1e-7 each)
pub fn top_prob(logits: &[f32]) -> f64 {
    let mx = logits.iter().cloned().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    if !mx.is_finite() {
        return 0.0;
    }
    let th = mx - 16.0;
    let sum: f32 = logits.iter().filter(|&&v| v > th).map(|&v| (v - mx).exp()).sum();
    1.0 / (sum as f64).max(1.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn finds_the_newest_continuation() {
        let c = [1, 2, 3, 9, 9, 1, 2, 3, 7, 8, 5, 1, 2, 3];
        assert_eq!(super::lookup(&c, 2), vec![7, 8]);
        assert_eq!(super::lookup(&[4, 5, 6], 3), Vec::<u32>::new());
    }

    #[test]
    fn control_drafts_when_it_pays_and_stops_when_not() {
        let mut c = super::SpecControl::new(5);
        c.copy_run = super::COPY_RUN;
        c.turn_copying = true;
        c.step_ms = 25.0;
        c.draft_ms = 10.0;
        c.verify_ms = vec![25.0, 25.0, 58.0, 70.0, 82.0, 88.0, 95.0];
        c.hit = vec![0.95, 0.93, 0.88, 0.87, 0.85];
        assert_eq!(c.choose(5), 5);
        c.hit = vec![0.65, 0.5, 0.45, 0.5, 0.4];
        assert_eq!(c.choose(5), 0);
        // probes now and then, less often while they do not pay
        let first = (1..=20).find(|_| c.choose(5) > 0).unwrap();
        assert!(first <= 16);
        c.note_pass(2, 0, 10.0, 58.0);
        assert_eq!(c.probe_gap, 32);
        // the main model as confident as when drafting paid: probe early
        c.conf_paid = 0.8;
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        c.note_conf(1.0);
        assert!((1..=8).any(|_| c.choose(5) > 0));
        assert!((super::top_prob(&[0.0, 0.0]) - 0.5).abs() < 1e-9);
        // unmeasured sizes follow the measured ones: T=6 is not written off on a pessimistic prior
        let mut c = super::SpecControl::new(5);
        c.copy_run = super::COPY_RUN;
        c.turn_copying = true;
        c.step_ms = 25.0;
        c.verify_ms[6] = 200.0;
        c.note_pass(3, 3, 9.0, 50.0);
        c.note_pass(4, 4, 9.0, 60.0);
        assert!(c.verify_cost(6) < 80.0, "{}", c.verify_cost(6));
    }

    #[test]
    fn learns_across_restarts_and_counts_unpaid_tokens() {
        let mut c = super::SpecControl::new(5);
        // the answer copies: every turn counts
        c.copy_run = super::COPY_RUN;
        c.turn_copying = true;
        c.step_ms = 25.0;
        for _ in 0..10 {
            c.note_step(25.0);
        }
        // a probe that does not pay counts its tokens and backs off further than before (2048)
        c.probing = true;
        c.probe_gap = 1024;
        c.note_pass(2, 0, 10.0, 58.0);
        assert_eq!((c.since_paid, c.probe_gap), (11, 2048));
        c.probing = true;
        c.note_pass(2, 0, 10.0, 58.0);
        assert_eq!(c.probe_gap, super::PROBE_MAX);
        let l = c.learned();
        let mut d = super::SpecControl::new(5);
        d.resume(&serde_json::from_slice(&serde_json::to_vec(&l).unwrap()).unwrap());
        assert_eq!((d.probe_gap, d.since_paid), (2048, 12));
        // a paying pass resets both
        d.step_ms = 25.0;
        d.note_pass(3, 3, 9.0, 50.0);
        assert_eq!((d.probe_gap, d.since_paid), (super::PROBE_MIN, 0));
        // sizes that do not fit are ignored
        let mut e = super::SpecControl::new(5);
        e.resume(&super::Learned { probe_gap: 9999, conf_paid: 2.0, hit: vec![0.5; 3], since_paid: 7, skip_until: 0, ms_on: None, ms_off: Some(-1.0), unprofitable: 0, calib: vec![] });
        assert_eq!(e.ms_off, None);
        assert_eq!((e.probe_gap, e.conf_paid, e.hit.len(), e.since_paid), (super::PROBE_MIN, 1.0, 5, 7));
    }

    #[test]
    fn off_probes_switch_drafting_back_on() {
        let mut c = super::SpecControl::new(5);
        c.copy_run = super::COPY_RUN;
        c.turn_copying = true;
        c.step_ms = 25.0;
        c.note_wall(super::WINDOW, 27.0 * super::WINDOW as f64);
        c.note_wall(super::WINDOW, 25.0 * super::WINDOW as f64);
        assert!(!c.drafting());
        // off: nothing until a probe is due, then 2 tokens
        let mut probe = None;
        for i in 0..super::OFF_PROBE + 1 {
            let k = c.choose(5);
            if k > 0 {
                probe = Some((i, k));
                break;
            }
        }
        let (i, k) = probe.expect("a probe while off");
        assert!(i < super::OFF_PROBE && k == 3);
        // fully accepted and paying: drafting on again
        c.note_pass(3, 3, 9.0, 50.0);
        assert!(c.drafting());
        // a probe that fails keeps it off
        let mut d = super::SpecControl::new(5);
        d.copy_run = super::COPY_RUN;
        d.turn_copying = true;
        d.step_ms = 25.0;
        d.note_wall(super::WINDOW, 27.0 * super::WINDOW as f64);
        d.note_wall(super::WINDOW, 25.0 * super::WINDOW as f64);
        while d.choose(5) == 0 {}
        d.note_pass(3, 0, 9.0, 50.0);
        assert!(!d.drafting());
    }

    #[test]
    fn confidence_cuts_drafts_it_learned_are_rejected() {
        let mut c = super::SpecControl::new(5);
        c.step_ms = 25.0;
        for (t, ms) in [(2, 37.5), (3, 55.0), (4, 69.0), (5, 80.0), (6, 92.0)] {
            c.verify_ms[t] = ms;
            c.measured[t] = true;
        }
        // confident everywhere (raw prior): verify all five
        assert_eq!(c.cut(&[5.0; 5], 5), 5);
        // the head says no from the third on
        assert_eq!(c.cut(&[5.0, 5.0, -5.0, -5.0, -5.0], 5), 2);
        // learned: logit 2..3 at position 0 is rejected here although the raw sigmoid says 0.9
        for _ in 0..400 {
            c.note_cut(&[2.5, 2.5], 2, 0);
        }
        assert!(c.accept_p(0, 2.5) < 0.05);
        assert_eq!(c.cut(&[2.5, 5.0, 5.0], 3), 0);
        // and it persists
        let mut d = super::SpecControl::new(5);
        d.resume(&c.learned());
        assert!(d.accept_p(0, 2.5) < 0.05);
    }

    #[test]
    fn copy_signal_from_prompt_and_output() {
        let mut c = super::SpecControl::new(5);
        c.begin(&[10, 11, 12, 13, 14, 15, 16]);
        assert!(!c.copying());
        // new text: no draft, no probe
        for t in [90, 91, 92, 93] {
            c.note_token(t);
        }
        assert!(!c.copying());
        assert_eq!(c.choose(5), 0);
        // the answer repeats 12 13 14 15: trigrams (12,13,14) and (13,14,15) hit -> copying
        for t in [12, 13, 14, 15] {
            c.note_token(t);
        }
        assert!(c.copying());
        assert!(c.choose(5) > 0 || c.probe_gap > 0);
        // a new token breaks the run
        c.note_token(77);
        assert!(!c.copying());
        // new text neither moves the windows nor counts as unpaid (R210: 30k prose tokens unloaded DSpark for 7 days)
        let mut p = super::SpecControl::new(5);
        p.begin(&[1, 2, 3]);
        for i in 0..40_000u32 {
            assert_eq!(p.choose(5), 0);
            p.note_step(20.0);
            p.note_wall(1, 20.0);
            p.note_token(1000 + i);
        }
        assert_eq!((p.since_paid, p.ms_on, p.ms_off), (0, None, None));
        // its own earlier output counts too
        let mut d = super::SpecControl::new(5);
        d.begin(&[1, 2]);
        for t in [50, 51, 52, 53, 9, 50, 51, 52, 53] {
            d.note_token(t);
        }
        assert!(d.copying());
    }

    #[test]
    fn end_to_end_windows_pick_the_faster_mode_and_recheck() {
        let mut c = super::SpecControl::new(5);
        // the answer copies: every turn counts
        c.copy_run = super::COPY_RUN;
        c.turn_copying = true;
        // first window drafting (unmeasured), second not
        assert!(c.drafting());
        c.note_wall(super::WINDOW, 25.0 * super::WINDOW as f64 * 1.1);
        assert!(!c.drafting());
        c.note_wall(super::WINDOW, 25.0 * super::WINDOW as f64);
        // drafting measured 10 % slower: off, its tokens counted as unprofitable, choose() drafts nothing
        assert!(!c.drafting());
        assert_eq!(c.choose(5), 0);
        assert_eq!(c.unprofitable, super::WINDOW as u64);
        let mut rechecked = false;
        for _ in 0..super::RECHECK_EVERY {
            let on = c.drafting();
            rechecked |= on;
            c.note_wall(super::WINDOW, if on { 27.5 } else { 25.0 } * super::WINDOW as f64);
        }
        assert!(rechecked, "the slower mode runs again within RECHECK_EVERY windows");
        // drafting becomes faster (code): it wins and the unprofitable count resets
        let mut on_windows = 0;
        for _ in 0..32 {
            let on = c.drafting();
            on_windows += on as usize;
            c.note_wall(super::WINDOW, if on { 18.0 } else { 25.0 } * super::WINDOW as f64);
        }
        assert!(c.ms_on.unwrap() < c.ms_off.unwrap());
        assert!(on_windows >= 20, "{on_windows}");
        assert_eq!(c.unprofitable, 0);
    }
}
