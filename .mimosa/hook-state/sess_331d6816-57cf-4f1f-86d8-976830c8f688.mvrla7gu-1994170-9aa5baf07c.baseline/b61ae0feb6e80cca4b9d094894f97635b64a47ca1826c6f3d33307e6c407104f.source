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

#[cfg(test)]
mod tests {
    #[test]
    fn finds_the_newest_continuation() {
        let c = [1, 2, 3, 9, 9, 1, 2, 3, 7, 8, 5, 1, 2, 3];
        assert_eq!(super::lookup(&c, 2), vec![7, 8]);
        assert_eq!(super::lookup(&[4, 5, 6], 3), Vec::<u32>::new());
    }
}
