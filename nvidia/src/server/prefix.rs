//! Prefix-cache policy for the GPU engine.
//!
//! The engine keeps one cached token lineage and a bounded history of prompt-boundary checkpoints. For a new prompt
//! the decision is pure: compare the true longest common prefix with the cached lineage, then continue from the live
//! model state when possible or restore the newest checkpoint at or before that prefix. Checkpoints newer than the
//! common prefix belong to a divergent branch and are dropped.
//!
//! Kept separate from `real.rs` so the policy (including exact-retry and branch-invalidation cases) is testable
//! without loading a GPU model.

/// A checkpoint's position in its token lineage.
pub trait CheckpointPosition {
    fn pos(&self) -> usize;
}

impl CheckpointPosition for usize {
    fn pos(&self) -> usize {
        *self
    }
}

/// What the cache did for this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAction {
    /// The new prompt extends the complete cached sequence (or equals it): keep the live model state.
    ExactCurrent,
    /// The prompt equals a cached prompt boundary exactly: restore that checkpoint and reuse its saved logits.
    ExactRetry,
    /// Restore the newest checkpoint and prefill the remaining prompt tail.
    PromptCheckpoint,
    /// The newest checkpoint is past the common prefix: restore an older valid checkpoint.
    OlderCheckpoint,
    /// No checkpoint exists at or before the common prefix: reset and prefill from token 0.
    ColdReset,
}

impl CacheAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactCurrent => "exact-current",
            Self::ExactRetry => "exact-retry",
            Self::PromptCheckpoint => "prompt-checkpoint",
            Self::OlderCheckpoint => "older-checkpoint",
            Self::ColdReset => "cold-reset",
        }
    }
}

/// The selected replay path. The caller restores `checkpoint_index` (when set), drops invalidated checkpoints and
/// prefills `prompt_len - start` tokens. A zero-length replay uses saved logits instead of running `prefill`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    pub action: CacheAction,
    pub start: usize,
    pub replayed_tokens: usize,
    pub checkpoint_index: Option<usize>,
    /// Number of leading checkpoints that remain valid for the new lineage.
    pub retain_checkpoints: usize,
    pub checkpoint_count: usize,
    /// The new prompt matches a cached prompt in full (`raw_lcp == prompt_len`).
    pub exact_retry: bool,
}

/// Count the identical leading tokens of two sequences. This is the real LCP: it is never shortened to force a
/// replay token, because that would turn an exact retry into a false cache miss.
pub fn longest_common_prefix(new: &[u32], cached: &[u32]) -> usize {
    new.iter().zip(cached).take_while(|(a, b)| a == b).count()
}

/// Choose where a new prompt resumes.
///
/// `checkpoints` must be sorted by position. The newest checkpoint `<= raw_lcp` is selected. Future checkpoints are
/// invalid for the new lineage and are counted out by `retain_checkpoints`.
pub fn choose_prefix_reuse<T: CheckpointPosition>(
    prompt_len: usize,
    cached_len: usize,
    raw_lcp: usize,
    current_pos: usize,
    checkpoints: &[T],
) -> Decision {
    debug_assert!(
        raw_lcp <= prompt_len.min(cached_len),
        "raw_lcp {raw_lcp} exceeds prompt {prompt_len} / cached {cached_len}"
    );

    let mut selected = None;
    let mut retain_checkpoints = 0;
    for (i, c) in checkpoints.iter().enumerate() {
        if c.pos() <= raw_lcp {
            selected = Some((i, c.pos()));
            retain_checkpoints = i + 1;
        }
    }
    let checkpoint_count = checkpoints.len();
    let exact_retry = raw_lcp == prompt_len;

    // A live state exactly at the cached end can continue without a restore. This includes a prompt identical to the
    // cached sequence; its logits are read from the model's logits buffer. An empty cache is a cold start for logging,
    // even though its model position is also 0.
    if cached_len > 0 && raw_lcp == cached_len && current_pos == cached_len {
        return Decision {
            action: CacheAction::ExactCurrent,
            start: cached_len,
            replayed_tokens: prompt_len - cached_len,
            checkpoint_index: None,
            retain_checkpoints,
            checkpoint_count,
            exact_retry,
        };
    }

    if let Some((index, pos)) = selected {
        let replayed_tokens = prompt_len - pos;
        let action = if replayed_tokens == 0 {
            CacheAction::ExactRetry
        } else if index + 1 == checkpoint_count {
            CacheAction::PromptCheckpoint
        } else {
            CacheAction::OlderCheckpoint
        };
        Decision {
            action,
            start: pos,
            replayed_tokens,
            checkpoint_index: Some(index),
            retain_checkpoints,
            checkpoint_count,
            exact_retry,
        }
    } else {
        Decision {
            action: CacheAction::ColdReset,
            start: 0,
            replayed_tokens: prompt_len,
            checkpoint_index: None,
            retain_checkpoints: 0,
            checkpoint_count,
            exact_retry,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decision(
        prompt_len: usize,
        cached_len: usize,
        raw_lcp: usize,
        current_pos: usize,
        checkpoints: &[usize],
    ) -> Decision {
        choose_prefix_reuse(prompt_len, cached_len, raw_lcp, current_pos, checkpoints)
    }

    #[test]
    fn lcp_is_not_reduced_for_logits() {
        assert_eq!(longest_common_prefix(&[1, 2, 3, 4], &[1, 2, 3, 4]), 4);
        assert_eq!(longest_common_prefix(&[1, 2, 3, 4, 5, 6], &[1, 2, 3, 4]), 4);
        assert_eq!(longest_common_prefix(&[7, 2, 3], &[1, 2, 3]), 0);
    }

    /// Test 1: append-only continuation continues at the cached end.
    #[test]
    fn append_only_continuation() {
        let d = decision(6, 4, 4, 4, &[2, 4]);
        assert_eq!(d.action, CacheAction::ExactCurrent);
        assert_eq!(d.start, 4);
        assert_eq!(d.replayed_tokens, 2);
        assert_eq!(d.checkpoint_index, None);
        assert_eq!(d.retain_checkpoints, 2);
    }

    /// Test 2: divergence just after the latest prompt boundary restores that boundary.
    #[test]
    fn divergence_after_latest_boundary() {
        let d = decision(180, 160, 110, 160, &[100]);
        assert_eq!(d.action, CacheAction::PromptCheckpoint);
        assert_eq!(d.start, 100);
        assert_eq!(d.replayed_tokens, 80);
        assert_eq!(d.checkpoint_index, Some(0));
        assert_eq!(d.retain_checkpoints, 1);
    }

    /// Test 3: divergence before the latest boundary selects the newest older checkpoint.
    #[test]
    fn divergence_before_latest_boundary() {
        let d = decision(300, 400, 245, 400, &[100, 200, 300]);
        assert_eq!(d.action, CacheAction::OlderCheckpoint);
        assert_eq!(d.start, 200);
        assert_eq!(d.replayed_tokens, 100);
        assert_eq!(d.checkpoint_index, Some(1));
        assert_eq!(d.retain_checkpoints, 2);
    }

    /// Test 4: divergence exactly on a checkpoint selects that checkpoint.
    #[test]
    fn divergence_on_checkpoint() {
        let d = decision(300, 400, 200, 400, &[100, 200, 300]);
        assert_eq!(d.start, 200);
        assert_eq!(d.checkpoint_index, Some(1));
        assert_eq!(d.retain_checkpoints, 2);
    }

    /// Test 5: no checkpoint before the prefix means a cold reset.
    #[test]
    fn no_usable_checkpoint() {
        let d = decision(60, 80, 50, 80, &[100, 200]);
        assert_eq!(d.action, CacheAction::ColdReset);
        assert_eq!(d.start, 0);
        assert_eq!(d.replayed_tokens, 60);
        assert_eq!(d.checkpoint_index, None);
        assert_eq!(d.retain_checkpoints, 0);
    }

    /// Test 6: an identical long prompt retry uses the checkpoint and replays no prompt tokens.
    #[test]
    fn exact_identical_prompt_retry() {
        let d = decision(250_000, 250_500, 250_000, 250_500, &[100, 200, 250_000]);
        assert_eq!(d.action, CacheAction::ExactRetry);
        assert_ne!(d.action, CacheAction::ColdReset);
        assert_eq!(d.start, 250_000);
        assert_eq!(d.replayed_tokens, 0);
        assert_eq!(d.checkpoint_index, Some(2));
        assert!(d.exact_retry);
    }

    /// Test 7: choosing an older checkpoint invalidates the divergent future branch.
    #[test]
    fn branch_invalidation() {
        let d = decision(350, 450, 250, 450, &[100, 200, 300, 400]);
        assert_eq!(d.start, 200);
        assert_eq!(d.retain_checkpoints, 2);
        let mut checkpoints = vec![100, 200, 300, 400];
        checkpoints.truncate(d.retain_checkpoints);
        checkpoints.push(350); // the new prompt boundary
        assert_eq!(checkpoints, vec![100, 200, 350]);
    }

    /// Test 8: a shorter prompt picks the newest checkpoint at or before its end and drops the rest.
    #[test]
    fn shorter_prompt() {
        let d = decision(350, 500, 350, 500, &[100, 200, 300, 400]);
        assert_eq!(d.action, CacheAction::OlderCheckpoint);
        assert_eq!(d.start, 300);
        assert_eq!(d.replayed_tokens, 50);
        assert_eq!(d.retain_checkpoints, 3);
    }

    /// Test 9: a mismatch at token 0 has no usable checkpoint.
    #[test]
    fn mismatch_at_token_zero() {
        let d = decision(100, 120, 0, 120, &[10, 50]);
        assert_eq!(d.action, CacheAction::ColdReset);
        assert_eq!(d.start, 0);
        assert_eq!(d.replayed_tokens, 100);
    }

    /// Test 10: the generated tail differs but the prompt boundary matches; restore the boundary, not token 0.
    #[test]
    fn generated_tail_divergence() {
        let d = decision(150, 200, 100, 200, &[100]);
        assert_eq!(d.action, CacheAction::PromptCheckpoint);
        assert_eq!(d.start, 100);
        assert_eq!(d.replayed_tokens, 50);
        assert_eq!(d.checkpoint_index, Some(0));
    }

    /// A prompt equal to the cached sequence continues from the live state instead of replaying it.
    #[test]
    fn exact_current_prompt() {
        let d = decision(500, 500, 500, 500, &[100, 200, 500]);
        assert_eq!(d.action, CacheAction::ExactCurrent);
        assert_eq!(d.start, 500);
        assert_eq!(d.replayed_tokens, 0);
        assert_eq!(d.checkpoint_index, None);
        assert_eq!(d.retain_checkpoints, 3);
        assert!(d.exact_retry);
    }

    #[test]
    fn empty_cache_is_a_cold_start() {
        let d = decision(10, 0, 0, 0, &[]);
        assert_eq!(d.action, CacheAction::ColdReset);
        assert_eq!(d.start, 0);
        assert_eq!(d.replayed_tokens, 10);
    }
}
