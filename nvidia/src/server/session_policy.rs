//! Pure cache policy: LRU eviction order and admission arithmetic for the multi-session cache.
//!
//! Nothing here touches the GPU, the host arena or the model, so the whole policy is unit-testable with the default
//! feature set. `real.rs` evaluates the same functions against the live session metadata; the GPU copies stay in
//! `gpumodel` and the arena accounting stays in `hostcache`.
//!
//! v1 eviction unit is a whole inactive session. The comparison lives here and only here (`eviction_score`), so the
//! policy can later weigh recompute cost, session size, hit count or priority without rewriting the cache manager.

/// One session as the policy sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionMeta {
    pub id: u64,
    /// the session that currently owns the GPU sequence; it is never an eviction victim
    pub active: bool,
    pub last_used: u64,
    /// arena extents held by the session
    pub payload_bytes: usize,
    /// heap metadata/logits/Engram of the session (also part of the configured budget)
    pub heap_bytes: usize,
}

/// Why a session was evicted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionReason {
    SessionCount,
    MemoryBudget,
}

impl EvictionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SessionCount => "session-count",
            Self::MemoryBudget => "memory-budget",
        }
    }
}

/// Why a session could not be admitted into the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionError {
    /// The session's own backing state is larger than the whole cache budget: it is not retainable at all.
    StateTooLarge { required: usize, budget: usize },
    /// The count cap cannot be met even after evicting every evictable session.
    SessionCountFull { retained: usize, max: usize },
    /// The sessions that must stay exceed the byte budget even after evicting every evictable session.
    BudgetExhausted { payload: usize, budget: usize },
}

impl AdmissionError {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StateTooLarge { .. } => "state-too-large",
            Self::SessionCountFull { .. } => "session-count-full",
            Self::BudgetExhausted { .. } => "cache-budget",
        }
    }
}

/// v1 score: the LRU clock of the last request that used the session. Lower is evicted first.
/// Future versions can add recompute cost, size, hit count or priority here only.
pub fn eviction_score(session: &SessionMeta) -> u64 {
    session.last_used
}

/// Inactive sessions in eviction order (score, then stable id), excluding `keep`. The active session is protected.
pub fn eviction_order(sessions: &[SessionMeta], keep: &[u64]) -> Vec<SessionMeta> {
    let mut victims: Vec<SessionMeta> = sessions
        .iter()
        .copied()
        .filter(|s| !s.active && !keep.contains(&s.id))
        .collect();
    victims.sort_by_key(|s| (eviction_score(s), s.id));
    victims
}

/// One planned eviction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eviction {
    pub id: u64,
    pub payload_bytes: usize,
    pub reason: EvictionReason,
}

/// The admission decision taken **before** any park/allocation happens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionPlan {
    pub victims: Vec<Eviction>,
    /// retained sessions after the plan (including the session being added)
    pub retained: usize,
    /// arena payload after the plan
    pub payload: usize,
}

/// Decide which inactive sessions must go so that, at the same time:
/// retained sessions <= `max_sessions`, arena payload <= `budget_bytes`, and the new session's payload fits.
///
/// `required_bytes` is the payload the caller is about to add (the parked backing state of the active session
/// and/or a new session entry). `add_sessions` is 1 when a new lineage entry will be created, 0 when an existing one
/// is reused. Count pressure is resolved first (`reason=session-count`), then byte pressure (`reason=memory-budget`).
pub fn plan_admission(
    sessions: &[SessionMeta],
    keep: &[u64],
    required_bytes: usize,
    add_sessions: usize,
    budget_bytes: usize,
    max_sessions: usize,
) -> Result<AdmissionPlan, AdmissionError> {
    if required_bytes > budget_bytes {
        return Err(AdmissionError::StateTooLarge {
            required: required_bytes,
            budget: budget_bytes,
        });
    }
    let mut retained = sessions.len() + add_sessions;
    let mut payload = sessions
        .iter()
        .map(|s| s.payload_bytes + s.heap_bytes)
        .sum::<usize>()
        + required_bytes;
    let candidates = eviction_order(sessions, keep);
    let mut victims = Vec::new();
    let mut next = 0usize;
    loop {
        if retained <= max_sessions && payload <= budget_bytes {
            break;
        }
        let Some(v) = candidates.get(next).copied() else {
            return Err(if retained > max_sessions {
                AdmissionError::SessionCountFull {
                    retained,
                    max: max_sessions,
                }
            } else {
                AdmissionError::BudgetExhausted {
                    payload,
                    budget: budget_bytes,
                }
            });
        };
        next += 1;
        let reason = if retained > max_sessions {
            EvictionReason::SessionCount
        } else {
            EvictionReason::MemoryBudget
        };
        victims.push(Eviction {
            id: v.id,
            payload_bytes: v.payload_bytes,
            reason,
        });
        retained -= 1;
        payload = payload.saturating_sub(v.payload_bytes + v.heap_bytes);
    }
    Ok(AdmissionPlan {
        victims,
        retained,
        payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(id: u64, active: bool, last_used: u64, payload_bytes: usize) -> SessionMeta {
        SessionMeta {
            id,
            active,
            last_used,
            payload_bytes,
            heap_bytes: 0,
        }
    }

    #[test]
    fn lru_order_skips_active_and_kept() {
        let s = [
            meta(1, false, 9, 10),
            meta(2, true, 1, 10),
            meta(3, false, 5, 10),
            meta(4, false, 5, 10),
        ];
        let order: Vec<u64> = eviction_order(&s, &[3]).iter().map(|v| v.id).collect();
        assert_eq!(
            order,
            vec![4, 1],
            "same score breaks on id; kept and active sessions excluded"
        );
    }

    #[test]
    fn no_eviction_when_everything_fits() {
        let s = [meta(1, true, 2, 100)];
        let p = plan_admission(&s, &[], 100, 1, 1000, 2).unwrap();
        assert!(p.victims.is_empty());
        assert_eq!(p.retained, 2);
        assert_eq!(p.payload, 200);
    }

    #[test]
    fn count_pressure_evicts_lru_first() {
        let s = [
            meta(1, true, 3, 100),
            meta(2, false, 1, 100),
            meta(3, false, 2, 100),
        ];
        let p = plan_admission(&s, &[], 100, 1, 10_000, 3).unwrap();
        assert_eq!(p.victims.len(), 1);
        assert_eq!(p.victims[0].id, 2, "least recently used inactive session");
        assert_eq!(p.victims[0].reason, EvictionReason::SessionCount);
        assert_eq!(p.retained, 3);
    }

    #[test]
    fn memory_pressure_evicts_until_payload_fits() {
        let s = [
            meta(1, true, 3, 400),
            meta(2, false, 1, 400),
            meta(3, false, 2, 400),
        ];
        let p = plan_admission(&s, &[], 400, 0, 1000, 8).unwrap();
        assert_eq!(p.victims.len(), 2);
        assert_eq!(p.victims[0].id, 2);
        assert_eq!(p.victims[1].id, 3);
        assert_eq!(p.victims[0].reason, EvictionReason::MemoryBudget);
        assert!(p.payload <= 1000);
    }

    #[test]
    fn protected_session_survives_pressure() {
        let s = [
            meta(1, true, 3, 400),
            meta(2, false, 1, 400),
            meta(3, false, 2, 400),
        ];
        let p = plan_admission(&s, &[2], 400, 0, 1250, 8).unwrap();
        assert_eq!(p.victims.len(), 1);
        assert_eq!(p.victims[0].id, 3, "the prefix source is kept");
    }

    #[test]
    fn heap_bytes_count_against_the_budget() {
        let mut s = [meta(1, true, 1, 100)];
        s[0].heap_bytes = 5000;
        let e = plan_admission(&s, &[], 100, 0, 2000, 8).unwrap_err();
        assert!(matches!(e, AdmissionError::BudgetExhausted { .. }));
        let p = plan_admission(&s, &[], 100, 0, 6000, 8).unwrap();
        assert!(p.payload <= 6000);
    }

    #[test]
    fn oversized_session_is_not_retainable() {
        let e = plan_admission(&[], &[], 5_000, 1, 1000, 4).unwrap_err();
        assert_eq!(
            e,
            AdmissionError::StateTooLarge {
                required: 5_000,
                budget: 1000
            }
        );
        assert_eq!(e.as_str(), "state-too-large");
    }

    #[test]
    fn count_cap_with_no_victims_is_reported() {
        let s = [meta(1, true, 1, 10)];
        let e = plan_admission(&s, &[], 10, 1, 1 << 20, 1).unwrap_err();
        assert!(matches!(e, AdmissionError::SessionCountFull { .. }));
    }

    #[test]
    fn budget_with_only_protected_sessions_is_reported() {
        let s = [meta(1, true, 1, 900)];
        let e = plan_admission(&s, &[], 200, 0, 1000, 8).unwrap_err();
        assert!(matches!(e, AdmissionError::BudgetExhausted { .. }));
    }

    #[test]
    fn both_limits_are_checked_together() {
        let s = [
            meta(1, true, 3, 300),
            meta(2, false, 1, 300),
            meta(3, false, 2, 300),
            meta(4, false, 4, 300),
        ];
        // count cap 3 and payload cap 700: adding one session and 200 bytes needs two evictions (LRU 2, then 3)
        let p = plan_admission(&s, &[], 200, 1, 700, 3).unwrap();
        let ids: Vec<u64> = p.victims.iter().map(|v| v.id).collect();
        assert_eq!(ids, vec![2, 3, 4]);
        assert_eq!(p.victims[0].reason, EvictionReason::SessionCount);
        assert_eq!(p.victims[1].reason, EvictionReason::SessionCount);
        assert_eq!(p.victims[2].reason, EvictionReason::MemoryBudget);
        assert!(p.retained <= 3);
        assert!(p.payload <= 700);
    }
}
