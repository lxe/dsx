//! Bounded host-side cache arena for inactive conversation lineages.
//!
//! Parked session state, prompt checkpoints and prefix anchors all live in host RAM while their lineage is not the
//! active one. A single arena owns every extent so switching sessions never registers/unregisters one CUDA host
//! buffer per session or per checkpoint (that churn is exactly what a per-session `HostBuf` would do).
//!
//! Slabs are allocated lazily: nothing is committed at startup, the first allocation creates one slab, later
//! allocations reuse free extents inside committed slabs, and only when nothing fits is another slab committed. When
//! a slab holds no live extent (for example after a whole-session eviction) it is dropped and the RAM returns to the
//! system, so `committed_bytes()` can fall again.
//!
//! Two numbers describe the cache, and both are part of the hard budget:
//!
//! * `payload_bytes()` — bytes handed out to live extents (parked states, checkpoints, anchors, saved logits are
//!   heap `Vec`s and are not arena extents);
//! * `committed_bytes()` — bytes of allocated slabs, including free space inside them.
//!
//! The budget bounds committed bytes: free extents are still committed memory, so counting only the payload would
//! understate RAM pressure. `allocate()` refuses any allocation that would push committed bytes past the budget.
//!
//! The arena is deliberately not generic over the GPU: copies are issued by `GpuModel` (one grouped DtoH / HtoD
//! batch plus a single synchronization at the park/restore boundary), and the arena only owns storage. A heap-backed
//! constructor exists so the allocator and the accounting are testable without a CUDA device.

use std::fmt;

/// Extent offsets are aligned to this many bytes: small enough to waste little, large enough for DMA-friendly
/// copies (CUDA copies do not require it, but 256-byte aligned rows keep the park/restore memcpy bandwidth stable).
pub const EXTENT_ALIGN: usize = 256;

/// Slabs are rounded up like the registered host buffers they wrap, so `committed_bytes()` matches the memory the
/// process actually holds.
pub const SLAB_ALIGN: usize = 2 << 20;

/// Storage behind one arena slab.
pub trait SlabStorage: Send {
    fn bytes(&self) -> usize;
    fn ptr(&self) -> *mut u8;
}

/// Pinned, CUDA-registered host memory: the production backing of every slab.
#[cfg(feature = "cuda")]
impl SlabStorage for crate::gpu::HostBuf {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn ptr(&self) -> *mut u8 {
        self.host_ptr()
    }
}

/// Plain heap memory, used by the arena's own tests (never by the server).
impl SlabStorage for Vec<u8> {
    fn bytes(&self) -> usize {
        self.len()
    }
    fn ptr(&self) -> *mut u8 {
        self.as_ptr().cast_mut()
    }
}

/// A byte range inside one slab. Extents are small values (a slab index, an offset and a length), so they can be
/// copied into session metadata freely; the arena tracks the ownership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    slab: u32,
    off: usize,
    len: usize,
}

impl Extent {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Why an allocation could not be satisfied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArenaError {
    /// One extent is larger than the whole arena budget.
    TooLarge { requested: usize, budget: usize },
    /// The arena would have to commit more memory than its budget allows.
    OutOfBudget {
        requested: usize,
        committed: usize,
        budget: usize,
    },
    /// The host allocation itself failed (CUDA registration, malloc, ...).
    HostAllocation(String),
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { requested, budget } => write!(
                f,
                "extent of {requested} bytes exceeds the {budget}-byte arena budget"
            ),
            Self::OutOfBudget {
                requested,
                committed,
                budget,
            } => {
                write!(
                    f,
                    "committing {requested} bytes on top of {committed} committed bytes exceeds the {budget}-byte arena budget"
                )
            }
            Self::HostAllocation(e) => write!(f, "host cache allocation failed: {e}"),
        }
    }
}

impl std::error::Error for ArenaError {}

/// What a prospective allocation would cost, without touching memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocPlan {
    /// A free extent in a committed slab is large enough.
    Reuse,
    /// A new slab of this many bytes must be committed.
    Grow(usize),
}

struct Slab {
    storage: Box<dyn SlabStorage>,
    /// allocations inside the slab, sorted by offset
    allocs: Vec<(usize, usize)>,
    /// live extents (the slab is dropped when this reaches zero)
    live: usize,
}

impl Slab {
    fn bytes(&self) -> usize {
        self.storage.bytes()
    }

    /// gaps between allocations, in offset order
    fn free_spans(&self) -> Vec<(usize, usize)> {
        let mut spans = vec![];
        let mut off = 0;
        for &(a, len) in &self.allocs {
            if a > off {
                spans.push((off, a - off));
            }
            off = a + len;
        }
        let end = self.bytes();
        if off < end {
            spans.push((off, end - off));
        }
        spans
    }
}

/// The bounded host cache: lazily committed slabs plus an extent allocator and its accounting.
pub struct HostCacheArena {
    budget_bytes: usize,
    slab_bytes: usize,
    /// slab slots stay in place when freed (`None`), so extent slab indices remain stable
    slabs: Vec<Option<Slab>>,
    payload_bytes: usize,
    peak_payload_bytes: usize,
    peak_committed_bytes: usize,
    make: Box<dyn Fn(usize) -> anyhow::Result<Box<dyn SlabStorage>>>,
}

fn align_up(n: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    n.div_ceil(align) * align
}

impl HostCacheArena {
    /// An arena whose slabs are plain heap memory (tests and tools; the server uses `pinned`).
    pub fn heap(budget_bytes: usize, slab_bytes: usize) -> Self {
        Self::with_storage(
            budget_bytes,
            slab_bytes,
            Box::new(|n| Ok(Box::new(vec![0u8; n]) as Box<dyn SlabStorage>)),
        )
    }

    /// An arena over CUDA-registered, page-locked host memory: slabs are DMA-able, so park/restore copies run
    /// asynchronously on the engine's stream.
    #[cfg(feature = "cuda")]
    pub fn pinned(budget_bytes: usize, slab_bytes: usize) -> Self {
        Self::with_storage(
            budget_bytes,
            slab_bytes,
            Box::new(|n| Ok(Box::new(crate::gpu::HostBuf::new(n)?) as Box<dyn SlabStorage>)),
        )
    }

    fn with_storage(
        budget_bytes: usize,
        slab_bytes: usize,
        make: Box<dyn Fn(usize) -> anyhow::Result<Box<dyn SlabStorage>>>,
    ) -> Self {
        // a slab can never be larger than the whole budget, or a cache smaller than one slab could never allocate
        let cap = budget_bytes.max(EXTENT_ALIGN);
        let slab_bytes = align_up(slab_bytes.max(EXTENT_ALIGN).min(cap), SLAB_ALIGN);
        Self {
            budget_bytes,
            slab_bytes,
            slabs: Vec::new(),
            payload_bytes: 0,
            peak_payload_bytes: 0,
            peak_committed_bytes: 0,
            make,
        }
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    pub fn committed_bytes(&self) -> usize {
        self.slabs.iter().flatten().map(|s| s.bytes()).sum()
    }

    pub fn peak_payload_bytes(&self) -> usize {
        self.peak_payload_bytes
    }

    pub fn peak_committed_bytes(&self) -> usize {
        self.peak_committed_bytes
    }

    /// committed slabs (a fully free slab is dropped and does not count).
    pub fn slab_count(&self) -> usize {
        self.slabs.iter().filter(|s| s.is_some()).count()
    }

    /// Payload that can still be handed out under the budget (payload is always <= committed).
    pub fn available_payload_bytes(&self) -> usize {
        self.budget_bytes.saturating_sub(self.payload_bytes)
    }

    /// Whether `bytes` fits without committing anything new, or how large the new slab would be.
    pub fn plan(&self, bytes: usize) -> Result<AllocPlan, ArenaError> {
        let want = align_up(bytes.max(1), EXTENT_ALIGN);
        if want > self.budget_bytes {
            return Err(ArenaError::TooLarge {
                requested: want,
                budget: self.budget_bytes,
            });
        }
        if self.best_free_span(want).is_some() {
            return Ok(AllocPlan::Reuse);
        }
        let size = align_up(self.slab_bytes.max(want), SLAB_ALIGN);
        let committed = self.committed_bytes();
        if committed + size > self.budget_bytes {
            return Err(ArenaError::OutOfBudget {
                requested: size,
                committed,
                budget: self.budget_bytes,
            });
        }
        Ok(AllocPlan::Grow(size))
    }

    fn best_free_span(&self, want: usize) -> Option<(usize, usize, usize)> {
        let mut best: Option<(usize, usize, usize)> = None;
        for (i, slab) in self.slabs.iter().enumerate() {
            let Some(s) = slab else { continue };
            for (off, len) in s.free_spans() {
                if len >= want && best.map_or(true, |(_, _, bl)| len < bl) {
                    best = Some((i, off, len));
                }
            }
        }
        best
    }

    /// Reserve an extent. The first fitting free span of the arena is reused (best fit); otherwise one slab is
    /// committed lazily. Never exceeds the hard budget.
    pub fn allocate(&mut self, bytes: usize) -> Result<Extent, ArenaError> {
        let want = align_up(bytes.max(1), EXTENT_ALIGN);
        if want > self.budget_bytes {
            return Err(ArenaError::TooLarge {
                requested: want,
                budget: self.budget_bytes,
            });
        }
        if let Some((i, off, _)) = self.best_free_span(want) {
            let s = self.slabs[i].as_mut().expect("slab slot");
            s.allocs.push((off, want));
            s.allocs.sort_unstable();
            s.live += 1;
            self.payload_bytes += want;
            self.peak_payload_bytes = self.peak_payload_bytes.max(self.payload_bytes);
            return Ok(Extent {
                slab: i as u32,
                off,
                len: want,
            });
        }
        let size = align_up(self.slab_bytes.max(want), SLAB_ALIGN);
        let committed = self.committed_bytes();
        if committed + size > self.budget_bytes {
            return Err(ArenaError::OutOfBudget {
                requested: size,
                committed,
                budget: self.budget_bytes,
            });
        }
        let storage =
            (self.make)(size).map_err(|e| ArenaError::HostAllocation(format!("{e:#}")))?;
        let size = storage.bytes();
        if committed + size > self.budget_bytes {
            return Err(ArenaError::OutOfBudget {
                requested: size,
                committed,
                budget: self.budget_bytes,
            });
        }
        let idx = self.slabs.len();
        self.slabs.push(Some(Slab {
            storage,
            allocs: vec![(0, want)],
            live: 1,
        }));
        self.payload_bytes += want;
        self.peak_payload_bytes = self.peak_payload_bytes.max(self.payload_bytes);
        self.peak_committed_bytes = self.peak_committed_bytes.max(self.committed_bytes());
        Ok(Extent {
            slab: idx as u32,
            off: 0,
            len: want,
        })
    }

    /// Return an extent. The slab is dropped (its RAM returned) as soon as its last extent is released, so
    /// `committed_bytes()` decreases after a whole-session eviction.
    pub fn free(&mut self, e: Extent) {
        if e.is_empty() {
            return;
        }
        let Some(slot) = self.slabs.get_mut(e.slab as usize) else {
            return;
        };
        let Some(s) = slot else { return };
        let Some(pos) = s
            .allocs
            .iter()
            .position(|&(off, len)| off == e.off && len == e.len)
        else {
            return;
        };
        s.allocs.remove(pos);
        s.live = s.live.saturating_sub(1);
        self.payload_bytes = self.payload_bytes.saturating_sub(e.len);
        if s.live == 0 {
            *slot = None;
        }
    }

    /// Drop every slab that holds no live extent (frees its RAM; committed bytes fall).
    pub fn release_empty_slabs(&mut self) -> usize {
        let mut freed = 0;
        for slot in self.slabs.iter_mut() {
            if slot.as_ref().is_some_and(|s| s.live == 0) {
                freed += slot.as_ref().map_or(0, |s| s.bytes());
                *slot = None;
            }
        }
        freed
    }

    /// Host address of an extent (the caller knows the extents are still owned).
    pub fn extent_ptr(&self, e: Extent) -> *mut u8 {
        let s = self.slabs[e.slab as usize]
            .as_ref()
            .expect("extent slab freed");
        unsafe { s.storage.ptr().add(e.off) }
    }

    /// Host bytes of an extent.
    pub fn extent_slice(&self, e: Extent) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.extent_ptr(e) as *const u8, e.len) }
    }

    pub fn extent_slice_mut(&mut self, e: Extent) -> &mut [u8] {
        let ptr = self.extent_ptr(e);
        unsafe { std::slice::from_raw_parts_mut(ptr, e.len) }
    }

    /// Release every extent. Slabs are dropped, so committed bytes return to zero (server shutdown).
    pub fn clear(&mut self) {
        self.slabs.clear();
        self.payload_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_slabs_and_reuse() {
        let mut a = HostCacheArena::heap(1 << 30, 64 << 20);
        assert_eq!(
            a.committed_bytes(),
            0,
            "nothing is committed before the first allocation"
        );
        let e1 = a.allocate(1 << 20).unwrap();
        assert_eq!(a.committed_bytes(), 64 << 20);
        assert_eq!(a.payload_bytes(), align_up(1 << 20, EXTENT_ALIGN));
        assert_eq!(a.slab_count(), 1);
        // a second extent fits in the free space of the committed slab
        let e2 = a.allocate(2 << 20).unwrap();
        assert_eq!(a.slab_count(), 1);
        assert_eq!(
            a.payload_bytes(),
            align_up(1 << 20, EXTENT_ALIGN) + align_up(2 << 20, EXTENT_ALIGN)
        );
        assert_ne!(e1, e2);
        a.free(e1);
        a.free(e2);
        assert_eq!(a.payload_bytes(), 0);
        assert_eq!(a.committed_bytes(), 0, "a fully free slab is dropped");
    }

    #[test]
    fn budget_is_never_exceeded() {
        let mut a = HostCacheArena::heap(64 << 20, 64 << 20);
        let e = a.allocate(63 << 20).unwrap();
        assert!(matches!(
            a.allocate(2 << 20),
            Err(ArenaError::OutOfBudget { .. })
        ));
        a.free(e);
    }

    #[test]
    fn a_budget_smaller_than_the_default_slab_still_allocates() {
        let mut a = HostCacheArena::heap(4 << 20, 256 << 20);
        assert_eq!(a.committed_bytes(), 0);
        let e = a.allocate(1 << 20).unwrap();
        assert!(
            a.committed_bytes() <= 4 << 20,
            "the slab is clamped to the budget"
        );
        assert!(matches!(
            a.allocate(4 << 20),
            Err(ArenaError::OutOfBudget { .. })
        ));
        a.free(e);
    }

    #[test]
    fn oversized_extent_is_too_large() {
        let mut a = HostCacheArena::heap(64 << 20, 64 << 20);
        assert!(matches!(
            a.allocate(65 << 20),
            Err(ArenaError::TooLarge { .. })
        ));
    }

    #[test]
    fn plan_reports_reuse_then_growth() {
        let mut a = HostCacheArena::heap(512 << 20, 64 << 20);
        assert_eq!(a.plan(1 << 20).unwrap(), AllocPlan::Grow(64 << 20));
        let e = a.allocate(1 << 20).unwrap();
        assert_eq!(a.plan(1 << 20).unwrap(), AllocPlan::Reuse);
        assert_eq!(a.plan(63 << 20).unwrap(), AllocPlan::Reuse);
        assert_eq!(a.plan(65 << 20).unwrap(), AllocPlan::Grow(66 << 20));
        a.free(e);
    }

    #[test]
    fn free_spans_coalesce_over_time() {
        let mut a = HostCacheArena::heap(1 << 30, 64 << 20);
        let e1 = a.allocate(8 << 20).unwrap();
        let e2 = a.allocate(8 << 20).unwrap();
        let e3 = a.allocate(8 << 20).unwrap();
        // middle out: the two neighbours coalesce into one span big enough for a 32 MB extent
        a.free(e2);
        a.free(e3);
        assert_eq!(a.plan(32 << 20).unwrap(), AllocPlan::Reuse);
        a.free(e1);
    }

    #[test]
    fn extents_do_not_overlap() {
        let mut a = HostCacheArena::heap(1 << 30, 32 << 20);
        let mut es = vec![];
        for _ in 0..4 {
            es.push(a.allocate(1 << 20).unwrap());
        }
        let mut ranges: Vec<(usize, usize)> = es.iter().map(|e| (e.off, e.off + e.len)).collect();
        ranges.sort_unstable();
        for w in ranges.windows(2) {
            assert!(w[0].1 <= w[1].0, "extents overlap");
        }
        for e in es {
            a.free(e);
        }
        assert_eq!(a.committed_bytes(), 0);
    }
}
