//! Releasing file-backed pages before large host allocations.

use memmap2::Mmap;
use std::path::Path;

/// Discard this process's PTEs for a read-only file mapping. The mapping stays valid; later reads fault the same
/// bytes back in from the file. The file must not be modified while it is mapped.
pub(crate) fn drop_map(m: &Mmap) {
    let _ = unsafe { m.unchecked_advise(memmap2::UncheckedAdvice::DontNeed) };
}

/// `POSIX_FADV_DONTNEED`: evict clean page-cache pages for a file that is no longer mapped.
pub(crate) fn drop_file(path: &Path) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        const POSIX_FADV_DONTNEED: i32 = 4;
        if let Ok(f) = std::fs::File::open(path) {
            unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = path;
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}
