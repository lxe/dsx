use std::{env, path::PathBuf, process::Command};

// Kernels: one HIP code object, embedded in the binary. DSX_AMD_ARCH picks the target: gfx942 (default) or gfx950.
fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=DSX_AMD_ARCH");
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/rocm/lib");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("dsxa.hsaco");
    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "/opt/rocm/bin/hipcc".into());
    if !std::path::Path::new(&hipcc).exists() {
        // no ROCm here (dev machine): build the host code only
        std::fs::write(&out, []).unwrap();
        return;
    }
    let arch = env::var("DSX_AMD_ARCH").unwrap_or_else(|_| "gfx942".into());
    assert!(matches!(arch.as_str(), "gfx942" | "gfx950"), "DSX_AMD_ARCH must be gfx942 or gfx950, got {arch}");
    let st = Command::new(hipcc)
        .args(["--genco", &format!("--offload-arch={arch}"), "-O3", "-std=c++17"])
        .args(["-mllvm", "-amdgpu-early-inline-all=true"])
        .arg("kernels/dsxa.hip")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("hipcc");
    assert!(st.success(), "kernel compilation failed");
}
