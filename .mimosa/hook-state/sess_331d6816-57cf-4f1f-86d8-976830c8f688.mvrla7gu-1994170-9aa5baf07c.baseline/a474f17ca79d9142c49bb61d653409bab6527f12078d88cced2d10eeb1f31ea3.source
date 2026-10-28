use std::{env, path::PathBuf, process::Command};

// Kernels: one HIP code object for gfx942, embedded in the binary.
fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/opt/rocm/lib");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("dsxa.hsaco");
    let hipcc = env::var("HIPCC").unwrap_or_else(|_| "/opt/rocm/bin/hipcc".into());
    if !std::path::Path::new(&hipcc).exists() {
        // no ROCm here (dev machine): build the host code only
        std::fs::write(&out, []).unwrap();
        return;
    }
    let st = Command::new(hipcc)
        .args(["--genco", "--offload-arch=gfx942", "-O3", "-std=c++17"])
        .args(["-mllvm", "-amdgpu-early-inline-all=true"])
        .arg("kernels/dsxa.hip")
        .arg("-o")
        .arg(&out)
        .status()
        .expect("hipcc");
    assert!(st.success(), "kernel compilation failed");
}
