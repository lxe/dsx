#![cfg(feature = "cuda")]
use dsx::gpu::*;

#[test]
fn round_bf16_kernel() {
    let g = Gpu::new(0).unwrap();
    let x: Vec<f32> = (0..256).map(|i| 1.0 + i as f32 * 1e-3).collect();
    let b = g.upload(&x).unwrap();
    let k = g.kernel("round_bf16").unwrap();
    g.launch(k, (1, 1, 1), 256, 0, &[Arg::P(b.ptr), Arg::I(256)]).unwrap();
    let v: Vec<f32> = g.download(b.ptr, 256).unwrap();
    for i in 0..256 {
        assert_eq!(v[i], dsx::numerics::round_bf16(x[i]));
    }
    eprintln!("{} sm_{} {} SMs", g.name, g.arch, g.sms);
}
