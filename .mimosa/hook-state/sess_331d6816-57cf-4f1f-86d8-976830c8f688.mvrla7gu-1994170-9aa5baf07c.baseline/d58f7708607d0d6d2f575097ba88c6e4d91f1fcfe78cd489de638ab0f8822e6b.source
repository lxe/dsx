//! dsx's T16 decoder against exllamav3's own output (core/tests/data/t16: quant/t16/t16_golden.py on the patched
//! exllamav3 build, column-major positions; weights are (in, out), y = x @ w_eff).
use dsx_core::gguf::GType;
use dsx_core::safetensors::SafeTensors;
use dsx_core::t16;
use half::f16;
use std::path::Path;

fn f16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect()
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn decoder_matches_exllamav3() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/t16");
    if !dir.exists() {
        eprintln!("no golden files in {}", dir.display());
        return;
    }
    for (kname, ty) in [("1.5", GType::T16K15), ("2", GType::T16K2), ("2.5", GType::T16K25)] {
        let st = SafeTensors::open_file(&dir.join(format!("t16_golden_K{kname}_cm.safetensors"))).unwrap();
        let get = |n: &str| st.bytes(st.get(n).unwrap()).to_vec();
        let (k, n) = (512, 256);
        let k2 = t16::k2(ty).unwrap();
        // the stored matrix as dsx keeps it: trellis | suh | svh
        let ex = get("trellis");
        assert_eq!(ex.len(), t16::trellis_bytes(k2, k, n));
        let mut m = t16::from_exl3(&ex, k2, k, n);
        let mul = get("mul1");
        assert_eq!(u32::from_le_bytes(mul[..4].try_into().unwrap()), t16::MUL, "mul1");
        m.extend(get("suh"));
        m.extend(get("svh"));
        // 1. the decoded trellis, value for value
        let inner = t16::decode_inner(&m[..t16::trellis_bytes(k2, k, n)], k2, k, n);
        let want = f16s(&get("w_inner"));
        let bad = inner.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(bad, 0, "K {kname}: {bad} of {} inner weights differ from exllamav3's reconstruct", want.len());
        // 2. the effective matrix (exllamav3 computes it in fp16: compare within its rounding)
        let eff = t16::dequant_effective(&m, ty, k, n); // [n][k]
        let weff = f16s(&get("w_eff")); // (in, out)
        let mx = weff.iter().fold(0f32, |a, v| a.max(v.abs()));
        let e = (0..k).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| (eff[j * k + i] - weff[i * n + j]).abs()).fold(0f32, f32::max) / mx;
        // 3. the forward pass
        let x = f32s(&get("x"));
        let y = f32s(&get("y"));
        let yd = t16::forward(&m, ty, k, n, &x);
        let ym = y.iter().fold(0f32, |a, v| a.max(v.abs()));
        let ey = y.iter().zip(&yd).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / ym;
        println!("K {kname}: inner exact; w_eff max err {e:.2e} of max; y max err {ey:.2e} of max");
        assert!(e < 3e-3 && ey < 3e-3, "K {kname}: w_eff {e}, y {ey}");
    }
}
