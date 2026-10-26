//! Small reference-test utility; never loads a model or accesses a GPU.
use dsx_core::{formats::dequant_row, gguf::GType};
use std::io::{Read, Write};
fn main() {
    let ty = GType::from_id(std::env::args().nth(1).expect("type id").parse().unwrap()).unwrap();
    assert!(matches!(ty, GType::Q2K | GType::Iq3Xxs));
    let mut raw = Vec::new();
    std::io::stdin().read_to_end(&mut raw).unwrap();
    let (n, b) = ty.block();
    assert_eq!(raw.len() % b, 0);
    let mut out = vec![0f32; raw.len() / b * n];
    dequant_row(ty, &raw, &mut out);
    let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
    for v in out {
        stdout.write_all(&v.to_le_bytes()).unwrap();
    }
}
