//! deqrow GGUF TENSOR ROW [N]: decode one row of a GGUF tensor with dsx's reference decoder (formats::dequant_row) and
//! print its first N values and the row's sum, to compare against gguf-py's dequantize.
use dsx_core::formats::dequant_row;
use dsx_core::gguf::Gguf;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(std::path::Path::new(&a[1]))?;
    let t = g.get(&a[2])?;
    let row: usize = a[3].parse()?;
    let n: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(8);
    let rb = t.ty.row_bytes(t.row_len());
    let b = g.bytes(t);
    let mut out = vec![0f32; t.row_len()];
    dequant_row(t.ty, &b[row * rb..(row + 1) * rb], &mut out);
    println!("{:?} {}x{} row {row}: first {:?} sum {}", t.ty, t.n_rows(), t.row_len(), &out[..n], out.iter().map(|&v| v as f64).sum::<f64>());
    Ok(())
}
