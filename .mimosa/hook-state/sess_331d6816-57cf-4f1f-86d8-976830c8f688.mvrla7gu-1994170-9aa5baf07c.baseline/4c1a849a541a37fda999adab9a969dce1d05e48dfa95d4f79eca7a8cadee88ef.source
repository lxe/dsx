//! Dump an image's ViT patches as raw f32 (for checking against the reference preprocessor).
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let im = dsx_core::vision::load(&std::fs::read(&a[1])?, 0)?;
    eprintln!("vit {}x{} llm {}x{} tokens {}", im.n_vit_h, im.n_vit_w, im.n_llm_h, im.n_llm_w, im.len());
    let bytes: Vec<u8> = im.patches.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[2], bytes)?;
    Ok(())
}
