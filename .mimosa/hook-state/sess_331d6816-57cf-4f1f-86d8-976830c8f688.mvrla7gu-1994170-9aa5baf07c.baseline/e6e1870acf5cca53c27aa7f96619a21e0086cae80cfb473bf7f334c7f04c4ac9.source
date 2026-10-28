//! Image input for V4.1's vision tower, as DeepSeek's reference `image_processor.py` prepares it.
//!
//! An image becomes an `n_vit_h x n_vit_w` grid of 14x14 patches for the ViT, and an `n_llm_h x n_llm_w` token grid
//! after the aligner's 3x3 merge, which the language model sees as
//!
//! ```text
//! [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]
//! ```
//!
//! Every position of that span carries the image token id; the token type tells them apart. IMAGE slots take the
//! aligner rows in reading order, the delimiters take learned embeddings.

use anyhow::{Context, Result, bail};
use base64::Engine as _;

pub const PATCH: usize = 14;
pub const DOWNSAMPLE: usize = 3;
pub const MAX_TOKENS: usize = 1024;
pub const MIN_PIXELS: usize = 544 * 544;
/// `<｜image｜>`: the prompt placeholder, and the id every position of an image span carries
pub const IMAGE_TOKEN_ID: u32 = 129264;

pub const IMAGE_START: i8 = 0;
pub const IMAGE: i8 = 1;
pub const IMAGE_NEW_LINE: i8 = 2;
pub const IMAGE_END: i8 = 3;

/// One image, ready for the ViT, and where its span sits in the prompt.
#[derive(Debug, Clone)]
pub struct ImageInput {
    /// first prompt position of the span
    pub start: usize,
    /// [n_vit_h * n_vit_w][3 * 14 * 14], channel-major within a patch, values in [-1, 1]
    pub patches: Vec<f32>,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    pub n_llm_h: usize,
    pub n_llm_w: usize,
    /// one per span position: IMAGE_START / IMAGE / IMAGE_NEW_LINE / IMAGE_END
    pub types: Vec<i8>,
    /// content hash of the decoded image (prefix-cache key)
    pub hash: u64,
}

impl ImageInput {
    pub fn len(&self) -> usize {
        self.types.len()
    }
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

fn llm_grid(best_h: usize, best_w: usize) -> (usize, usize) {
    ((best_h / PATCH).div_ceil(DOWNSAMPLE), (best_w / PATCH).div_ceil(DOWNSAMPLE))
}

fn solve_resize_ratio(height: f64, width: f64) -> (usize, usize) {
    let r = height / width;
    let max_w = ((MAX_TOKENS - 2) as f64 / r + 0.25).sqrt() - 0.5;
    let max_h = max_w * r;
    let cell = PATCH * DOWNSAMPLE;
    if max_w < 1.0 {
        return ((MAX_TOKENS - 2) / 2 * cell, cell);
    }
    if max_h < 1.0 {
        return (cell, (MAX_TOKENS - 3) * cell);
    }
    let beta = (max_w.floor() * cell as f64 / width).min(max_h.floor() * cell as f64 / height);
    (((height * beta) / PATCH as f64).floor() as usize * PATCH, ((width * beta) / PATCH as f64).floor() as usize * PATCH)
}

/// (n_llm_h, n_llm_w, best_height, best_width) for an image of this size: `plan_image_grid` in the reference
pub fn plan_grid(width: usize, height: usize) -> (usize, usize, usize, usize) {
    let (mut w, mut h) = (width as f64, height as f64);
    if w * h > 0.0 && w * h < MIN_PIXELS as f64 {
        let ratio = (MIN_PIXELS as f64 / (w * h)).sqrt();
        w = (w * ratio).trunc();
        h = (h * ratio).trunc();
    }
    let (mut best_w, mut best_h) = ((w / PATCH as f64).ceil() as usize * PATCH, (h / PATCH as f64).ceil() as usize * PATCH);
    let (mut lh, mut lw) = llm_grid(best_h, best_w);
    if num_image_tokens(lh, lw) > MAX_TOKENS {
        (best_h, best_w) = solve_resize_ratio(h, w);
        (lh, lw) = llm_grid(best_h, best_w);
        assert!(num_image_tokens(lh, lw) <= MAX_TOKENS);
    }
    (lh, lw, best_h, best_w)
}

pub fn token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<i8> {
    let mut t = Vec::with_capacity(num_image_tokens(n_llm_h, n_llm_w));
    t.push(IMAGE_START);
    for _ in 0..n_llm_h {
        t.extend(std::iter::repeat_n(IMAGE, n_llm_w));
        t.push(IMAGE_NEW_LINE);
    }
    t.push(IMAGE_END);
    t
}

// ---- Pillow's BICUBIC resize (libImaging/Resample.c), bit for bit: the ViT turns a 0.3% pixel difference on
// text into ~25% at its output, so an approximate resampler is not good enough.

const PRECISION_BITS: u32 = 32 - 8 - 2;

fn bicubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// (bounds [xmin, xcount] per output pixel, fixed-point coefficients [out][ksize], ksize)
fn coeffs(in_size: usize, out_size: usize) -> (Vec<(usize, usize)>, Vec<i32>, usize) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let (mut bounds, mut kk) = (Vec::with_capacity(out_size), vec![0i32; out_size * ksize]);
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
        let mut ww = 0.0;
        for x in 0..xmax {
            let w = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for x in 0..ksize {
            let v = if x < xmax && ww != 0.0 { k[x] / ww } else if x < xmax { k[x] } else { 0.0 };
            let f = v * (1u32 << PRECISION_BITS) as f64;
            kk[xx * ksize + x] = if v < 0.0 { (-0.5 + f) as i32 } else { (0.5 + f) as i32 };
        }
        bounds.push((xmin, xmax));
    }
    (bounds, kk, ksize)
}

fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// `Image.resize((w, h), BICUBIC)` on an RGB image: horizontal pass, then vertical, 8-bit between them
pub fn pil_resize(img: &image::RgbImage, w: usize, h: usize) -> image::RgbImage {
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    let src = img.as_raw();
    let mut tmp = src.clone();
    let mut tw = w0;
    if w != w0 {
        let (b, kk, ks) = coeffs(w0, w);
        let mut out = vec![0u8; w * h0 * 3];
        for y in 0..h0 {
            for xx in 0..w {
                let (xmin, xn) = b[xx];
                let k = &kk[xx * ks..];
                for c in 0..3 {
                    let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                    for x in 0..xn {
                        ss = ss.wrapping_add(src[(y * w0 + xmin + x) * 3 + c] as i32 * k[x]);
                    }
                    out[(y * w + xx) * 3 + c] = clip8(ss);
                }
            }
        }
        tmp = out;
        tw = w;
    }
    if h != h0 {
        let (b, kk, ks) = coeffs(h0, h);
        let mut out = vec![0u8; tw * h * 3];
        for yy in 0..h {
            let (ymin, yn) = b[yy];
            let k = &kk[yy * ks..];
            for x in 0..tw {
                for c in 0..3 {
                    let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                    for y in 0..yn {
                        ss = ss.wrapping_add(tmp[((ymin + y) * tw + x) * 3 + c] as i32 * k[y]);
                    }
                    out[(yy * tw + x) * 3 + c] = clip8(ss);
                }
            }
        }
        tmp = out;
    }
    image::RgbImage::from_raw(w as u32, h as u32, tmp).expect("resize buffer")
}

/// Encoded bytes of a data URL (base64 or percent-encoded body).
pub fn data_url_bytes(url: &str) -> Result<Vec<u8>> {
    let (header, payload) = url.split_once(',').context("data URL without a comma")?;
    if header.contains(";base64") {
        let clean: String = payload.chars().filter(|c| !c.is_ascii_whitespace()).collect();
        return base64::engine::general_purpose::STANDARD.decode(clean).context("data URL: bad base64");
    }
    let b = payload.as_bytes();
    let (mut out, mut i) = (Vec::with_capacity(b.len()), 0);
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok());
            if let Some(v) = hex {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    Ok(out)
}

/// Decode, pad and patchify one image (the reference's `load_image`), for a span starting at `start`.
pub fn load(bytes: &[u8], start: usize) -> Result<ImageInput> {
    let img = image::load_from_memory(bytes).context("image: unsupported or corrupt data (PNG, JPEG, GIF, WebP)")?.to_rgb8();
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    if w0 == 0 || h0 == 0 {
        bail!("image: empty ({w0}x{h0})");
    }
    let (n_llm_h, n_llm_w, best_h, best_w) = plan_grid(w0, h0);
    // ImageOps.pad: fit inside (best_w, best_h) keeping the aspect ratio (Python rounding), bicubic, center on gray
    let (im_r, dst_r) = (w0 as f64 / h0 as f64, best_w as f64 / best_h as f64);
    let (mut rw, mut rh) = (best_w, best_h);
    if im_r != dst_r {
        if im_r > dst_r {
            rh = (h0 as f64 / w0 as f64 * best_w as f64).round_ties_even() as usize;
        } else {
            rw = (w0 as f64 / h0 as f64 * best_h as f64).round_ties_even() as usize;
        }
    }
    let (rw, rh) = (rw.max(1), rh.max(1));
    let resized = if (rw, rh) == (w0, h0) { img } else { pil_resize(&img, rw, rh) };
    let mut canvas = image::RgbImage::from_pixel(best_w as u32, best_h as u32, image::Rgb([127, 127, 127]));
    let (x0, y0) = if rw != best_w {
        (((best_w - rw) as f64 * 0.5).round_ties_even() as i64, 0)
    } else {
        (0, ((best_h - rh) as f64 * 0.5).round_ties_even() as i64)
    };
    image::imageops::replace(&mut canvas, &resized, x0, y0);
    let (n_vit_h, n_vit_w) = (best_h / PATCH, best_w / PATCH);
    let mut patches = vec![0f32; n_vit_h * n_vit_w * 3 * PATCH * PATCH];
    for py in 0..n_vit_h {
        for px in 0..n_vit_w {
            let base = (py * n_vit_w + px) * 3 * PATCH * PATCH;
            for y in 0..PATCH {
                for x in 0..PATCH {
                    let p = canvas.get_pixel((px * PATCH + x) as u32, (py * PATCH + y) as u32);
                    for c in 0..3 {
                        patches[base + c * PATCH * PATCH + y * PATCH + x] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
                    }
                }
            }
        }
    }
    // FNV-1a over the canvas: identical images share a cache key, different ones don't
    let hash = canvas.as_raw().iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    Ok(ImageInput { start, patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w, types: token_types(n_llm_h, n_llm_w), hash })
}

/// Expand each placeholder in `ids` into its image span. Returns the expanded ids, one [`ImageInput`] per image
/// (with `start` set), in order.
pub fn expand(ids: &[u32], images: Vec<Vec<u8>>) -> Result<(Vec<u32>, Vec<ImageInput>)> {
    let n_ph = ids.iter().filter(|&&t| t == IMAGE_TOKEN_ID).count();
    if n_ph != images.len() {
        bail!("prompt has {n_ph} image placeholders for {} images", images.len());
    }
    let (mut out, mut inputs) = (Vec::with_capacity(ids.len()), Vec::with_capacity(images.len()));
    let mut it = images.into_iter();
    for &t in ids {
        if t != IMAGE_TOKEN_ID {
            out.push(t);
            continue;
        }
        let im = load(&it.next().unwrap(), out.len())?;
        out.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, im.len()));
        inputs.push(im);
    }
    Ok((out, inputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_matches_reference() {
        // values from the reference plan_image_grid (fp8-1017/inference/image_processor.py)
        assert_eq!(plan_grid(640, 480), (12, 16, 490, 644));
        assert_eq!(plan_grid(100, 100), (13, 13, 546, 546));
        assert_eq!(plan_grid(4000, 3000), (27, 36, 1134, 1512));
        assert_eq!(plan_grid(4000, 100), (3, 96, 112, 4004));
        assert_eq!(plan_grid(100, 4000), (96, 3, 4004, 112));
        assert_eq!(plan_grid(1920, 1080), (23, 41, 966, 1708));
        assert_eq!(plan_grid(544, 544), (13, 13, 546, 546));
        assert_eq!(plan_grid(1, 1), (13, 13, 546, 546));
        assert_eq!(num_image_tokens(14, 18), 14 * 19 + 2);
    }

    #[test]
    fn data_urls() {
        assert_eq!(data_url_bytes("data:image/png;base64,aGk=").unwrap(), b"hi");
        assert_eq!(data_url_bytes("data:text/plain,a%20b").unwrap(), b"a b");
    }
}
