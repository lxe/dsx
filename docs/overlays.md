# Overlays

An overlay is a small GGUF that replaces some of a model's tensors when it loads. A weight edit that leaves the
routed experts alone, such as an abliteration (refusal removal), changes only a few dense matrices. An overlay
carries just those matrices, so the edited model runs from the same T16 file as the original.

```bash
./target/release/dsx-server --model dsx-t16/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf \
  --overlay DeepSeek-V4.1-Flash-1017-dsx-t16-uncensored-overlay.gguf
```

Every overlay tensor must already exist in the model with the same shape and type, and an overlay names the model
it was made for (`dsx.overlay.base`). dsx refuses to load a mismatch. The multi-session cache treats the model with
and without an overlay as different models.

## Uncensored overlay

Built from [dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8),
which removes the model's refusals by editing weights. Against DeepSeek's release, that checkpoint changes two
matrices in each of layers 10-36 and nothing else:

| Tensor (HF / GGUF) | Shape | Layers |
|---|---|---|
| `attn.wo_b` / `attn_output_b` | 5120 x 8192 | 10-36 |
| `ffn.shared_experts.w2` / `ffn_down_shexp` | 5120 x 2304 | 10-36 |

The routed experts, attention inputs, Engram, MTP and vision weights are byte-identical to DeepSeek's, so the T16
experts apply unchanged. The overlay holds the 54 edited matrices in Q8_0, the same type as the T16 model's dense
weights.

Download: [lxe/DeepSeek-V4.1-Flash-1017-dsx-t16-uncensored-overlay](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-t16-uncensored-overlay)
(1.54 GB). Based on dealignai's release; the refusal removal is their work.

## Making one

`quant/overlay_gguf.py` reads only the named tensors from a Hugging Face checkpoint in DeepSeek's FP8 layout, by
range requests, and quantizes them to Q8_0:

```bash
python3 quant/overlay_gguf.py uncensored-overlay.gguf DeepSeek-V4.1-Flash-1017-dsx-t16.gguf \
  dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8 10-36 attn.wo_b,ffn.shared_experts.w2 \
  --check deepseek-ai/DeepSeek-V4.1-Flash-1017
```

`--check` first quantizes the same tensors from DeepSeek's release and requires them to match the T16 model byte for
byte, which shows the overlay differs from the model only where the edit does.
