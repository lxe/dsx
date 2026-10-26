# Credits

[exllamav3](https://github.com/turboderp-org/exllamav3) (turboderp, MIT) for the trellis quantizer T16 patches · [Unsloth](https://github.com/unslothai/unsloth) for dsx-t16's IQ4_NL Engram table and the expert weights in dsx-ud, from their [UD-IQ1_M GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) · [DeepSeek](https://huggingface.co/deepseek-ai) for the model (MIT) ·
[Xolotl / Strata](https://github.com/Indras-Mirror/Xolotl) for the GSQ tools and V4.1 GGUF format (MIT; derived files in `quant/v41` carry its license) ·
DeepSeek for the reference `model.py` the calibration runs · GPTQ ([Frantar et al.](https://arxiv.org/abs/2210.17323)) ·
[Lucebox](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF) for the eval set ·
antirez for the calibration prompts · the web UI is [llama.cpp](https://github.com/ggml-org/llama.cpp)'s, by Georgi
Gerganov and the ggml authors (MIT), vendored in [`core/webui`](core/webui) with its license.


MIT license; vendored and derived components retain their own license notices.

Native Q2_K and IQ3_XXS layouts, codebook tables and reference fixtures follow
[ggml](https://github.com/ggml-org/llama.cpp/tree/da8e2a54b919557b816b1c2beb432ccb766f7ce0/ggml)
(MIT); the table files retain the ggml license notice. Fixture regeneration is documented in
[`docs/native-quant.md`](docs/native-quant.md).
