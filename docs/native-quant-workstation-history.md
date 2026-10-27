# Historical full-model workstation measurements

These measurements are from the October 23–24, 2026 development builds, not the rebased PR revision. They document observed end-to-end throughput; current-tree CUDA and full-model validation remain outstanding.

Model: **DeepSeek-V4.1-Flash-1017**, Unsloth **UD-IQ2_XXS mixed GGUF**, six shards, no LoRA. This export contains Q2_K and IQ3_XXS expert tensors; its name does not describe every tensor's quantization.

Hardware: Intel Core i9-10900X, 128 GiB RAM, RTX 3090 24 GiB + CMP 170HX 64 GiB, RTX first. Recorded links: RTX PCIe 3.0 x16, CMP PCIe 2.0 x16.

Configuration: 14 expert workers on logical CPUs 0–13; context 4096; chunk 128; arena 80 GB; main-gb=-6; aux-gb=-3; MemoryMax=105G; DSX_ADAPT=0; DSX_AUX_ADAPTIVE=0; no ranking or defer. CPU affinity includes SMT; this is the measured configuration, not a recommendation.

Each experiment ran A–B–A–B, four requests per run: a short German explanation, Python merge_intervals, HTML/CSS, then the repeated Python prompt. Completion counts were 48, 195, 256, 195; HTML/CSS reached the token cap. Throughput is the report's approximate streaming decode rate `(completion_tokens - 1) / (total_seconds - ttft_seconds)`, excluding time to first token. Table entries are medians across all eight request rates per variant, not peak rates or pooled token/time throughput. All eight rates are retained below. Both comparisons report matching answer text; this is not a general quality evaluation.

| Change | Before (tokens/s) | After (tokens/s) | Throughput gain |
|---|---:|---:|---:|
| IQ3_XXS SIMD | 6.40 | 9.69 | +51.3% |
| Q2_K register unpack (on IQ3-patched baseline) | 9.72 | 9.78 | +0.6% |

The Q2_K increment is small and may be within run-to-run noise; these runs do not establish a meaningful full-model gain from Q2_K alone. Neither gain is a prediction for another model or hardware. Startup, disk-cache state and memory pressure were not eliminated: early requests incurred NVMe reads and memory-limit events, whereas repeated prompts were substantially warmer. These are small workstation experiments, not a controlled population study. Placement/ranking experiments are deliberately excluded because their gains are not changes in this PR.

## Source provenance and request measurements

Archive hashes identify locally retained source reports; the original archives are not committed because they contain workstation-specific paths and telemetry. The following is a sanitized extraction of `comparison.json`.

### IQ3_XXS SIMD

Source: `results.tar(9).gz`; SHA256 `fed4b49354268fb5cfe43d20ab41e1d811251973731d5615f555aa2527596dcd`.

| Run | Request 0 TPS | Request 1 TPS | Request 2 TPS | Request 3 TPS |
|---|---:|---:|---:|---:|
| 01-original-A | 6.118204 | 6.525545 | 6.424244 | 6.222571 |
| 02-patched-A | 8.759343 | 9.648479 | 9.411971 | 9.830334 |
| 03-original-B | 6.047242 | 6.455458 | 6.382253 | 6.534760 |
| 04-patched-B | 8.968939 | 9.729435 | 9.769229 | 9.847985 |

### Q2_K register unpack (on IQ3-patched baseline)

Source: `results.tar(20261024-072627).gz`; SHA256 `42f112eaf765232526327ecfad3b08bd2a4d8e64535a5eeb8cfbd58321714cbc`.

| Run | Request 0 TPS | Request 1 TPS | Request 2 TPS | Request 3 TPS |
|---|---:|---:|---:|---:|
| 01-iq3-A | 8.822054 | 9.715548 | 9.728497 | 9.851284 |
| 02-q2-A | 8.920249 | 9.790999 | 9.765933 | 9.826745 |
| 03-iq3-B | 9.243675 | 9.688954 | 9.736528 | 9.825936 |
| 04-q2-B | 8.903381 | 9.787629 | 9.773301 | 9.837366 |

## Sanitized timing data

```json
[
  {
    "change": "IQ3_XXS SIMD",
    "archive": "results.tar(9).gz",
    "sha256": "fed4b49354268fb5cfe43d20ab41e1d811251973731d5615f555aa2527596dcd",
    "before": 6.403248104694692,
    "after": 9.68895655040427,
    "gain": 51.31315220006209,
    "answers_match": true,
    "runs": [
      {
        "name": "01-original-A",
        "requests": [
          {
            "index": 0,
            "completion_tokens": 48,
            "ttft_seconds": 1.8578355270001339,
            "total_seconds": 9.539828173000387,
            "approx_decode_tokens_per_second": 6.118204242810785
          },
          {
            "index": 1,
            "completion_tokens": 195,
            "ttft_seconds": 1.4773869299997386,
            "total_seconds": 31.2067049999996,
            "approx_decode_tokens_per_second": 6.525544902954476
          },
          {
            "index": 2,
            "completion_tokens": 256,
            "ttft_seconds": 1.2068076650002695,
            "total_seconds": 40.900195977000294,
            "approx_decode_tokens_per_second": 6.4242437051640895
          },
          {
            "index": 3,
            "completion_tokens": 195,
            "ttft_seconds": 1.3063347819997944,
            "total_seconds": 32.48315804899994,
            "approx_decode_tokens_per_second": 6.222571117607866
          }
        ]
      },
      {
        "name": "02-patched-A",
        "requests": [
          {
            "index": 0,
            "completion_tokens": 48,
            "ttft_seconds": 1.906713564000711,
            "total_seconds": 7.272412617000555,
            "approx_decode_tokens_per_second": 8.759343290735499
          },
          {
            "index": 1,
            "completion_tokens": 195,
            "ttft_seconds": 1.508122884999466,
            "total_seconds": 21.614919920999455,
            "approx_decode_tokens_per_second": 9.648478554423903
          },
          {
            "index": 2,
            "completion_tokens": 256,
            "ttft_seconds": 1.2510044950004158,
            "total_seconds": 28.344161489000726,
            "approx_decode_tokens_per_second": 9.411970707454612
          },
          {
            "index": 3,
            "completion_tokens": 195,
            "ttft_seconds": 1.3041412390002733,
            "total_seconds": 21.038974482000413,
            "approx_decode_tokens_per_second": 9.83033388786353
          }
        ]
      },
      {
        "name": "03-original-B",
        "requests": [
          {
            "index": 0,
            "completion_tokens": 48,
            "ttft_seconds": 1.8544821739997133,
            "total_seconds": 9.626620000999537,
            "approx_decode_tokens_per_second": 6.047242219087459
          },
          {
            "index": 1,
            "completion_tokens": 195,
            "ttft_seconds": 1.5042872109997916,
            "total_seconds": 31.55637484199997,
            "approx_decode_tokens_per_second": 6.455458348919483
          },
          {
            "index": 2,
            "completion_tokens": 256,
            "ttft_seconds": 1.2574644110000008,
            "total_seconds": 41.21201021300021,
            "approx_decode_tokens_per_second": 6.382252504225293
          },
          {
            "index": 3,
            "completion_tokens": 195,
            "ttft_seconds": 1.305815172999246,
            "total_seconds": 30.99320842599991,
            "approx_decode_tokens_per_second": 6.534760339067202
          }
        ]
      },
      {
        "name": "04-patched-B",
        "requests": [
          {
            "index": 0,
            "completion_tokens": 48,
            "ttft_seconds": 1.7641092399999252,
            "total_seconds": 7.004416974000378,
            "approx_decode_tokens_per_second": 8.968938922241534
          },
          {
            "index": 1,
            "completion_tokens": 195,
            "ttft_seconds": 1.8886984709997705,
            "total_seconds": 21.828192289999606,
            "approx_decode_tokens_per_second": 9.72943454638464
          },
          {
            "index": 2,
            "completion_tokens": 256,
            "ttft_seconds": 1.2273287649995837,
            "total_seconds": 27.32969440299985,
            "approx_decode_tokens_per_second": 9.76922948427198
          },
          {
            "index": 3,
            "completion_tokens": 195,
            "ttft_seconds": 1.3070996110000124,
            "total_seconds": 21.006561707999936,
            "approx_decode_tokens_per_second": 9.8479846325116
          }
        ]
      }
    ]
  },
  {
    "change": "Q2_K register unpack (on IQ3-patched baseline)",
    "archive": "results.tar(20261024-072627).gz",
    "sha256": "42f112eaf765232526327ecfad3b08bd2a4d8e64535a5eeb8cfbd58321714cbc",
    "before": 9.722022358799308,
    "after": 9.780465049614877,
    "gain": 0.6011371776230678,
    "answers_match": true,
    "runs": [
      {
        "name": "01-iq3-A",
        "requests": [
          {
            "index": 0,
            "completion_tokens": 48,
            "ttft_seconds": 1.8392717789993185,
            "total_seconds": 7.166829105000943,
            "approx_decode_tokens_per_second": 8.822054296931213
          },
          {
            "index": 1,
            "completion_tokens": 195,
            "ttft_seconds": 1.4870374109996192,
            "total_seconds": 21.455031579000206,
            "approx_decode_tokens_per_second": 9.71554770938845
          },
          {
            "index": 2,
            "completion_tokens": 256,
            "ttft_seconds": 1.2280970710016845,
            "total_seconds": 27.439751326000987,
            "approx_decode_tokens_per_second": 9.728497008210166
          },
          {
            "index": 3,
            "completion_tokens": 195,
            "ttft_seconds": 1.3045699060003244,
            "total_seconds": 20.997434864000752,
            "approx_decode_tokens_per_second": 9.85128372198508
          }
        ]
      },
      {
        "name": "02-q2-A",
        "requests": [
          {
            "index": 0,
            "completion_tokens": 48,
            "ttft_seconds": 1.782967101000395,
            "total_seconds": 7.051878340000258,
            "approx_decode_tokens_per_second": 8.920248959996046
          },
          {
            "index": 1,
            "completion_tokens": 195,
            "ttft_seconds": 1.4377662990009412,
            "total_seconds": 21.251882946000478,
            "approx_decode_tokens_per_second": 9.79099918791371
          },
          {
            "index": 2,
            "completion_tokens": 256,
            "ttft_seconds": 1.2064389360002679,
            "total_seconds": 27.317616127998917,
            "approx_decode_tokens_per_second": 9.76593273160203
          },
          {
            "index": 3,
            "completion_tokens": 195,
            "ttft_seconds": 1.3040824069994414,
            "total_seconds": 21.046122392999678,
            "approx_decode_tokens_per_second": 9.826745368643367
          }
        ]
      },
      {
        "name": "03-iq3-B",
        "requests": [
          {
            "index": 0,
            "completion_tokens": 48,
            "ttft_seconds": 1.310389956999643,
            "total_seconds": 6.394947864999267,
            "approx_decode_tokens_per_second": 9.24367483868245
          },
          {
            "index": 1,
            "completion_tokens": 195,
            "ttft_seconds": 1.3406403040007717,
            "total_seconds": 21.363440499999342,
            "approx_decode_tokens_per_second": 9.688954496922447
          },
          {
            "index": 2,
            "completion_tokens": 256,
            "ttft_seconds": 1.2425053720016876,
            "total_seconds": 27.432538858000044,
            "approx_decode_tokens_per_second": 9.736528215449873
          },
          {
            "index": 3,
            "completion_tokens": 195,
            "ttft_seconds": 1.305037022999386,
            "total_seconds": 21.04870233699876,
            "approx_decode_tokens_per_second": 9.825936416296678
          }
        ]
      },
      {
        "name": "04-q2-B",
        "requests": [
          {
            "index": 0,
            "completion_tokens": 48,
            "ttft_seconds": 2.052739624001333,
            "total_seconds": 7.331633254001645,
            "approx_decode_tokens_per_second": 8.903380763896397
          },
          {
            "index": 1,
            "completion_tokens": 195,
            "ttft_seconds": 1.453053394001472,
            "total_seconds": 21.2739932530003,
            "approx_decode_tokens_per_second": 9.787628708833543
          },
          {
            "index": 2,
            "completion_tokens": 256,
            "ttft_seconds": 1.2117356960006873,
            "total_seconds": 27.30322615700061,
            "approx_decode_tokens_per_second": 9.773301390396211
          },
          {
            "index": 3,
            "completion_tokens": 195,
            "ttft_seconds": 1.3041466109989415,
            "total_seconds": 21.024873030999515,
            "approx_decode_tokens_per_second": 9.837365818494751
          }
        ]
      }
    ]
  }
]
```
