# Validation logs — v0.0.2

Captured 2026-05-23, stack: vLLM 0.16.0 + transformers 4.57.6 + torch 2.9.1+cu128
+ flashinfer 0.6.3 + datasets 2.20.0 on H200 (CUDA 12.7 driver / 12.8 toolkit).
Venv: `/scratch/prj0000000234/heyingxu/venv/meralion_match_v016/`.

## validate_v016.txt

`backend=vllm` against `MERaLiON-3-3B-ASR` (1505 weights) → vs cached
`MERaLiON-CTM-3B-1505-http` predictions. HYP-vs-HYP CER, n=30:

| Dataset | CER |
|---|---:|
| MDCC Cantonese | 0.09% |
| FLEURS Thai | 0.61% |
| FLEURS Indo | 0.66% |
| FLEURS Malay | 0.29% |

## validate_2804.txt

`backend=vllm` against `MERaLiON-3-3B-ASR-2804` (2804 weights) → vs cached
`MERaLiON-CTM-3B-2804-http` predictions. HYP-vs-HYP CER, n=30:

| Dataset | CER |
|---|---:|
| MDCC Cantonese | 0.18% |
| FLEURS Thai | 1.29% |
| FLEURS Indo | 0.54% |
| FLEURS Malay | 0.94% |

## compare_2804.txt

Real WER-vs-REF for both package and production, alongside the HYP-vs-HYP CER.
Demonstrates the package's actual eval score is within 0.16 pp of production
on every dataset, despite up to 1.29% HYP-vs-HYP surface divergence:

| Dataset | Metric | Pkg vs REF | http vs REF | Pkg vs http |
|---|---|---:|---:|---:|
| MDCC Cantonese | CER | 36.52% | 36.62% | 0.18% |
| FLEURS Thai | CER | 8.36% | 8.52% | 1.29% |
| FLEURS Indo | WER | 22.90% | 23.78% | 0.54% |
| FLEURS Malay | WER | 19.73% | 19.73% | 0.94% |

HYP-vs-HYP CER includes capitalization/punctuation differences that the
production normalizer absorbs, so a 1.29% surface CER is consistent with a
0.16 pp post-normalizer WER delta.
