# triton-kernels

Learning GPU programming with [Triton](https://github.com/triton-lang/triton): puzzles first, then the official tutorial kernels — each with a correctness test, a benchmark, and a roofline analysis. Kernels that graduate from here are used in [mini-vllm](https://github.com/adrianyao64-hub/mini-vllm).

## Triton Puzzles

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adrianyao64-hub/triton-kernels/blob/main/puzzles/Triton-Puzzles.ipynb)

[`puzzles/Triton-Puzzles.ipynb`](puzzles/Triton-Puzzles.ipynb) is the notebook from Sasha Rush's [srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles) (commit `4d794ab`, Apache-2.0, see [`puzzles/LICENSE-Triton-Puzzles`](puzzles/LICENSE-Triton-Puzzles)). The solutions filled in are my own.

The puzzles run in Triton's interpreter mode, so a CPU runtime is enough. Workflow: open the notebook in Colab with the badge, solve, and use *File → Save a copy in GitHub* to commit back to the same path.

| # | Puzzle | Status | Key idea |
| --- | --- | --- | --- |
| 1 | Constant Add | ⬜ | |
| 2 | Constant Add Block | ⬜ | |
| 3 | Outer Vector Add | ⬜ | |
| 4 | Outer Vector Add Block | ⬜ | |
| 5 | Fused Outer Multiplication | ⬜ | |
| 6 | Fused Outer Multiplication – Backwards | ⬜ | |
| 7 | Long Sum | ⬜ | |
| 8 | Long Softmax | ⬜ | |
| 9 | Simple FlashAttention | ⬜ | |
| 10 | Two Dimensional Convolution | ⬜ | |
| 11 | Matrix Multiplication | ⬜ | |
| 12 | Quantized Matrix Mult | ⬜ | |

## Tutorials

Re-implementations of the official [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html), each tested against PyTorch and benchmarked on a GPU via [Modal](https://modal.com) (Triton has no macOS build).

| # | Kernel | Status | Achieved vs. peak | Notes |
| --- | --- | --- | --- | --- |
| 01 | Vector add | ⬜ | | |
| 02 | Fused softmax | ⬜ | | |
| 03 | Matmul | ⬜ | | |
| 05 | LayerNorm | ⬜ | | |
| 06 | Fused attention | ⬜ | | |

```bash
uv sync    # local env (Python 3.12, locked deps)
```

## Layout

```
puzzles/     Triton Puzzles notebook (+ its Apache-2.0 license)
tutorials/   tutorial kernels with tests and benchmarks
```
