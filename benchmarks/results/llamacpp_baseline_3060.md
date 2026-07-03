| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| qwen3 0.6B Q4_K - Medium       | 372.65 MiB |   596.05 M | CUDA       |  99 |           pp512 |   13301.44 ± 1392.80 |
| qwen3 0.6B Q4_K - Medium       | 372.65 MiB |   596.05 M | CUDA       |  99 |           tg128 |        345.30 ± 3.62 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | CUDA       |  99 |           pp512 |    12012.39 ± 597.70 |
| llama 1B Q8_0                  |   1.22 GiB |     1.24 B | CUDA       |  99 |           tg128 |        212.27 ± 2.50 |
| gemma3 1B Q4_K - Medium        | 762.49 MiB |   999.89 M | CUDA       |  99 |           pp512 |    10965.69 ± 507.58 |
| gemma3 1B Q4_K - Medium        | 762.49 MiB |   999.89 M | CUDA       |  99 |           tg128 |        225.08 ± 2.93 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | CUDA       |  99 |           pp512 |     3929.40 ± 186.02 |
| phi3 3B Q4_K - Medium          |   2.31 GiB |     3.84 B | CUDA       |  99 |           tg128 |        103.83 ± 1.82 |
| granite 3B Q4_K - Medium       |   1.44 GiB |     2.53 B | CUDA       |  99 |           pp512 |     4601.18 ± 175.37 |
| granite 3B Q4_K - Medium       |   1.44 GiB |     2.53 B | CUDA       |  99 |           tg128 |        145.54 ± 1.28 |
| olmoe A1.7B Q4_K - Medium      |   3.92 GiB |     6.92 B | CUDA       |  99 |           pp512 |      5863.29 ± 76.46 |
| olmoe A1.7B Q4_K - Medium      |   3.92 GiB |     6.92 B | CUDA       |  99 |           tg128 |        280.56 ± 1.63 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |           pp512 |      2098.38 ± 42.62 |
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.25 B | CUDA       |  99 |           tg128 |         65.49 ± 0.98 |

build: 6f4f53f (1)
