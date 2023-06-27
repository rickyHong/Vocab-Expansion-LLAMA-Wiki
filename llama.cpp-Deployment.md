The research community has developed many excellent model quantization and deployment tools to help users **easily deploy large models locally on their own computers (CPU!)**. In the following, we'll take the [llama.cpp tool](https://github.com/ggerganov/llama.cpp) as an example and introduce the detailed steps to quantize and deploy the model on MacOS and Linux systems. For Windows, you may need to install build tools like cmake. **For a local quick deployment experience, it is recommended to use the instruction-finetuned Alpaca model.**

Before running, please ensure:

1. The system should have `make` (built-in for MacOS/Linux) or `cmake` (need to be installed separately for Windows) build tools.
2. It is recommended to use Python 3.10
3. The latest llama.cpp adds GPU support. Please refer to [https://github.com/ggerganov/llama.cpp/discussions/915](https://github.com/ggerganov/llama.cpp/discussions/915)

### Step 1: Clone and build llama.cpp

1. Clone llama.cpp repository

```
git clone https://github.com/ggerganov/llama.cpp
```

2. (Optional) If you want to use `k-quants` series (usually has better quantization perf.), please edit `llama.cpp` file (near line 2500):

- Original code: `if (nx % QK_K != 0 || ny % QK_K != 0) {` 

- Modified one: `if (nx % QK_K != 0) {`

3. Run the following commands to build the llama.cpp project, generating `./main` and `./quantize` binary files.

```bash
make
```

- **Windows/Linux are recommended to build with BLAS/cuBLAS**, which improves the speed of prompt processing. check：https://github.com/ggerganov/llama.cpp#blas-build
- no further build requirements for macOS users, as llama.cpp has been optimized for ARM NEON and the BLAS is automatically enabled.
  - **Recommended for M-series**: build with Metal will significantly improve inference speed, just replace with `LLAMA_METAL=1 make`. Refer to [llama.cpp](https://github.com/ggerganov/llama.cpp#metal-build)


### Step 2: Generate a quantized model

Depending on the type of model you want to convert (LLaMA or Alpaca), place the `tokenizer.*` files from the downloaded LoRA model package into the `zh-models` directory, and place the `params.json`  and the `consolidate.*.pth` model file obtained in the last step of [Model Conversion](./Manual-Conversion) into the `zh-models/7B` directory. Note that the `.pth` model file and `tokenizer.model` are corresponding, and the `tokenizer.model` for LLaMA and Alpaca should not be mixed. The directory structure should be similar to:

```
llama.cpp/zh-models/
   - 7B/
     - consolidated.00.pth
     - params.json
   - tokenizer.model
```

Convert the above `.pth` model weights to ggml's FP16 format, and generate a file with the path `zh-models/7B/ggml-model-f16.bin`.

```bash
python convert.py zh-models/7B/
```

Further quantize the FP16 model to 4-bit, and generate a quantized model file with the path `zh-models/7B/ggml-model-q4_0.bin`. For more quantization methods and their performances, please refer to the end of this wiki.

```bash
./quantize ./zh-models/7B/ggml-model-f16.bin ./zh-models/7B/ggml-model-q4_0.bin q4_0
```

### Step 3: Load and start the model

Run the `./main` binary file, with the `-m` command specifying the 4-bit quantized model (or loading the ggml-FP16 model). Below is an example of decoding parameters.

**If you have already compiled with Meta, you can add `-ngl 1` to enable Apple Silicon GPU inference.**

```bash
./main -m zh-models/7B/ggml-model-q4_0.bin --color -f ./prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.1
```

Please enter your prompt after the `>`, use `\` as the end of the line for multi-line inputs. To view help and parameter instructions, please execute the `./main -h` command. Here's a brief introduction to several important parameters:

```
-c controls the length of context, larger values allow for longer dialogue history to be referenced
-ins activates the instruction mode (similar to ChatGPT)
-f load prompt template, please use prompts/alpaca.txt for alpaca models
-n controls the maximum length of generated responses
-b batch size
-t number of threads
--repeat_penalty controls the penalty for repeated text in the generated response
--temp is the temperature coefficient, lower values result in less randomness in the response, and vice versa
--top_p, top_k control the sampling parameters
```

Please refer to official guidelines for further information: [https://github.com/ggerganov/llama.cpp/tree/master/examples/main](https://github.com/ggerganov/llama.cpp/tree/master/examples/main)

### About quantization performance

The table below provides reference statistical data for different quantization methods. The inference models used were Chinese Alpaca-Plus-7B and Alpaca-Plus-13B, and the testing was done on an M1 Max chip (8x performance cores, 2x efficiency cores). The reported speed refers to the "eval time", which is the speed of model response generation. For more information on quantization parameters, please refer to the [llama.cpp quantization table](https://github.com/ggerganov/llama.cpp#quantization).

Takeaways:

- The default quantization method is q4_0, which is the fastest but has the highest loss. Each method has its pros and cons, and the appropriate method should be selected according to the actual situation.
- If resources are sufficient and speed requirements are not too strict, q8_0 can be used, which is similar to the performance of an F16 model.
- It should be noted that F16 and q8_0 do not improve much in speed when the number of threads is increased.
- The speed is the fastest when the number of threads `-t` is consistent with the number of physical cores. If it exceeds this number, the speed will actually slow down (on M1 Max, changing from 8 to 10 threads resulted in 3x slow down).
- If you enabled GPU decoding with Metal, there will be another speed up (marked with `-ngl 1`). Now supports Q4_0, Q2_K, Q6_K, and Q4_K series.


#### Alpaca-Plus-7B

|                   |    F16 |   Q2_K | Q3_K_S | Q3_K_M<br/>(Q3_K) | Q3_K_L |   Q4_0 |   Q4_1 | Q4_K_S | Q4_K_M<br/>(Q4_K) |   Q5_0 |   Q5_1 | Q5_K_S | Q5_K_M<br/>(Q5_K) |   Q6_K |   Q8_0 |
| ----------------- | -----: | -----: | -----: | ----------------: | -----: | -----: | -----: | -----: | ----------------: | -----: | -----: | -----: | ----------------: | -----: | -----: |
| PPL               | 10.793 | 18.292 | 15.276 |            12.504 | 11.548 | 12.416 | 12.002 | 11.717 |            11.062 | 11.155 | 10.905 | 10.930 |            10.869 | 10.845 | 10.790 |
| Size              | 13.77G |  2.95G |  3.04G |             3.37G |  3.69G |  4.31G |  5.17G |  3.93G |             4.18G |  4.74G |  5.17G |  4.76G |             4.89G |  5.65G |  7.75G |
| ms/tok @ `-t 2`   |    144 |        |        |                   |        |     87 |     88 |        |                   |    143 |    157 |        |                   |        |    103 |
| ms/tok @ `-t 4`   |    123 |        |        |                   |        |     50 |     52 |        |                   |     75 |     82 |        |                   |        |     72 |
| ms/tok @ `-t 8`   |    126 |     48 |     57 |                52 |     54 |     41 |     49 |     45 |                47 |     46 |     49 |     52 |                54 |     58 |     69 |
| ms/tok @ `-ngl 1` |      x |     28 |     32 |                32 |     33 |     28 |      x |     32 |                30 |      x |      x |     32 |                32 |     33 |      x |

#### Alpaca-Plus-13B

|                   |   F16 |   Q2_K | Q3_K_S | Q3_K_M<br/>(Q3_K) | Q3_K_L |  Q4_0 |  Q4_1 | Q4_K_S | Q4_K_M<br/>(Q4_K) |  Q5_0 |  Q5_1 | Q5_K_S | Q5_K_M<br/>(Q5_K) |   Q6_K |   Q8_0 |
| ----------------- | ----: | -----: | -----: | ----------------: | -----: | ----: | ----: | -----: | ----------------: | ----: | ----: | -----: | ----------------: | -----: | -----: |
| PPL               | 9.147 | 15.455 | 11.488 |            10.229 | 9.5372 | 9.917 | 9.689 |  9.947 |             9.295 | 9.325 | 9.344 |  9.286 |             9.246 |  9.169 |  9.147 |
| Size              | 26.4G |  5.61G |  5.77G |             6.43G |  7.04G | 8.25G |  9.9G |  7.49G |             7.99G | 9.08G |  9.9G |  9.11G |             9.37G | 10.83G | 14.85G |
| ms/tok @ `-t 2`   |       |        |        |                   |        |   166 |   166 |        |                   |   273 |   304 |        |                   |        |    192 |
| ms/tok @ `-t 4`   |       |        |        |                   |        |    89 |    94 |        |                   |   142 |   155 |        |                   |        |    132 |
| ms/tok @ `-t 8`   |       |     83 |     99 |                94 |     99 |    77 |    89 |     77 |                81 |    86 |    93 |     93 |                93 |    104 |    132 |
| ms/tok @ `-ngl 1` |     x |     52 |     56 |                57 |     59 |    49 |     x |     58 |                55 |     x |     x |     57 |                57 |     59 |      x |

#### Alpaca-33B

|                 |    F16 |  Q2_K  | Q3_K_S | Q3_K_M<br/>(Q3_K) | Q4_0   | Q4_1   | Q4_K_S | Q4_K | Q5_0   | Q5_1   | Q5_K_S | Q6_K   |   Q8_0 |
| :-------------- | -----: | ----: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | -----: | -----: | -----: | -----: |
| PPL             | 10.692 | 13.040 | 11.363 | 11.365 | 10.999 | 11.085 | 11.007 | 10.840 | 10.717 | 10.747 | 10.802 | 10.713 |        |
| Size            | 61.03G | 12.74G | 14.21G | 14.65G | 17.16G | 19.07G | 17.16G | 18.43G | 20.98G | 24.58G | 20.98G | 25.03G | 32.42G |
| ms/tok @ `-t 2` |      - |        |        |        | 482    | 481    |        |     | 702    | 919    |        |        |      - |
| ms/tok @ `-t 4` |      - |        |        |        | 251    | 249    |        |     | 355    | 487    |        |        |      - |
| ms/tok @ `-t 8` |      - |  174  | 238 | 242    | 170    | 185    |        | 194 | 224    | 306    |        |        |      - |
| ms/tok @ `-ngl 1` | - | 127 | 130 | 128 | 120 | x | x | 181 | x | x | x | x | x |