The research community has developed many excellent model quantization and deployment tools to help users **easily deploy large models locally on their own computers (CPU!)**. In the following, we'll take the [llama.cpp tool](https://github.com/ggerganov/llama.cpp) as an example and introduce the detailed steps to quantize and deploy the model on MacOS and Linux systems. For Windows, you may need to install build tools like cmake. **For a local quick deployment experience, it is recommended to use the instruction-finetuned Alpaca model.**

Before running, please ensure:

1. The model quantization process requires loading the entire unquantized model into memory, so make sure there is enough available memory (7B version requires more than 13G).
2. When loading the quantized model (e.g., the 7B version), ensure that the available memory on the machine is greater than 4-6G (affected by context length).
3. The system should have `make` (built-in for MacOS/Linux) or `cmake` (need to be installed separately for Windows) build tools.
4. It is recommended to use Python 3.9 or 3.10 to build and run the [llama.cpp tool](https://github.com/ggerganov/llama.cpp).
5. The latest llama.cpp adds GPU support. Please refer to [https://github.com/ggerganov/llama.cpp/discussions/915](https://github.com/ggerganov/llama.cpp/discussions/915)

### Step 1: Clone and build llama.cpp

Run the following commands to build the llama.cpp project, generating `./main` and `./quantize` binary files.

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
```

### Step 2: Generate a quantized model

Depending on the type of model you want to convert (LLaMA or Alpaca), place the `tokenizer.*` files from the downloaded LoRA model package into the `zh-models` directory, and place the `params.json`  and the `consolidate.*.pth` model file obtained in the last step of [Model Conversion](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Manual-Conversion) into the `zh-models/7B` directory. Note that the `.pth` model file and `tokenizer.model` are corresponding, and the `tokenizer.model` for LLaMA and Alpaca should not be mixed. The directory structure should be similar to:

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

Run the `./main` binary file, with the `-m` command specifying the 4-bit quantized model (or loading the ggml-FP16 model). Below is an example of decoding parameters:

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

#### 7B

|                 |    F16 |   Q4_0 |   Q4_1 |   Q5_0 |   Q5_1 |   Q8_0 |
| --------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| PPL             | 10.793 | 12.416 | 12.002 | 11.155 | 10.905 | 10.790 |
| Size            | 13.77G |  4.31G |  5.17G |  4.74G |  5.17G |  7.75G |
| ms/tok @ `-t 2` |    144 |     87 |     88 |    143 |    157 |    103 |
| ms/tok @ `-t 4` |    123 |     50 |     52 |     75 |     82 |     72 |
| ms/tok @ `-t 8` |    126 |     41 |     49 |     46 |     49 |     69 |


#### 13B

|                 |   F16 |  Q4_0 |  Q4_1 |  Q5_0 |  Q5_1 |   Q8_0 |
| --------------- | ----: | ----: | ----: | ----: | ----: | -----: |
| PPL             | 9.147 | 9.917 | 9.689 | 9.325 | 9.344 |  9.147 |
| Size            | 26.4G | 8.25G |  9.9G | 9.08G |  9.9G | 14.85G |
| ms/tok @ `-t 2` |     - |   166 |   166 |   273 |   304 |    192 |
| ms/tok @ `-t 4` |     - |    89 |    94 |   142 |   155 |    132 |
| ms/tok @ `-t 8` |     - |    77 |    89 |    86 |    93 |    132 |