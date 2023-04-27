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

Further quantize the FP16 model to 4-bit, and generate a quantized model file with the path `zh-models/7B/ggml-model-q4_0.bin`.

```bash
./quantize ./zh-models/7B/ggml-model-f16.bin ./zh-models/7B/ggml-model-q4_0.bin 2
```

#### About quantization parameters
Here, we use the default `-t` param (default value: 4), using Chinese Alpaca-7B with Apple M1 Max. More details: [llama.cpp#PPL](https://github.com/ggerganov/llama.cpp#perplexity-measuring-model-quality)。
| Param | Algorithm | Speed | Model Size | PPL | Note |
|---|---|---|---|---|---|
| 2 | q4_0 | 57ms/token | 4.31G | 25.7 | default |
| 3 | q4_1 | 102ms/token | 5.17G | 24.5 | - |
| 5（ARM only）| q4_2 | 85ms/token | 4.31G | 24.8 |  experimental, under dev  |
| 6 | q4_3 | 156ms/token | 5.17G | 22.9 | experimental, under dev  |
| 8 | q5_0 | 86ms/token | 4.74G | 22.4 | experimental, under dev |
| 7 | q8_0 | 74ms/token | 7.75G | 21.8 | experimental, under dev |
| - | f16 | 88ms/token | 13.77G | 21.8 | no quantization |

Note: we recommend use q8_0 version, which has a good tradeoff.

### Step 3: Load and start the model

Run the `./main` binary file, with the `-m` command specifying the 4-bit quantized model (or loading the ggml-FP16 model). Below is an example of decoding parameters:

```bash
./main -m zh-models/7B/ggml-model-q4_0.bin --color -f ./prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
```

Please enter your prompt after the `>`, use `\` as the end of the line for multi-line inputs. To view help and parameter instructions, please execute the `./main -h` command. Here's a brief introduction to several important parameters:

```
-c controls the length of context, larger values allow for longer dialogue history to be referenced
-ins activates the conversation mode for the ChatGPT class
-n controls the maximum length of generated responses
-b batch size
-t number of threads
--repeat_penalty controls the penalty for repeated text in the generated response
--temp is the temperature coefficient, lower values result in less randomness in the response, and vice versa
--top_p, top_k control the sampling parameters
```

#### About prediction speed
It is not always faster with bigger `-t`.
The following table shows the speed with different `-t` (M1 Max chips with 8 performance cores and 2 efficiency cores).
It can be seen that the speed is the fastest when it is consistent with the number of cores, but it slows down when it exceeds this value.
| param | Speed（7B-q4_0） | Speed（13B-q4_0） |
|---|---|---|
| 1 | 230ms/token | 434ms/token |
| 2 | 110ms/token | 208ms/token |
| 4 | 58ms/token | 111ms/token |
| 6 | 44ms/token | 80ms/token |
| 8 | **36ms/token** | **64ms/token** |
| 10 | *112ms/token* | *202ms/token* | 