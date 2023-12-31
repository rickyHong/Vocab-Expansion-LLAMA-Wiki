以[llama.cpp工具](https://github.com/ggerganov/llama.cpp)为例，介绍模型量化并在**本地CPU上部署**的详细步骤。Windows则可能需要cmake等编译工具的安装（Windows用户出现模型无法理解中文或生成速度特别慢时请参考[FAQ#6](./常见问题#问题6windows下模型无法理解中文生成速度很慢等问题)）。**本地快速部署体验推荐使用经过指令精调的Alpaca模型，有条件的推荐使用8-bit模型，效果更佳。** 下面以中文Alpaca-7B模型为例介绍，运行前请确保：

1. 系统应有`make`（MacOS/Linux自带）或`cmake`（Windows需自行安装）编译工具
2. 建议使用Python 3.10以上编译和运行该工具
3. 最新版llama.cpp添加了对GPU的支持，感兴趣的可以参考[https://github.com/ggerganov/llama.cpp/discussions/915](https://github.com/ggerganov/llama.cpp/discussions/915)


### Step 1: 克隆和编译llama.cpp

1. 克隆最新版llama.cpp仓库代码

```
git clone https://github.com/ggerganov/llama.cpp
```

2. （可选）如需使用`qX_k` 量化方法（相比常规量化方法效果更好），请手动打开`llama.cpp`文件，修改下列行（约2500行左右）：

- 原始代码：`if (nx % QK_K != 0 || ny % QK_K != 0) {` 

- 改为：`if (nx % QK_K != 0) {`

3. 对llama.cpp项目进行编译，生成`./main`和`./quantize`二进制文件。

```bash
make
```

- Windows/Linux用户：**推荐与[BLAS（或cuBLAS如果有GPU）一起编译](https://github.com/ggerganov/llama.cpp#blas-build)**，可以提高prompt处理速度，参考：[llama.cpp#blas-build](https://github.com/ggerganov/llama.cpp#blas-build)

- macOS用户：无需额外操作，llama.cpp已对ARM NEON做优化，并且已自动启用BLAS。
  - **M系列芯片推荐**：使用Metal启用GPU推理，显著提升速度。只需将编译命令改为：`LLAMA_METAL=1 make`，参考[llama.cpp#metal-build](https://github.com/ggerganov/llama.cpp#metal-build)


###  Step 2: 生成量化版本模型

**提示：最新版llama.cpp已支持直接转换HF版本模型（教程可参考[二代模型wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llamacpp_zh)）。下面的教程以`.pth`格式为例进行介绍。**

将合并模型（`.pth`格式模型）中最后一步生成的`tokenizer.model`文件放入`zh-models`目录下，模型文件`consolidated.*.pth`和配置文件`params.json`放入`zh-models/7B`目录下。请注意LLaMA和Alpaca的`tokenizer.model`不可混用（原因见[训练细节](./训练细节)）。例如，如果是`.pth`格式的模型，目录结构类似：

```
llama.cpp/zh-models/
   - 7B/
     - consolidated.00.pth
     - params.json
   - tokenizer.model
```

将上述`.pth`模型权重转换为ggml的FP16格式，生成文件路径为`zh-models/7B/ggml-model-f16.bin`。

```bash
python convert.py zh-models/7B/
```

进一步对FP16模型进行4-bit量化，生成量化模型文件路径为`zh-models/7B/ggml-model-q4_0.bin`。不同量化方法的性能对比见本文最后。

```bash
./quantize ./zh-models/7B/ggml-model-f16.bin ./zh-models/7B/ggml-model-q4_0.bin q4_0
```

### Step 3: 加载并启动模型

运行`./main`二进制文件，`-m`命令指定GGML格式模型。以下是命令示例（并非最优参数）。

**如已通过Metal编译，则只需加上`-ngl 1`即可启用GPU推理。**

```bash
./main -m zh-models/7B/ggml-model-q4_0.bin --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.1
```
在提示符 `>` 之后输入你的prompt，`cmd/ctrl+c`中断输出，多行信息以`\`作为行尾。如需查看帮助和参数说明，请执行`./main -h`命令。下面介绍一些常用的参数：

```
-c 控制上下文的长度，值越大越能参考更长的对话历史（默认：512）
-ins 启动类ChatGPT对话交流的instruction运行模式
-f 指定prompt模板，alpaca模型请加载prompts/alpaca.txt
-n 控制回复生成的最大长度（默认：128）
-b 控制batch size（默认：8），可适当增加
-t 控制线程数量（默认：4），可适当增加
--repeat_penalty 控制生成回复中对重复文本的惩罚力度
--temp 温度系数，值越低回复的随机性越小，反之越大
--top_p, top_k 控制解码采样的相关参数
```

更详细的官方说明请参考：[https://github.com/ggerganov/llama.cpp/tree/master/examples/main](https://github.com/ggerganov/llama.cpp/tree/master/examples/main)


### 关于量化方法选择及推理速度

下表给出了不同量化方法的相关统计数据供参考。推理模型为中文Alpaca-Plus-7B、Alpaca-Plus-13B，测试设备为M1 Max芯片（8x性能核心，2x能效核心），分别汇报CPU速度（8线程）和GPU速度，单位为ms/tok。速度方面报告的是`eval time`，即模型回复生成的速度。更多关于量化参数的介绍可参考[llama.cpp量化统计表](https://github.com/ggerganov/llama.cpp#quantization)。

相关结论：

- 默认的量化方法为q4_0，虽然速度最快但损失也是最大的，其余方法各有利弊，按实际情况选择
- 需要注意的是F16以及q8_0并不会因为增加线程数而提高太多速度
- 线程数`-t`与物理核心数一致时速度最快，超过之后速度反而变慢（M1 Max上从8改到10之后耗时变为3倍）
- 如果使用了Metal版本（即启用了苹果GPU解码），速度还会有进一步显著提升，表中标注为`-ngl 1`
- 综合推荐（仅供参考）：7B推荐Q5_1或Q5_K_S，13B推荐Q5_0或Q5_K_S
- 机器资源够用且对速度要求不是那么苛刻的情况下可以使用q8_0或Q6_K，接近F16模型的效果

#### Alpaca-Plus-7B

|           |    F16 |   Q2_K | Q3_K_S | Q3_K_M<br/>(Q3_K) | Q3_K_L |   Q4_0 |   Q4_1 | Q4_K_S | Q4_K_M<br/>(Q4_K) |   Q5_0 |   Q5_1 | Q5_K_S | Q5_K_M<br/>(Q5_K) |   Q6_K |   Q8_0 |
| --------- | -----: | -----: | -----: | ----------------: | -----: | -----: | -----: | -----: | ----------------: | -----: | -----: | -----: | ----------------: | -----: | -----: |
| PPL       | 10.793 | 18.292 | 15.276 |            12.504 | 11.548 | 12.416 | 12.002 | 11.717 |            11.062 | 11.155 | 10.905 | 10.930 |            10.869 | 10.845 | 10.790 |
| Size      | 13.77G |  2.95G |  3.04G |             3.37G |  3.69G |  4.31G |  5.17G |  3.93G |             4.18G |  4.74G |  5.17G |  4.76G |             4.89G |  5.65G |  7.75G |
| CPU Speed |    126 |     48 |     57 |                52 |     54 |     41 |     49 |     45 |                47 |     46 |     49 |     52 |                54 |     58 |     69 |
| GPU Speed |     56 |     28 |     32 |                32 |     33 |     28 |     26 |     32 |                30 |      x |      x |     32 |                32 |     33 |      x |

#### Alpaca-Plus-13B

|           |   F16 |   Q2_K | Q3_K_S | Q3_K_M<br/>(Q3_K) | Q3_K_L |  Q4_0 |  Q4_1 | Q4_K_S | Q4_K_M<br/>(Q4_K) |  Q5_0 |  Q5_1 | Q5_K_S | Q5_K_M<br/>(Q5_K) |   Q6_K |   Q8_0 |
| --------- | ----: | -----: | -----: | ----------------: | -----: | ----: | ----: | -----: | ----------------: | ----: | ----: | -----: | ----------------: | -----: | -----: |
| PPL       | 9.147 | 15.455 | 11.488 |            10.229 | 9.5372 | 9.917 | 9.689 |  9.947 |             9.295 | 9.325 | 9.344 |  9.286 |             9.246 |  9.169 |  9.147 |
| Size      | 26.4G |  5.61G |  5.77G |             6.43G |  7.04G | 8.25G |  9.9G |  7.49G |             7.99G | 9.08G |  9.9G |  9.11G |             9.37G | 10.83G | 14.85G |
| CPU Speed |       |     83 |     99 |                94 |     99 |    77 |    89 |     77 |                81 |    86 |    93 |     93 |                93 |    104 |    132 |
| GPU Speed |     x |     52 |     56 |                57 |     59 |    49 |     x |     58 |                55 |     x |     x |     57 |                57 |     59 |      x |

#### Alpaca-Plus-33B

|                 |    F16 |  Q2_K  | Q3_K_S | Q3_K_M<br/>(Q3_K) | Q3_K_L | Q4_0   | Q4_1   | Q4_K_S | Q4_K_M<br/>(Q4_K) | Q5_0   | Q5_1   | Q5_K_S | Q5_K_M<br/>(Q5_K) | Q6_K   |   Q8_0 |
| :-------------- | -----: | ----: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | -----: | -----: | -----: | -----: | -----: | -----: |
| PPL             |  8.120 | 11.655 | 9.417 | 9.217 | 8.980 |  8.217 |  8.760 | 8.602 | 8.283 | 8.152 | 8.251 | 8.312 | 8.210 | 8.157 | 8.119 |
| Size            | 61.03G | 12.74G | 14.21G | 14.65G | 16.11G | 17.16G | 19.07G | 17.16G | 18.43G | 20.98G | 24.58G | 20.98G | 21.59G | 25.03G | 32.42G |
| CPU Speed |      - |  174  | 238 | 242    | 258 | 170    | 185    | 178 | 194 | 224    | 306    |        |        |        |      - |
| GPU Speed | - | 127 | 130 | 128 | 132 | 120 | x | 127 | 181 | x | x | x |  | x | x |

#### Alpaca-65B (n/a)

|           |     F16 |   Q2_K | Q3_K_S | Q3_K_M<br/>(Q3_K) | Q3_K_L | Q4_0 | Q4_1 | Q4_K_S | Q4_K_M<br/>(Q4_K) | Q5_0 | Q5_1 | Q5_K_S | Q5_K_M<br/>(Q5_K) | Q6_K | Q8_0 |
| :-------- | ------: | -----: | -----: | ----------------: | -----: | ---: | ---: | -----: | ----------------: | ---: | ---: | -----: | ----------------: | ---: | ---: |
| PPL       |         |        |        |                   |        |      |      |        |                   |      |      |        |                   |      |      |
| Size      | 121.61G | 25.56G |        |                   |        |      |      |        |                   |      |      |        |                   |      |      |
| CPU Speed |       - |        |        |                   |        |      |      |        |                   |      |      |        |                   |      |      |
| GPU Speed |       - |        |        |                   |        |      |      |        |                   |      |      |        |                   |      |      |
