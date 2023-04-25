接下来以[llama.cpp工具](https://github.com/ggerganov/llama.cpp)为例，介绍MacOS和Linux系统中，将模型进行量化并在**本地CPU上部署**的详细步骤。Windows则可能需要cmake等编译工具的安装（Windows用户出现模型无法理解中文或生成速度特别慢时请参考[FAQ#6](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/常见问题#问题6windows下模型无法理解中文生成速度很慢等问题)）。**本地快速部署体验推荐使用经过指令精调的Alpaca模型，有条件的推荐使用FP16模型，效果更佳。** 下面以中文Alpaca-7B模型为例介绍，运行前请确保：

1. 模型量化过程需要将未量化模型全部载入内存，请确保有足够可用内存（7B版本需要13G以上）
2. 加载使用4-bit量化后的模型时（例如7B版本），确保本机可用内存大于4-6G（受上下文长度影响）
3. 系统应有`make`（MacOS/Linux自带）或`cmake`（Windows需自行安装）编译工具
4. [llama.cpp](https://github.com/ggerganov/llama.cpp)官方建议使用Python 3.9~3.11编译和运行该工具
5. 最新版llama.cpp添加了对GPU的支持，感兴趣的可以参考[https://github.com/ggerganov/llama.cpp/discussions/915](https://github.com/ggerganov/llama.cpp/discussions/915)


### Step 1: 克隆和编译llama.cpp

运行以下命令对llama.cpp项目进行编译，生成`./main`和`./quantize`二进制文件。

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
```

###  Step 2: 生成量化版本模型

将合并模型（选择生成`.pth`格式模型）中最后一步生成的`tokenizer.model`文件放入`zh-models`目录下，模型文件`consolidated.*.pth`和配置文件`params.json`放入`zh-models/7B`目录下。请注意LLaMA和Alpaca的`tokenizer.model`不可混用（原因见[训练细节](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/训练细节)）。目录结构类似：

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

进一步对FP16模型进行4-bit量化，生成量化模型文件路径为`zh-models/7B/ggml-model-q4_0.bin`。

```bash
./quantize ./zh-models/7B/ggml-model-f16.bin ./zh-models/7B/ggml-model-q4_0.bin 2
```

#### 关于量化参数（上述命令中的最后一个参数）
测试中使用了默认`-t`参数（默认值：4），推理模型为中文Alpaca-7B，测试环境M1 Max。测试命令更多关于量化参数可参考[llama.cpp#PPL](https://github.com/ggerganov/llama.cpp#perplexity-measuring-model-quality)。

| 参数 | 对应量化算法 | 推理速度 | 模型大小 | 小样本数据PPL | 备注 | 
|---|---|---|---|---|---|
| 2 | q4_0 | 57ms/token | 4.31G | 25.7 | 默认 |
| 3 | q4_1 | 102ms/token | 5.17G | 24.5 | - |
| 5（ARM only）| q4_2 | 85ms/token | 4.31G | 24.8 |  实验性，需要等稳定版本 |
| 6 | q4_3 | 156ms/token | 5.17G | 22.9 | 实验性，需要等稳定版本 |
| - | f16 | 88ms/token | 13.77G | 21.8 | 非量化版本 |

### Step 3: 加载并启动模型

运行`./main`二进制文件，`-m`命令指定4-bit量化或FP16的GGML模型。以下是命令示例（并非最优参数）：

```bash
./main -m zh-models/7B/ggml-model-q4_0.bin --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
```
在提示符 `>` 之后输入你的prompt，`cmd/ctrl+c`中断输出，多行信息以`\`作为行尾。如需查看帮助和参数说明，请执行`./main -h`命令。下面介绍一些常用的参数：

```
-ins 启动类ChatGPT对话交流的运行模式
-f 指定prompt模板，alpaca模型请加载prompts/alpaca.txt
-c 控制上下文的长度，值越大越能参考更长的对话历史（默认：512）
-n 控制回复生成的最大长度（默认：128）
-b 控制batch size（默认：8），可适当增加
-t 控制线程数量（默认：4），可适当增加
--repeat_penalty 控制生成回复中对重复文本的惩罚力度
--temp 温度系数，值越低回复的随机性越小，反之越大
--top_p, top_k 控制解码采样的相关参数
```

#### 关于量化模型预测速度
关于速度方面，`-t`参数并不是越大越好，要根据自己的处理器进行适配。
下表给出了M1 Max芯片（8大核2小核）的推理速度对比。
可以看到，与核心数一致的时候速度最快，超过这个数值之后速度反而变慢。
| 参数 | 推理速度（7B-q4_0） | 推理速度（13B-q4_0） |
|---|---|---|
| 1 | 230ms/token | 434ms/token |
| 2 | 110ms/token | 208ms/token |
| 4 | 58ms/token | 111ms/token |
| 6 | 44ms/token | 80ms/token |
| 8 | **36ms/token** | **64ms/token** |
| 10 | *112ms/token* | *202ms/token* | 