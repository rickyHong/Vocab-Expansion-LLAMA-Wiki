我们提供了命令行和Web图形界面两种方式使用原生Transformers进行推理。

以加载Chinese-Alpaca-7B模型为例（加载Chinese-Alpaca-Plus的方式见下面的**加载Chinese-Alpaca-Plus**）说明启动方式。

### 命令行交互形式
```bash
python scripts/inference_hf.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_or_alpaca_lora \
    --with_prompt \
    --interactive
```

如果之前已执行了`merge_llama_with_chinese_lora_to_hf.py`脚本将lora权重合并，那么无需再指定`--lora_model`，启动方式更简单：

```bash
python scripts/inference_hf.py \
    --base_model path_to_merged_llama_or_alpaca_hf_dir \
    --with_prompt \
    --interactive
```


参数说明：

* `--base_model {base_model} `：存放**HF格式**的LLaMA模型权重和配置文件的目录。如果之前合并生成的是PyTorch格式模型，[请转换为HF格式](./手动模型合并与转换)
* `--lora_model {lora_model}` ：中文LLaMA/Alpaca LoRA解压后文件所在目录，也可使用[🤗Model Hub模型调用名称](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main#model-hub)。若不提供此参数，则只加载`--base_model`指定的模型
* `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与`--lora_model`相同；若也未提供`--lora_model`参数，则其默认值与`--base_model`相同
* `--with_prompt`：是否将输入与prompt模版进行合并。**如果加载Alpaca模型，请务必启用此选项！**
* `--interactive`：以交互方式启动，以便进行多次**单轮问答**（此处不是llama.cpp中的上下文对话）
* `--data_file {file_name}`：非交互方式启动下，按行读取`file_name`中的的内容进行预测
* `--predictions_file {file_name}`：非交互式方式下，将预测的结果以json格式写入`file_name`
* `--use_cpu`: 仅使用CPU进行推理
* `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如`0,1,2`

### Web图形界面交互形式

该方式将启动Web前端页面进行交互，并且支持多轮对话。除transformers之外，需要安装gradio和mdtex2html：

```bash
pip install gradio
pip install mdtex2html
```

启动命令如下：

```
python scripts/gradio_demo.py \
	--base_model path_to_original_llama_hf_dir \
	--lora_model path_to_chinese_alpaca_lora
```

同样，如果已经执行了`merge_llama_with_chinese_lora_to_hf.py`脚本将lora权重合并，那么无需再指定`--lora_model`：

```
python scripts/gradio_demo.py --base_model path_to_merged_alpaca_hf_dir 
```

参数说明：

* `--base_model {base_model} `：存放**HF格式**的LLaMA模型权重和配置文件的目录。如果之前合并生成的是PyTorch格式模型，[请转换为HF格式](./手动模型合并与转换)
* `--lora_model {lora_model}` ：中文Alpaca LoRA解压后文件所在目录，也可使用[🤗Model Hub模型调用名称](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main#model-hub)。若不提供此参数，则只加载`--base_model`指定的模型
* `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与`--lora_model`相同；若也未提供`--lora_model`参数，则其默认值与`--base_model`相同
* `--use_cpu`: 仅使用CPU进行推理
* `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如`0,1,2`

### 加载Chinese-Alpaca-Plus

目前两个脚本都不支持直接从LoRA权重加载Chinese-Alpaca-Plus进行推理；如要进行Chinese-Alpaca-Plus进的推理，请先合并模型，流程如下：

1. 使用merge_llama_with_chinese_lora.py合并lora，生成完整的hf格式模型权重：
```bash
python scripts/merge_llama_with_chinese_lora.py \
    --base_model path_to_hf_llama \
    --lora_model path_to_chinese_llama_plus_lora,path_to_chinese_alpaca_plus_lora \
    --output_type huggingface \
    --output_dir path_to_merged_chinese_alpaca_plus
```
2. 使用inference_hf.py或gradio_demo.py加载合并后的模型进行推理，如：
```bash
python scripts/inference_hf.py \
    --base_model path_to_merged_chinese_alpaca_plus \
    --with_prompt --interactive
```

### 注意事项

- 因不同框架的解码实现细节有差异，该脚本并不能保证复现llama.cpp的解码效果
- 该脚本仅为方便快速体验用，并未对推理速度做优化
- 如在CPU上运行7B模型推理，请确保有32GB内存；如在GPU上运行7B模型推理，请确保有20GB显存
