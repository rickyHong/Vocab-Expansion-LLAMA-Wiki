如果想在不安装其他库或Python包的情况下快速体验模型效果，可以使用[scripts/inference_hf.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/inference_hf.py) 脚本启动非量化模型。该脚本支持CPU和GPU的单卡推理。以启动Chinese-Alpaca-7B模型为例，脚本运行方式如下（该方式无法加载Chinese-Alpaca-Plus，见**注意事项）：

```bash
CUDA_VISIBLE_DEVICES={device_id} python scripts/inference_hf.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_or_alpaca_lora \
    --with_prompt \
    --interactive
```


如果已经执行了`merge_llama_with_chinese_lora_to_hf.py`脚本将lora权重合并，那么无需再指定`--lora_model`，启动方式更简单：

```bash
CUDA_VISIBLE_DEVICES={device_id} python scripts/inference_hf.py \
    --base_model path_to_merged_llama_or_alpaca_hf_dir \
    --with_prompt \
    --interactive
```

参数说明：

* `{device_id}`：CUDA设备编号。如果为空，那么在CPU上进行推理
* `--base_model {base_model} `：存放**HF格式**的LLaMA模型权重和配置文件的目录。如果之前合并生成的是PyTorch格式模型，[请转换为HF格式](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/手动模型合并与转换#step-2-合并lora权重生成全量模型权重)
* `--lora_model {lora_model}` ：中文LLaMA/Alpaca LoRA解压后文件所在目录，也可使用[🤗Model Hub模型调用名称](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main#model-hub)。若不提供此参数，则只加载`--base_model`指定的模型
* `--tokenizer_path {tokenizer_path}`：存放对应tokenizer的目录。若不提供此参数，则其默认值与`--lora_model`相同；若也未提供`--lora_model`参数，则其默认值与`--base_model`相同
* `--with_prompt`：是否将输入与prompt模版进行合并。**如果加载Alpaca模型，请务必启用此选项！**
* `--interactive`：以交互方式启动，以便进行多次**单轮问答**（此处不是llama.cpp中的上下文对话）
* `--data_file {file_name}`：非交互方式启动下，按行读取`file_name`中的的内容进行预测
* `--predictions_file {file_name}`：非交互式方式下，将预测的结果以json格式写入`file_name`

注意事项：

- 因不同框架的解码实现细节有差异，该脚本并不能保证复现llama.cpp的解码效果
- 该脚本仅为方便快速体验用，并未对多机多卡、低内存、低显存等情况等条件做任何优化
- 如在CPU上运行7B模型推理，请确保有32GB内存；如在GPU上运行7B模型推理，请确保有20GB显存
- inference_hf.py暂不支持从lora权重加载Chinese-Alpaca-Plus进行推理；如要进行Chinese-Alpaca-Plus进的推理，建议先合并模型，流程如下：
1. 使用merge_llama_with_chinese_lora.py合并lora，生成完整的hf格式模型权重：
```bash
python merge_llama_with_chinese_lora.py \
    --base_model path_to_hf_llama \
    --lora_model path_to_chinese_llama_plus_lora,path_to_chinese_alpaca_plus_lora \
    --output_type huggingface \
    --output_dir path_to_merged_chinese_alpaca_plus
```
2. 使用inference_hf.py加载合并后的模型进行推理：
```bash
python inference_hf.py \
    --base_model path_to_merged_chinese_alpaca_plus \
    --with_prompt --interactive
```
