接下来以[text-generation-webui工具](https://github.com/oobabooga/text-generation-webui)为例，介绍无需合并模型即可进行**本地化部署**的详细步骤。

### Step 1: 克隆text-generation-webui
运行以下命令克隆text-generation-webui并按要求安装必要的依赖
```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

### Step 2: 准备模型权重
将下载后的lora权重及HuggingFace格式的llama-7B模型权重分别放到loras、models文件夹下，目录文件如下所示
```bash
ls loras/chinese-alpaca-lora-7b
adapter_config.json  adapter_model.bin  special_tokens_map.json  tokenizer_config.json  tokenizer.model
ls models/llama-7b-hf
pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin config.json pytorch_model.bin.index.json generation_config.json
```
然后复制lora权重的tokenizer到`models/llama-7b-hf`下并修改`/modules/LoRA.py`文件(webui默认从`./models`下加载tokenizer.model,因此需使用扩展中文词表后的tokenizer.model)
```bash
cp loras/chinese-alpaca-lora-7b/tokenizer.model models/llama-7b-hf/
cp loras/chinese-alpaca-lora-7b/special_tokens_map.json models/llama-7b-hf/
cp loras/chinese-alpaca-lora-7b/tokenizer_config.json models/llama-7b-hf/
```
在`PeftModel.from_pretrained`方法之前添加一行代码修改原始llama的embed_size
```bash
shared.model.resize_token_embeddings(len(shared.tokenizer))
shared.model = PeftModel.from_pretrained(shared.model, Path(f"{shared.args.lora_dir}/{lora_name}"), **params)  # 该行源代码中就有，无需改动
```
### Step 3: 加载模型并启动webui
运行以下命令即可与chinese-llama/alpaca对话了。
```bash
python server.py --model llama-7b-hf --lora chinese-alpaca-lora-7b --cpu
```
更详细的官方说明请参考：[webui using LoRAs](https://github.com/oobabooga/text-generation-webui/blob/main/docs/Using-LoRAs.md)。此外，我们推荐直接运行合并后的chinese-alpaca-7b，相对加载两个权重推理速度会有较大的提升。