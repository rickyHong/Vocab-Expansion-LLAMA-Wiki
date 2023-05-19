Next, we will use the [text-generation-webui tool](https://github.com/oobabooga/text-generation-webui) as an example to introduce the detailed steps for local deployment without the need for model merging.

### Step 1: Clone text-generation-webui
Run the following command to clone text-generation-webui and install the necessary dependencies as required
```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

### Step 2: Preparing model weights
Put the downloaded lora weights and the HuggingFace format llama-7B model weights into the loras and models folders, respectively, as shown below
```
ls loras/chinese-alpaca-lora-7b
adapter_config.json  adapter_model.bin  special_tokens_map.json  tokenizer_config.json  tokenizer.model
ls models/llama-7b-hf
pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin config.json pytorch_model.bin.index.json generation_config.json
```
Copy the tokenizer of lora weights to the models/llama-7b-hf directory and modify `/modules/LoRA.py`
```
cp loras/chinese-alpaca-lora-7b/tokenizer.model models/llama-7b-hf/
cp loras/chinese-alpaca-lora-7b/special_tokens_map.json models/llama-7b-hf/
cp loras/chinese-alpaca-lora-7b/tokenizer_config.json models/llama-7b-hf/
```
Modifying `/modules/LoRA.py` is as simple as adding a line before the `PeftModel.from_pretrained` method
```
shared.model.resize_token_embeddings(len(shared.tokenizer))
shared.model = PeftModel.from_pretrained(shared.model, Path(f"{shared.args.lora_dir}/{lora_name}"), **params)
```

### Step 3:Load the model and start webui
Run the following command to talk to chinese-llama/alpaca
```
python server.py --model llama-7b-hf --lora chinese-alpaca-lora-7b --cpu
```
Please refer to [webui using LoRAs](https://github.com/oobabooga/text-generation-webui/blob/main/docs/Using-LoRAs.md) for instructions on how to use LoRAs.In addition, we recommend directly running the merged chinese-alpaca-7b, which will greatly improve the inference speed compared with loading two weights.
### loading Chinese-Alpaca-Plus

If you want to apply Chinese-Alpaca-Plus, please follow the steps below:

1. Using [merge_llama_with_chinese_lora.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_llama_with_chinese_lora.py)to obtaining a single model weight file in HF format:
```bash
python scripts/merge_llama_with_chinese_lora.py \
    --base_model path_to_hf_llama \
    --lora_model path_to_chinese_llama_plus_lora,path_to_chinese_alpaca_plus_lora \
    --output_type huggingface \
    --output_dir path_to_webui/models/merged_chinese_alpaca_plus
```
2. Run the following command to talk to chinese-llama/alpaca plus
```bash
python server.py --model merged_chinese_alpaca_plus --cpu
```
