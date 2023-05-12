Next, we will use the [text-generation-webui tool](https://github.com/oobabooga/text-generation-webui) as an example to introduce the detailed steps for local deployment without the need for model merging.

#### Step 1: Clone the text-generation-webui and install necessary dependencies
```
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

#### Step 2: put the downloaded lora weights into the loras folder.
```
ls loras/chinese-alpaca-lora-7b
adapter_config.json  adapter_model.bin  special_tokens_map.json  tokenizer_config.json  tokenizer.model
```

#### Step 3: put the HuggingFace-formatted llama-7B model files into the models  folder.
```
ls models/llama-7b-hf
pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin config.json pytorch_model.bin.index.json generation_config.json
```

#### Step 4: copy the tokenizer of lora weights to the models/llama-7b-hf directory
```
cp loras/chinese-alpaca-lora-7b/tokenizer.model models/llama-7b-hf/
cp loras/chinese-alpaca-lora-7b/special_tokens_map.json models/llama-7b-hf/
cp loras/chinese-alpaca-lora-7b/tokenizer_config.json models/llama-7b-hf/
```

#### Step 5: modify /modules/LoRA.py file
```
shared.model.resize_token_embeddings(len(shared.tokenizer))
shared.model = PeftModel.from_pretrained(shared.model, Path(f"{shared.args.lora_dir}/{lora_name}"), **params)
```

#### Step 6: Great! You can now run the tool. Please refer to [webui using LoRAs](https://github.com/oobabooga/text-generation-webui/blob/main/docs/Using-LoRAs.md)for instructions on how to use LoRAs
```
python server.py --model llama-7b-hf --lora chinese-alpaca-lora-7b
```
