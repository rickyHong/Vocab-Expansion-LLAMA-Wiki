### Preparation

1. Pull the latest code of our repository first: `git pull`

2. Make sure the machine has enough memory to load the complete model (e.g., 13-15G for the 7B model) for the model merging operation.

3. Before merging, make sure that the SHA256 of the base model and the LoRA model patch files are consistent with those in [SHA256.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/SHA256.md), otherwise, the merge operation cannot be performed. The original LLaMA contains the following files: `tokenizer.model`, `tokenizer_checklist.chk`, `consolidated.00.pth`, `params.json`

4. Dependencies (python>=3.9):

```bash
pip install torch==1.13.1
pip install transformers==4.28.1
pip install sentencepiece==0.1.97
pip install peft==0.3.0
```

*This project is not responsible for the compliance and correctness of using third-party (non-Facebook official) weights, such as the `elinas/llama-7b-hf-transformers-4.29` in the HuggingFace model library (use at your own risk).*

### Step 1: Convert the original LLaMA model to HF format

Use the script [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) provided by the [latest 🤗transformers](https://huggingface.co/docs/transformers/installation#install-from-source) to convert the original LLaMA model to HuggingFace format. 

⚠️ Please put the original LLaMA's `tokenizer.model` file in`--input_dir`, and the other files in `${input_dir}/${model_size}`.

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir path_to_original_llama_root_dir \
    --model_size 7B \
    --output_dir path_to_original_llama_hf_dir
```

`--output_dir` directory will contain HF format model files. For e.g.,

```
config.json
generation_config.json
pytorch_model-00001-of-00002.bin
pytorch_model-00002-of-00002.bin
pytorch_model.bin.index.json
special_tokens_map.json
tokenizer_config.json
tokenizer.json
tokenizer.model
```

### Step 2: Merge LoRA weights to generate full model weights

**[New]** **Please use our new script, which significantly lowers memory usage. Just replace the script with `scripts/merge_llama_with_chinese_lora_low_mem.py`.**

This step will expand the Chinese vocabulary of the original LLaMA model (HF format), merge LoRA weights, and generate full model weights. There are two options available here:

- Generate  `pth` model file for quantization and deployment: [using llama.cpp for quantization and deployment](./llama.cpp-Deployment)
- Generate HuggingFace model file（`bin` file) for training and inference: [inference with transformers](./Inference-with-Transformers), [deployment with text-generation-webui](./text-generation-webui)

Note that the merging steps of different models are different. Please read the following guide and follow the steps strictly.

#### Single LoRA weight merging (applicable to Chinese-LLaMA, Chinese-LLaMA-Plus, and Chinese-Alpaca)

Execute the following command:

```bash
python scripts/merge_llama_with_chinese_lora.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_or_alpaca_lora \
    --output_type [pth|huggingface] \
    --output_dir path_to_output_dir 
```
where:

- `--base_model`: directory where the HF format LLaMA model weights and configuration files are saved (generated in Step 1)
- `--lora_model`: directory where the Chinese LLaMA/Alpaca LoRA model compressed file downloaded in the previous section is located, or the model name on Hugging Face Model Hub: `ziqingyang/chinese-alpaca-lora-7b` or `ziqingyang/chinese-llama-lora-7b`
- `--output_type`: the saving format, either `pth` or `huggingface`. Default: `pth`.
- `--output_dir`: directory to save the consolidated model weights (default: `./`)
- (optional) `--offload_dir` (only applicable to the old script `scripts/merge_llama_with_chinese_lora.py`): for low-RAM users, please specify a offload directory
- (optional) `--verbose` (only applicable to the new script `scripts/merge_llama_with_chinese_lora_low_mem.py`): show detailed messages of the merging process


#### Multiple LoRA weights merging (applicable to Chinese-Alpaca-Plus)

Merging Chinese-Alpaca-Plus requires two LoRA weights, namely Chinese-LLaMA-Plus-LoRA and Chinese-Alpaca-Plus-LoRA. To complete the merge, execute the following command:

```bash
python scripts/merge_llama_with_chinese_lora.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_plus_lora,path_to_chinese_alpaca_plus_lora \
    --output_type [pth|huggingface] \
    --output_dir path_to_output_dir 
```

The meaning of the each options is the same as those in Single LoRA weight merging. Note that the `--lora_model` requires **two** LoRA models, separated by a comma. ⚠️ **The order of the two LoRA models is important and cannot be reversed, LLaMA-Plus-LoRA first then Alpaca-Plus-LoRA.** 

### Step 3: After Check

**Check SHA256 after merging! Check SHA256 after merging! Check SHA256 after merging!**

SHA256：https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/SHA256.md
