### Preparation

1. Make sure the machine has enough memory to load the complete model (e.g., 13-15G for the 7B model) for the model merging operation.

2. Before merging, make sure that the SHA256 of the base model and the LoRA model patch files are consistent with those in [SHA256.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/SHA256.md), otherwise, the merge operation cannot be performed. The original LLaMA contains the following files: `tokenizer.model`, `tokenizer_checklist.chk`, `consolidated.00.pth`, `params.json`

3. Dependencies:
   - `torch` (1.10, 1.12, 1.13)
   - `transformers`（4.28.1）
   - `sentencepiece`（0.1.97）
   - `peft`（0.3.0dev）
   - python >= 3.9

```bash
pip install torch==1.13.1
pip install transformers
pip install sentencepiece
pip install git+https://github.com/huggingface/peft
```

*This project is not responsible for the compliance and correctness of using third-party (non-Facebook official) weights, such as the `decapoda-research/llama-7b-hf` in the HuggingFace model library (use at your own risk).*

### Step 1: Convert the original LLaMA model to HF format

Use the script [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) provided by the [latest 🤗transformers](https://huggingface.co/docs/transformers/installation#install-from-source) to convert the original LLaMA model to HuggingFace format. 

⚠️ Please put the original LLaMA's `tokenizer.model` file in`--input_dir`, and the other files in `${input_dir}/${model_size}`.

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir path_to_original_llama_root_dir \
    --model_size 7B \
    --output_dir path_to_original_llama_hf_dir
```

### Step 2: Merge LoRA weights to generate full model weights

This step will expand the Chinese vocabulary of the original LLaMA model (HF format), merge LoRA weights, and generate full model weights. There are two options available here

1. generate a `pth` model file for quantization and deployment
- [using llama.cpp for quantization and deployment](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/llama.cpp-Deployment)

2. generate a HuggingFace model file（`bin` file) for simple inference. 
- [inference with transformers](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Inference-with-Transformers)
- [deployment with text-generation-webui](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/text-generation-webui)

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
- (optional) `--offload_dir`: for low-RAM users, please specify a offload directory


#### Multiple LoRA weights merging (applicable to Chinese-Alpaca-Plus)

Merging Chinese-Alpaca-Plus requires two LoRA weights, namely Chinese-LLaMA-Plus-LoRA and Chinese-Alpaca-Plus-LoRA. To complete the merge, execute the following command:

```bash
python scripts/merge_llama_with_chinese_lora.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_plus_lora,path_to_chinese_alpaca_plus_lora \
    --output_type [pth|huggingface] \
    --output_dir path_to_output_dir 
```

The meaning of the each options is the same as those in Single LoRA weight merging.
Note that the `--lora_model` requires **two** LoRA models, separated by a comma. 

⚠️ **The order of the two LoRA models is important and cannot be reversed, LLaMA-Plus-LoRA first then Alpaca-Plus-LoRA.** ⚠️

