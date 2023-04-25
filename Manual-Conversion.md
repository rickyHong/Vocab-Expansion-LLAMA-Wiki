### Preparation

1. Make sure the machine has enough memory to load the complete model (e.g., 13-15G for the 7B model) for the model merging operation.

2. Before merging, make sure that the SHA256 of the base model and the LoRA model patch files are consistent with those in [SHA256.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/SHA256.md), otherwise, the merge operation cannot be performed.

   - The original LLaMA contains the following files: `tokenizer.model`, `tokenizer_checklist.chk`, `consolidated.00.pth`, `params.json`

   - The SHA256 of the weight file `consolidated.00.pth`: `700df0d3013b703a806d2ae7f1bfb8e59814e3d06ae78be0c66368a50059f33d`

3. Dependencies:
   - ⚠️ **You MUST use the [latest 🤗Transformers library](https://huggingface.co/docs/transformers/installation#install-from-source)**. The current release v4.27 does not support LLaMA. 
   - install `sentencepiece` and `peft` using `pip` command


 ```bash
 pip install git+https://github.com/huggingface/transformers
 pip install sentencepiece
 pip install peft
 ```

### Step 1: Convert the original LLaMA model to HF format

Use the script [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) provided by the [latest 🤗transformers](https://huggingface.co/docs/transformers/installation#install-from-source) to convert the original LLaMA model to HuggingFace format. *This project is not responsible for the compliance and correctness of using third-party (non-Facebook official) weights, such as the `decapoda-research/llama-7b-hf` in the HuggingFace model library (use at your own risk).*

⚠️ Please put the original LLaMA's `tokenizer.model` file in`--input_dir`, and the other files in `${input_dir}/${model_size}`.

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir path_to_original_llama_root_dir \
    --model_size 7B \
    --output_dir path_to_original_llama_hf_dir
```

### Step 2: Merge LoRA weights to generate full model weights

This step will expand the Chinese vocabulary of the original LLaMA model (HF format), merge LoRA weights, and generate full model weights. There are two options available here:

- ✅ If you need quantize and deploy our model: output the weight of PyTorch version (`. pth` file) using `scripts/merge_llama_with_chinese_lora.py` script
- ❎ If you DO NOT need quantize and deploy our model: output the weight of the HuggingFace version (such as for further fine-tuning), using `scripts/merge_llama_with_chinese_lora_to_hf.py` script (thanks @sgsdxzy)

The parameters that need to be set for the above two scripts are consistent, but the output file format is different. The followings are command lines for generating `.pth` file (need further quantize and deploy our model). 

```bash
python scripts/merge_llama_with_chinese_lora.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_or_alpaca_lora \
    --output_dir path_to_output_dir
```

where:

- `--base_model`: directory where the HF format LLaMA model weights and configuration files are saved (generated in Step 1)
- `--lora_model`: directory where the Chinese LLaMA/Alpaca LoRA model compressed file downloaded in the previous section is located, or the model name on Hugging Face Model Hub: `ziqingyang/chinese-alpaca-lora-7b` or `ziqingyang/chinese-llama-lora-7b`
- `--output_model`: directory to save the consolidated model weights (default: `./`)
- (optional) `--offload_dir`: for low-RAM users, please specify a offload directory

*(Optional) If necessary, you can convert the `.pth` files generated in this step to HuggingFace format using the script in Step 1.*