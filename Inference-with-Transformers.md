We provide two scripts to use the native Transformers for inference: a command-line interface and a web graphical interface.

Taking the loading of the Chinese-Alpaca-7B model as an example (for loading Plus models, refer to **Loading Chinese-Alpaca-Plus** below):

### Command-line Interface

```bash
python scripts/inference/inference_hf.py \
    --base_model path_to_original_llama_hf_dir \
    --lora_model path_to_chinese_llama_or_alpaca_lora \
    --with_prompt \
    --interactive
```

If you have already merged the models with `merge_llama_with_chinese_lora_to_hf.py` , you don't need to specify `--lora_model`:

```bash
python scripts/inference/inference_hf.py \
    --base_model path_to_merged_llama_or_alpaca_hf_dir \
    --with_prompt \
    --interactive
```

Parameter description:

- `--base_model {base_model}`: Directory containing the LLaMA model weights and configuration files in HF format.
- `--lora_model {lora_model}`: Directory of the Chinese LLaMA/Alpaca LoRa files after decompression, or the [🤗Model Hub model name](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md#model-hub). If this parameter is not provided, only the model specified by `--base_model` will be loaded.
- `--tokenizer_path {tokenizer_path}`: Directory containing the corresponding tokenizer. If this parameter is not provided, its default value is the same as `--lora_model`; if the `--lora_model` parameter is not provided either, its default value is the same as `--base_model`.
- `--with_prompt`: Whether to merge the input with the prompt template. **If you are loading an Alpaca model, be sure to enable this option!**
- `--interactive`: Launch interactively for multiple **single-round question-answer** sessions (this is not the contextual dialogue in llama.cpp).
- `--data_file {file_name}`: In non-interactive mode, read the content of `file_name` line by line for prediction.
- `--predictions_file {file_name}`: In non-interactive mode, write the predicted results in JSON format to `file_name`.
- `--use_cpu`: Only use CPU for inference.
- `--gpus {gpu_ids}`: the GPU id(s) to use, default 0. You can specify multiple GPUs, for instance `0,1,2`.

### Web Graphical Interface

This method will start a web frontend page for interaction and support multi-turn conversations. In addition to `Transformers`, you need to install `Gradio` and `mdtex2html`:

```bash
pip install gradio
pip install mdtex2html
```

The launch command:

```
python scripts/inference/gradio_demo.py \
	--base_model path_to_original_llama_hf_dir \
	--lora_model path_to_chinese_alpaca_lora
```

If you have already merged the LoRA weights with `merge_llama_with_chinese_lora_to_hf.py`, you don't need to specify `--lora_model`:

```
python scripts/gradio_demo.py --base_model path_to_merged_alpaca_hf_dir 
```

Parameter description:

* `--base_model {base_model}`: Directory containing the LLaMA model weights and configuration files in HF format.
* `--lora_model {lora_model}`: Directory of the Chinese LLaMA/Alpaca LoRa files after decompression, or the [🤗Model Hub model name](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md#model-hub). If this parameter is not provided, only the model specified by `--base_model` will be loaded.
* `--tokenizer_path {tokenizer_path}`: Directory containing the corresponding tokenizer. If this parameter is not provided, its default value is the same as `--lora_model`; if the `--lora_model` parameter is not provided either, its default value is the same as `--base_model`.
* `--use_cpu`: Only use CPU for inference.
* `--gpus {gpu_ids}`: the GPU id(s) to use, default 0. You can specify multiple GPUs, for instance `0,1,2`.

### Loading Chinese-Alpaca-Plus

Currently, neither of the scripts supports directly loading Chinese-Alpaca-Plus for inference from LoRA weights. If you want to perform inference using Chinese-Alpaca-Plus, please follow the steps below:

1. Using `merge_llama_with_chinese_lora.py` to obtaining a single model weight file in HF format
```bash
python scripts/merge_llama_with_chinese_lora.py \
    --base_model path_to_hf_llama \
    --lora_model path_to_chinese_llama_plus_lora,path_to_chinese_alpaca_plus_lora \
    --output_type huggingface \
    --output_dir path_to_merged_chinese_alpaca_plus
```
2. Loading the merged model with `inference_hf.py` or `gradio_demo.py`:
```bash
python scripts/inference_hf.py \
    --base_model path_to_merged_chinese_alpaca_plus \
    --with_prompt --interactive
```

### Note

- Due to differences in decoding implementation details between different frameworks, this script cannot guarantee to reproduce the decoding effect of llama.cpp.
- This script is for convenient and quick experience only, and has not been optimized for fast inference.
- When running 7B model inference on a CPU, make sure you have 32GB of memory; when running 7B model inference on a single GPU, make sure you have 20GB VRAM.