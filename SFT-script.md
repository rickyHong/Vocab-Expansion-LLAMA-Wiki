### ⚠️Important⚠️

**Due to frequent changes in the Peft library, this code is only applicable to specific versions of Peft. Please install [Peft commit id 13e53fc](https://github.com/huggingface/peft/tree/13e53fc) from source. Using other versions of Peft may result in undesirable training behavior and results.**


Enter the `scripts` directory of the project, and run `bash run_sft.sh` to start instruction fine-tuning (use a single GPU by default). Users should edit the script set value of parameters. The contents of `run_sft.sh` are as follows:

```bash
########parameters########
lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=path/to/hf/llama/or/merged/llama/dir/or/model_id
chinese_tokenizer_path=path/to/chinese/llama/tokenizer/dir
dataset_dir=path/to/sft/data/dir
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=100
gradient_accumulation_steps=1
output_dir=output_dir
peft_model=path/to/peft/model/dir
validation_file=validation_file_name

deepspeed_config_file=ds_zero2_no_offload.json

########launch########
torchrun --nnodes 1 --nproc_per_node 1 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 512 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --peft_path ${peft_model} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
```

The meanings of most arguments are self-evident. Here are explanations for some of the arguments:

* `--tokenizer_name_or_path`: Directory where the Chinese-Alpaca tokenizer is located
* `--dataset_dir`: Directory of the SFT data, which can contain one or more `json` files in the Stanford Alpaca format.
* `--validation_file`: A single `json` file used as the development set, in the Stanford Alpaca format.

Stanford Alpaca format:
```
[
  {"instruction" : ...,
   "input" : ...,
   "output" : ...},
  ...
]
```

Configuration:

* If you want to continue training the LoRA weight of the Chinese-Alpaca model:
  - `--model_name_or_path`: The original LLaMA model in HF format (if continue training non-Plus model), **or** the Chinese-LLaMA model (in HF format) which has been merged with the Chinese-LLaMA-Plus-LoRA weight （if continue training Plus model）
  - `--peft_path`: Location of the Chinese-Alpaca-LoRA weight and config
  
  * No need to specify `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--trainable` and `--modules_to_save`
  

* If you want to train a completely new LoRA weight based on Chinese-LLaMA:

  * `--model_name_or_path`: the Chinese-LLaMA model (in HF format) which has been merged with the corresponding LoRA weight （no matter if it is Plus model or not）

  * `--peft_path`: Do not provide this parameter and remove `--peft_path` from the script

  * Specify `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--trainable` and `--modules_to_save`





The hyperparameters listed here (especially the learning rate and parameters related to the total batch size) are for reference only. Please feel free to adjust them based your training data and hardware conditions.

VRAM-saving tips:
* If the VRAM is insufficient, you can remove `--modules_to_save ${modules_to_save} \` from the script. This will exclude training for embed_tokens and lm_head (which have large parameters) and only train the LoRA parameters, thus saving memory.
* If errors occur in the program after executing the previous step, please remove `--gradient_checkpointing \` and try again.

To launch with multi-node and multi-GPU:
```bash
torchrun \
  --nnodes ${num_nodes} \
  --nproc_per_node ${num_gpu_per_node} 
  --node_rank ${node_rank} \
  --master_addr ${master_addr} \
  --master_port ${master_port} \
  run_clm_sft_with_peft.py \
    ...
```
