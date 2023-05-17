### ⚠️Important⚠️

**Due to frequent changes in the Peft library, this code is only applicable to specific versions of Peft. Please install [Peft commit id 13e53fc](https://github.com/huggingface/peft/tree/13e53fc) from source. Using other versions of Peft may result in undesirable training behavior and results.**

The script [scripts/run_clm_pt_with_peft.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/run_clm_pt_with_peft.py) is used for Pre-training Stage 2. However, we do not recommend performing Pre-training Stage 1 if the computational resources and time are limited as the model takes longer to converge.

Enter the `scripts` directory of the project, and run `bash run_pt.sh` to start pre-training (use a single GPU by default). Users should edit the script set value of parameters. The contents of `run_pt.sh` are as follows:



Execute the following command to start pre-training (some variables need to be specified by the suer):

```bash
########parameters########
lr=2e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=path/to/hf/llama/dir
chinese_tokenizer_path=path/to/chinese/llama/tokenizer/dir
dataset_dir=path/to/pt/data/dir
data_cache=temp_data_cache_dir
per_device_train_batch_size=1
per_device_eval_batch_size=1
training_steps=100
gradient_accumulation_steps=1
output_dir=output_dir

deepspeed_config_file=ds_zero2_no_offload.json

#######launch########
torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --seed $RANDOM \
    --fp16 \
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 500 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size 512 \
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
    --gradient_checkpointing \
    --ddp_find_unused_parameters False


```

The meanings of most arguments are self-evident. Here are explanations for some of the arguments:

* `--model_name_or_path`: Directory that stores the original LLaMA model in HuggingFace format
* `--tokenizer_name_or_path`: Directory that stores the Chinese-LLaMA tokenizer
* `dataset_dir`: Directory of the pre-training data, which can contain multiple plain text files whose filenames end with `txt`
* `data_cache_dir`: Directory that stores data cache files


The hyperparameters listed here (especially the learning rate and parameters related to the total batch size) are for reference only. Please feel free to adjust them based your training data and hardware conditions.

VRAM-saving tips:

* If the VRAM is insufficient, you can remove --modules_to_save ${modules_to_save} \ from the script. This will exclude training for embed_tokens and lm_head (which have large parameters) and only train the LoRA parameters, thus saving memory (It is suggested to experiment based on Chinese-LLaMA instead of excluding the training of embed_tokens and lm_head from the pre-training stage).


To launch with multi-node and multi-GPU:
```bash
torchrun \
  --nnodes ${num_nodes} \
  --nproc_per_node ${num_gpu_per_node} 
  --node_rank ${node_rank} \
  --master_addr ${master_addr} \
  --master_port ${master_port} \
  run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    ...
```

