
Enter the `scripts` directory of the project, and the command for launching the SFT script [run_clm_sft_with_peft.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/run_clm_sft_with_peft.py) is as follows (some variables need to be specified by the suer):

```bash

lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_droppout=0.05

python run_clm_sft_with_peft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_batch_size} \
    --per_device_eval_batch_size ${per_device_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --leraning_rate ${lr} \
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
    --force_resize_embeddings [True|False]
```

The meanings of most arguments are self-evident. Here are explanations for some of the arguments:



* `${pretrained_model}`: Location of the Chinese-LLaMA model which has been merged with the LoRA weight (in HF format).
* `${chinese_tokenizer_path}`: Directory where the Chinese-Alpaca tokenizer is located
* `${dataset_dir}: Directory of the SFT data, which can contain one or more `json` files in the Stanford Alpaca format.
* `${--validation_file}`: A single `json` file used as the development set, in the Stanford Alpaca format.

Stanford Alpaca format:
```
[
  {"instruction" : ...,
   "input" : ...,
   "output" : ...},
  ...
]
```

If you want to continue training the LoRA weight of the Chinese-Alpaca model:

* `${peft_model}`: Location of the Chinese-Alpaca-LoRA weight and config
* No need to specify `lora_rank`, `lora_alpha`, `lora_dropout`, `trainable` and `modules_to_save`
* Set`--force_resize_embeddings True`

If you want to train a completely new LoRA weight based on Chinese-LLaMA:

* `${peft_model}`: Do not provide this parameter.
* Specify `lora_rank`, `lora_alpha`, `lora_dropout`, `trainable` and `modules_to_save`
* Set `--force_resize_embeddings False`


The hyperparameters listed here (especially the learning rate and parameters related to the total batch size) are for reference only. Please feel free to adjust them based your training data and hardware conditions.


To use DeepSpeed, you can add the following argument `--deepspeed ${deepspeed_config_file}`.  Additionally, you should use `torchrun` to launch the script:
```bash
torchrun \
  --nnodes ${num_nodes} \
  --nproc_per_node ${num_gpu_per_node} 
  --node_rank ${node_rank} \
  --master_addr ${master_addr} \
  --master_port ${master_port} \
  run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    ...
```


### ⚠️Important⚠️

**Due to frequent changes in the Peft library, this code is only applicable to specific versions of Peft. Please install [Peft commit id 13e53fc](https://github.com/huggingface/peft/tree/13e53fc) from source. Using other versions of Peft may result in undesirable training behavior and results.
