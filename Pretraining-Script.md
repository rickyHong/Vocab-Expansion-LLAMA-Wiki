
The script [scripts/run_clm_pt_with_peft.py](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/run_clm_pt_with_peft.py) is used for Pre-training Stage 2. However, we do not recommend performing Pre-training Stage 1 if the computational resources and time are limited as the model takes longer to converge.

Execute the following command to start pre-training (some variables need to be specified by the suer):

```bash

lr=2e-4
lora_rank=8
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_droppout=0.1

python scripts/run_clm_pt_with_peft.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir $data_cache \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_batch_size} \
    --per_device_eval_batch_size ${per_device_batch_size} \
    --do_train \
    --seed $RANDOM \
    --fp16
    --max_steps ${training_steps} \
    --lr_scheduler_type cosine \
    --leraning_rate ${lr} \
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
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16
```

其中一些参数的含义不言自明。部分参数的解释如下：
* `${pretrained_model}`: 原版LLaMA模型HuggingFace格式权重位置
* `${chinese_tokenizer_path}`: Chinese-LLaMA tokenizer所在的目录
* `${dataset_dir}`: 预训练数据的目录，可包含多个纯文本文件
* `${--data_cache_dir}` 指定一个存放数据缓存文件的目录


The hyperparameters listed here (especially the learning rate and parameters related to the total batch size) are for reference only. Please feel free to adjust them based your training data and hardware conditions.


To use DeepSpeed, you can add the following parameter `--deepspeed ${deepspeed_config_file}`.  Additionally, you should use torchrun to launch the script:
```bash
torchrun --nnodes ${num_nodes} --nproc_per_node ${num_gpu_per_node} --node_rank ${node_rank} --master_addr ${master_addr} --master_port ${master_port} run_clm_pt_with_peft.py ...
```


### ⚠️Important⚠️

**Due to frequent changes in the Peft library, this code is only applicable to specific versions of Peft. Please install [Peft commit id 13e53fc](https://github.com/huggingface/peft/tree/13e53fc) from source. Using other versions of Peft may result in undesirable training behavior and results.

