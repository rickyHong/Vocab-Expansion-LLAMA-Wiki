### Q1: Why can't you release the complete model weights?

A: This question has been emphasized repeatedly before. The open source license for the LLaMA model does not allow us to do so, so related derivative work is seeking ways to bypass the restrictions. Please believe that we set up so many steps not to increase everyone's workload, but because of objective circumstances. After Facebook fully opens up the weights, we will release the complete model and directly loadable quantized models as soon as possible. During this period, we will also closely monitor other LLaMA-related repositories to see if there are better methods.

### Q2: Will there be versions of 33B and 65B in the future?

A: We cannot guarantee this at this time.

### Q3: The model doesn't perform well on some tasks!

A: There are several possible reasons: 1) LLaMA itself has limited support for Chinese, and most related derivative work is pre-trained/finetuned directly on the original version, while we have taken a more bold strategy - expanding the Chinese vocabulary, which may further exacerbate the problem of insufficient Chinese training, but whether it is beneficial for subsequent pre-training in the long run remains to be seen over time; 2) the quality of instruction data needs to be further improved; 3) there is still a lot of room for adjustment in training time, hyperparameters, etc.; 4) there is no RLHF; 5) the Q4 quantization may cause a decrease in performance, so you can try loading the FP16 model, which is relatively better (but slower).

### Q4: Why expand the vocabulary? Can't you just pre-train the original LLaMA with Chinese data?

A: The original LLaMA model's vocabulary size is 32K, mainly trained on English (see the [LLaMA paper](https://arxiv.org/abs/2302.13971v1) for more details), and support for multiple languages is not particularly ideal (you can compare the vocabulary size of the multilingual classic model XLM-R, which is 250K). Preliminary statistics show that the LLaMA vocabulary contains very few Chinese characters, so when cutting the words, the Chinese words are cut into smaller pieces, requiring multiple byte tokens to form a complete Chinese character, which leads to a decrease in information density. For example, in the model with the expanded vocabulary, a single Chinese character tends to be cut into one token, while in the original LLaMA, it may require 2-3 tokens to combine into one Chinese character, significantly reducing the efficiency of encoding and decoding.

### Q5: The reply is very short

Answer: It has been found that the Q4 quantitative model is more inclined to give short answers than the FP16 model. You can command to output a long reply in the prompt, such as "Please elaborate..." and so on. The remaining possible reasons include training data distribution, training parameters, decoding parameters, etc.

### Q6: Under Windows, the model cannot understand Chinese, the generation speed is very slow, etc.

Answer: If the model cannot understand Chinese and the generation speed is slow for Windows users, please refer to the solution in the following issue.

- About not being able to understand Chinese:
   - [Unicode (Windows) Support for llama.cpp](https://github.com/josStorer/llama.cpp-unicode-windows) (thanks @josStorer for development)
   - [#issue 11](https://github.com/ymcui/Chinese-LLaMA-Alpaca/issues/11) (Thanks to @LainNya, @boholder, @hyperzlib and others for their solutions)
- Regarding the slow generation: [#issue 51](https://github.com/ymcui/Chinese-LLaMA-Alpaca/issues/51) (thanks to @wscsjnhboy for the solution)

### Q7: Chinese-LLaMA 13B model cannot be launched with llama.cpp, reporting inconsistent dimensions.

Answer: Problem solved with the new merge script.

~~Answer: This is related to the fact that the 13B model is split into two files with different sizes. See [Issue#133](https://github.com/ymcui/Chinese-LLaMA-Alpaca/issues/133). Users with strong hands-on skills can try to solve this issue using the method mentioned in the issue. On the other hand, the Chinese-LLaMA model itself is not designed for dialogue or interaction, but rather to provide a foundation for further fine-tuning on Chinese instruction tasks or other tasks. Therefore, it is not recommended to load the Chinese-LLaMA model with llama.cpp.~~

### Q8: Chinese-Alpaca-Plus does not show better performance than the others

Answer: As we changed LoRA rank of Alpaca-Plus models, the base model should be LLaMA-Plus but not the original LLaMA. Please carefully read our guidelines for [merging Alpaca-Plus models](./Manual-Conversion#multiple-lora-weights-merging-applicable-to-chinese-alpaca-plus).

### Q9: The model does not perform well on NLU tasks, such as text classification.

Answer: Unlike other Chinese LLMs, we specifically removed those NLU-like instruction data when training Alpaca models. For example, we removed the data that was created from NLU datasets and automatically converted to instruction-like data using templates. In you need to improve the model's NLU performance, please consider finetuning our models using those data.

### Q10: Why 33B not 30B?

Answer: Actually, it is 33B. In [LLaMA paper](https://arxiv.org/abs/2302.13971v1), the actual parameter size is 32.5B and is referred as 33B in the rest of the paper. However, due to a [typo in releasing LLaMA](https://github.com/facebookresearch/llama/issues/49), it is wrongly named as 30B. We stick to the actual parameter size naming here. Keep in mind that both 30B and 33B refer to the same thing, the second largest model of LLaMA family.

### Q11: Inconsistent SHA256

Answer: 1) make sure you have installed the recommended version of dependencies. 2) Please first make sure that you have checked the SHA256 of original LLaMA and our LoRA model (download from official links not third-party). If you have checked that they are the same as listed in SHA256.md, then the merged file would be fine. As there are some rare cases that the SHA256 differs but the actual performance is not affected.