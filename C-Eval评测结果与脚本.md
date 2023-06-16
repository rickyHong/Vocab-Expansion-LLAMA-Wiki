### C-Eval部分效果展示

本项目在近期推出的[C-Eval评测数据集](https://cevalbenchmark.com)上测试了相关模型效果，其中测试集包含12.3K个选择题，涵盖52个学科。以下是部分模型的valid和test集评测结果（Average），完整结果请参考[技术报告](https://arxiv.org/abs/2304.08177)。

| 模型                    | Valid (zero-shot) | Valid (5-shot) | Test (zero-shot) | Test (5-shot) |
| ----------------------- | :---------------: | :------------: | :--------------: | :-----------: |
| Chinese-Alpaca-33B      |       43.3        |      42.6      |       41.6       |     40.4      |
| Chinese-LLaMA-33B       |       34.9        |      38.4      |       34.6       |     39.5      |
| Chinese-Alpaca-Plus-13B |       43.3        |      42.4      |       41.5       |     39.9      |
| Chinese-LLaMA-Plus-13B  |       27.3        |      34.0      |       27.8       |     33.3      |
| Chinese-Alpaca-Plus-7B  |       36.7        |      32.9      |       36.4       |     32.3      |
| Chinese-LLaMA-Plus-7B   |       27.3        |      28.3      |       26.9       |     28.4      |

接下来将介绍C-Eval数据集的预测方法，用户也可参考我们的Colab Notebook：<a href="https://colab.research.google.com/drive/12YewimRT7JuqJGOejxN7YG8jq2de4DnF?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### 数据准备

从[C-Eval官方](https://github.com/SJTU-LIT/ceval "Markdown")指定路径下载评测数据集，并解压至`data`文件夹：
```
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip -d data
```
将`data`文件夹放置于本项目的`scripts/ceval`目录下。


### 运行预测脚本

运行以下脚本：
```bash
model_path=path/to/chinese_llama_or_alpaca
output_path=path/to/your_output_dir

cd scripts/ceval
python eval.py \
    --model_path ${model_path} \
    --cot False \
    --few_shot False \
    --with_prompt True \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --ntrain 5 \
    --do_save_csv False \
    --do_test False \
    --output_dir ${output_path} \
```

#### 参数说明

* `model_path`：待评测模型所在目录（合并LoRA后的HF格式模型）

* `cot`：是否使用chain-of-thought

* `few_shot`：是否使用few-shot

* `ntrain`：`few_shot=True`时，指定few-shot实例的数量（5-shot：`ntrain=5`）；`few_shot=False`时该项不起作用

* `with_prompt`：模型输入是否包含针对Alpaca模型的指令模板

* `constrained_decoding`：由于C-Eval评测的标准答案格式为选项'A'/'B'/'C'/'D'，所以我们提供了两种从模型生成内容中抽取答案的方案：
  * 当`constrained_decoding=True`，计算模型生成的第一个token分别为'A', 'B', 'C', 'D'的概率，选择其中概率最大的一个作为答案

  * 当`constrained_decoding=False`，用正则表达式从模型生成内容中提取答案

* `temperature`：模型解码时的温度

* `n_times`：指定评测的重复次数，将在`output_dir`下生成指定次数的文件夹

* `do_save_csv`：是否将模型生成结果、提取的答案等内容保存在csv文件中

* `output_dir`：指定评测结果的输出路径

* `do_test`：在valid或test集上测试：当`do_test=False`，在valid集上测试；当`do_test=True`，在test集上测试


### 评测输出
- 模型预测完成后，生成目录`outputs/take*`，其中`*`代表数字，范围为0至`n_times-1`，分别储存了`n_times`次解码的结果。

- `outputs/take*`下包含`submission.json`和`summary.json`两个json文件。若`do_save_csv=True`，还将包含52个保存的模型生成结果、提取的答案等内容的csv文件。

* `submission.json`为依据官方提交规范生成的存储模型评测答案的文件，形式如：
  
  ```
  {
      "computer_network": {
          "0": "A",
          "1": "B",
          ...
      },
        "marxism": {
          "0": "B",
          "1": "A",
          ...
        },
    	...
  }
  ```
  
* `summary.json`包含模型在52个主题下、4个大类下和总体平均的评测结果。例如，json文件最后的`All`字段中会显示总体平均效果：
  
  ```json
	"All": {
	  "score": 0.36701337295690933,
	  "num": 1346,
    "correct": 494.0
  }
  ```
  其中`score`为准确率，`num`为测试的总样本条数，`correct`为正确的数量。
  

⚠️ **注意，当在测试集上预测时（`do_test=True`），因为没有测试集标签，`score`和`correct`将为0，为正常现象。** 测试集结果需要将`submission.json`文件提交至C-Eval官方进行获取，具体请参考C-Eval官方提交流程。