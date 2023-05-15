### 什么是LangChain？

[LangChain](https://github.com/hwchase17/langchain "Markdown")是一个用于开发由LLM驱动的应用程序的框架，旨在帮助开发人员使用LLM构建端到端的应用程序。

借助LangChain提供的组件和接口，开发人员可以方便地设计与搭建诸如问答、摘要、聊天机器人、代码理解、信息提取等多种基于LLM能力的应用程序。

### 如何在LangChain中使用Chinese-Alpaca？

因为将LoRA权重合并进LLaMA后的模型与原版LLaMA除了词表不同之外结构上没有其他区别，因此可以参考任何基于LLaMA的LangChain教程进行集成。
以下文档通过两个示例，分别介绍在LangChain中如何使用Chinese-Alpaca实现
* 检索式问答
* 摘要生成

例子中的超参、prompt模版均未调优，仅供演示参考用。关于LangChain的更详细的使用说明，请参见其[官方文档](https://docs.langchain.com/docs/)。


### 一、准备工作

#### Step 1: 环境准备

```
git clone https://github.com/hwchase17/langchain
cd langchain
pip install -e . 
pip install sentence_transformers faiss-cpu
```

#### Step2 : 模型准备

参考[手动模型合并与转换](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/%E6%89%8B%E5%8A%A8%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6%E4%B8%8E%E8%BD%AC%E6%8D%A2 "Markdown")，合并模型生成HF格式的Chinese-Alpaca模型权重，并将模型保存至本地。
在检索式问答中，LangChain通过问句与文档内容的相似性匹配，来选取文档中与问句最相关的部分作为上下文，与问题组合生成LLM的输入。因此，需要准备一个合适的embedding model用于匹配过程中的文本/问题向量化。本文以[GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main "Markdown")为例进行说明（实际上，也可以根据实际需要选择其他合适的embedding model）。

### 二、检索式问答

该任务使用LLM完成针对特定文档的自动问答，流程包括：文本读取、文本分割、文本/问题向量化、文本-问题匹配、将匹配文本作为上下文和问题组合生成对应Prompt中作为LLM的输入、生成回答。

```bash
cd scripts/langchain_demo
python langchain_qa.py \
  --embedding_path text2vec-large-chinese \
  --model_path chinese-alpaca-plus-7b-merged-hf
  --file_path doc.txt \
  --chain_type refine
```

参数说明：

* `--embedding_path`: embedding model所在目录或或模型名
* `--model_path`: 合并后的Alpaca模型所在目录
* `--file_path`: 待进行检索与提问的文档
* `--chain_type`: 可以为`refine`(默认)或`stuff`，为两种不同的chain，详细解释见[这里](https://docs.langchain.com/docs/components/chains/index_related_chains)。简单来说，stuff适用于较短的篇章，而refine适用于较长的篇章。
* `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如`0,1,2`

运行示例:

```bash
> python langchain_qa.py --embedding_path text2vec-large-chinese --model_path chinese-alpaca-plus-7b-merged-hf --file_path doc.txt --chain_type refine
# 中间输出信息省略
> 请输入问题：李白的诗是什么风格？
> 他的作品想像奇特丰富，风格雄起浪漫，意境独特，清新俊逸；善于利用夸饰与譬喻等手法、自然优美的词句，表现出奔放的情感。
```

### 三、摘要生成

该任务使用LLM完成给定文档的摘要生成，以帮助提炼文档中的核心信息。

```
cd scripts/langchain_demo
python langchain_sum.py \
  --model_path chinese-alpaca-plus-7b-merged-hf
  --file_path doc.txt \
  --chain_type refine
```

参数说明：

* `--model_path`: 合并后的Alpaca模型所在目录
* `--file_path`: 待进行摘要的文档
* `--chain_type`: 可以为`refine`(默认)或`stuff`，为两种不同的chain，详细解释见[这里](https://docs.langchain.com/docs/components/chains/index_related_chains)。简单来说，stuff适用于较短的篇章，而refine适用于较长的篇章。
* `--gpus {gpu_ids}`: 指定使用的GPU设备编号，默认为0。如使用多张GPU，以逗号分隔，如`0,1,2`

运行示例:

```bash
> python langchain_sum.py --model_path chinese-alpaca-plus-7b-merged-hf --file_path doc.txt
# 中间输出信息省略
> 李白是唐朝注意的浪漫主义诗人，被尊称为“诗仙”、“诗侠”、“酒仙”、“谪仙人“等称号。虽然性格桀骜不驯，但他留下了许多脍炙人口的诗歌作品，这些作品流传至今，被广泛传颂。尽管他只待长安不到两年就离开，但在晚年，他结识了杜甫和高适，并成为好友。然而，安史之乱导致他被捕入狱，最终在63岁去世，虽然他的大部分作品已经散佚，但他留下的九百多首诗歌仍然广受赞誉。
```

### 已知问题

LangChain中默认会初始化FastTokenizer，而这一步在某些transformers+tokenizers版本中会非常慢，目前无法通过传参的方式修改这一行为。因此建议修改LangChain中`HuggingFacePipeline.from_model_id`中的相关代码，将其中tokenizer的初始化部分
```python
tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
```
修改为
```python
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **_model_kwargs)
```
以缓解此问题。