### What is LangChain?

[LangChain](https://github.com/hwchase17/langchain "Markdown") is a framework for developing LLM-driven applications, designed to assist developers in building end-to-end applications using LLM.

With the components and interfaces provided by LangChain, developers can easily design and build various LLM-powered applications such as question-answering systems, summarization tools, chatbots, code comprehension tools, information extraction systems, and more.

### How to use Chinese-Alpaca with LangChain?

Because the Chinese-Alpaca model obtained by merging LoRA weights into LLaMA has no structural differences from the original LLaMA except for the vocabulary, you can refer to any LangChain tutorial based on LLaMA for integration. 
The following documentation provides two examples of how to use Chinese-Alpaca in LangChain for

* Retrieval QA
* Summarization

The hyperparameters and prompt templates in the examples are not optimal and are only meant for demonstration. For more detailed instructions on using LangChain, please refer to its [official documentation](https://docs.langchain.com/docs/).


### Preparation

#### Environment

```
git clone https://github.com/hwchase17/langchain
cd langchain
pip install -e . 
pip install sentence_transformers faiss-cpu
```

#### Models

Referring to [Manual Conversion](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Manual-Conversion), merge the LoRA weights and generate  the Chinese-Alpaca model in HF format.
In Retrieval QA, LangChain selects the most relevant part of a document as context by matching the similarity between the query and the document content. This context is then combined with the question to generate the input for the LLM. Therefore, it is necessary to prepare a suitable embedding model for text/question vectorization during the matching process. We takes [GanymedeNil/text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese/tree/main) as an example for demonstration (in practice, you can choose other suitable embedding models based on your specific needs).

### Retrieval QA

This task utilizes LLM to perform automatic question answering for specific documents. The process includes reading texts, text segmentation, text/question vectorization, text-question matching, using the matched text as context along with the question to generate corresponding prompts as input to LLM, and generating answers.

```bash
cd scripts/langchain_demo
python langchain_qa.py \
  --embedding_path text2vec-large-chinese \
  --model_path chinese-alpaca-plus-7b-merged-hf
  --file_path doc.txt \
  --chain_type refine
```

Parameter Description:

* `--embedding_path`: Directory where the embedding model is located.
* `--model_path`: Directory where the merged Chinese-Alpaca model is located.
* `--file_path`: Document for retrieval QA.
* `--chain_type`: `refine`(default) or`stuff`, which represents different chains. For detailed explanations, refer to[here](https://docs.langchain.com/docs/components/chains/index_related_chains)。In simple terms, `stuff` is suitable for shorter documents, while `refine` is suitable for longer documents.
* `--gpus {gpu_ids}`: the GPU id(s) to use, default 0. You can specify multiple GPUs, for instance `0,1,2`.

Running example:

```bash
> python langchain_qa.py --embedding_path text2vec-large-chinese --model_path chinese-alpaca-plus-7b-merged-hf --file_path doc.txt --chain_type refine
# 中间输出信息省略
> 请输入问题：李白的诗是什么风格？
> 他的作品想像奇特丰富，风格雄起浪漫，意境独特，清新俊逸；善于利用夸饰与譬喻等手法、自然优美的词句，表现出奔放的情感。
```

### 3. Summarization

This task utilizes LLM to generate summarizations of given documents, helping to extract the core information.

```
cd scripts/langchain_demo
python langchain_sum.py \
  --model_path chinese-alpaca-plus-7b-merged-hf
  --file_path doc.txt \
  --chain_type refine
```

Parameter Description:

* `--model_path`: Directory where the merged Chinese-Alpaca model is located.
* `--file_path`: Document to be summarized.
* `--chain_type`: `refine`(default) or`stuff`, which represents different chains. For detailed explanations, refer to[here](https://docs.langchain.com/docs/components/chains/index_related_chains)。In simple terms, `stuff` is suitable for shorter documents, while `refine` is suitable for longer documents.
* `--gpus {gpu_ids}`: the GPU id(s) to use, default 0. You can specify multiple GPUs, for instance `0,1,2`.

Running example:

```bash
> python langchain_sum.py --model_path chinese-alpaca-plus-7b-merged-hf --file_path doc.txt
# 中间输出信息省略
> 李白是唐朝注意的浪漫主义诗人，被尊称为“诗仙”、“诗侠”、“酒仙”、“谪仙人“等称号。虽然性格桀骜不驯，但他留下了许多脍炙人口的诗歌作品，这些作品流传至今，被广泛传颂。尽管他只待长安不到两年就离开，但在晚年，他结识了杜甫和高适，并成为好友。然而，安史之乱导致他被捕入狱，最终在63岁去世，虽然他的大部分作品已经散佚，但他留下的九百多首诗歌仍然广受赞誉。
```

### Known Issue

LangChain initializes a FastTokenizer by default. However, this step can be very slow in certain environments. Currently, it is not possible to change this behavior by passing parameters. Therefore, If your program gets stuck at `loading LLM...`, you can change the tokenizer initialization code in `HuggingFacePipeline.from_model_id`

```python
tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
```

to

```python
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **_model_kwargs)
```
