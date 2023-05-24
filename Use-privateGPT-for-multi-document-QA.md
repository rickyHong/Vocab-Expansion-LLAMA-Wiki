[privateGPT](https://github.com/imartinez/privateGPT) is an open-source project based on [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [LangChain](https://github.com/hwchase17/langchain) among others. It aims to provide an interface for localizing document analysis and interactive Q&A using large models. Users can utilize privateGPT to analyze local documents and use GPT4All or llama.cpp compatible large model files to ask and answer questions about document content, ensuring data localization and privacy.

Since this project is based on related derivatives of LLaMA, this article will introduce the usage of privateGPT using the GGML format model in llama.cpp as an example.

For more detailed content and usage, please refer to the privateGPT official directory: https://github.com/imartinez/privateGPT

### Prerequisites: Install llama-cpp-python

Since the GGML model in llama.cpp is used in privateGPT, it is necessary to install the llama-cpp-python extension in advance. The following installation method does not use any acceleration library.

```
pip install llama-cpp-python
```

If you wish to install a version compatible with OpenBLAS/cuBLAS/CLBlast, please refer to: https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast

#### Must read for Mac M series chip users

**Make sure that the Python in the current installation environment supports the arm64 architecture, otherwise the execution speed will be more than 10x slower.** The test method is to execute the following python command after installing llama-cpp-python, and the model path should be replaced with your local GGML model file supported by llama.cpp.

```
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="./models/7B/ggml-model.bin")
```

If `NEON = 1` is displayed, it indicates normal; `NEON = 0` indicates that it was not installed correctly according to the arm64 architecture. Below is an example of a log supporting ARM NEON acceleration.

```
system_info: n_threads = 8 / 10 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | VSX = 0 |
```

#### How to install python compatible with arm64?

If you use conda, you can use the following command to create the related environment, here choosing Python 3.10 to meet the requirements of privateGPT.

```
CONDA_SUBDIR=osx-arm64 conda create -n privategpt python=3.10 -c conda-forge
```

### Step 1: Clone the directory and install the dependency packages

After **correctly installing** llama-cpp-python, you can continue to install privateGPT, the specific command is as follows (note that python>=3.10).

```
git clone https://github.com/imartinez/privateGPT.git
cd privateGPT
pip3 install -r requirements.txt
```

### Step 2: Modify the configuration file

Create a configuration file named `.env` in the root directory of privateGPT, an example of a well-written configuration file is:

```
MODEL_TYPE=LlamaCpp
PERSIST_DIRECTORY=db
MODEL_PATH=/your-path-to-ggml-model/ggml-model-q5_1.bin
MODEL_N_CTX=2048
EMBEDDINGS_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

- MODEL_TYPE: Fill in LlamaCpp

- PERSIST_DIRECTORY: Fill in the location where the analysis files are stored. Here, a `db` directory will be created in the root directory of privateGPT.

- MODEL_PATH: Points to the location where the large model is stored, which here points to the GGML file supported by llama.cpp.

- MODEL_N_CTX: The maximum token limit of the large model, set to 2048.

- EMBEDDINGS_MODEL_NAME: SentenceTransformers word vector model location, can specify the path on HuggingFace (will be automatically downloaded). Other officially supported models can be referred to: https://www.sbert.net/docs/pretrained_models.html

### Step 3: Analyze local files

privateGPT supports the analysis of the following common document formats, for example (only the most commonly used are listed):

  - Word files: `.doc`, `.docx`
  - PPT files: `.ppt`, `.pptx`
  - PDF files: `.pdf`
  - Pure text files: `.txt`
  - CSV files: `.csv`
  - Markdown files: `.md`
  - Email files: `.eml`, `.msg`

Place the documents to be analyzed (not limited to a single document) in the `source_documents` directory under the root node of privateGPT. Here, [the LangChain sample data of this project](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/langchain_demo/doc.txt) is used as an example for introduction. The directory structure is similar:

  ```
  > ls source_documents
  doc.txt
  ```

Next, run the ingest command to analyze the document.

```
python ingest.py
```

The output is as follows (the author uses M1 Max, and the analysis only took a few seconds). It should be noted that the word vector model in the configuration file will be downloaded for the first time (if the given is a huggingface address, not a local path).

```
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.42s/it]
Loaded 1 new documents from source_documents
Split into 7 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run privateGPT.py to query your documents
```

⚠️ Note: If there are related analysis files in the `db` directory, the data files will be accumulated. If you only want to parse the current document, please clear the `db` directory before ingesting.

### Step 4: Ask questions about the document locally

After the analysis of the document in the previous step is completed, you can run the following command to start asking questions about the document:

```
python privateGPT.py
```

After the following prompt appears, you can enter a question:

```
Enter a query: 
```

For example, ask `李白的诗是什么风格？`, the result is as follows:

```
> Answer:
 很显然，李白的风格是浪漫主义。他的作品以超然的情思、瑰丽的意境和豪放的情感而著称。他的诗歌常常描绘自然风景，并运用夸张的手法来表现自己的情感。他最著名的作品包括《将进酒》、《庐山谣》、《夜泊牛渚怀古》等。同时，他还尝试用各种方式表达自己对事物的态度，例如“抽刀断水水更流”这句诗描述了当时社会状况和李白个人所处的境况。

> source_documents/doc.txt:
李白的诗歌在唐朝已被选进殷璠编选的《河岳英灵集》、于敦煌石室发现的《唐写本唐人选唐诗》、韦庄编选的《又玄集》和韦縠编选的《才调集》。唐文宗御封李白的诗歌、裴旻的剑舞、张旭的草书称为“三绝”[2]。其作品想像奇特丰富，风格雄奇浪漫，意境独特，清新俊逸；善于利用夸饰与譬喻等手法、自然优美的词句，表现出奔放的情感。诗句行云流水，浑然天成。李白诗篇传诵千年，众多诗句已成经典，清赵翼称：“李杜诗篇万口传”（例如“抽刀断水水更流，举杯消愁愁更愁”等，更被谱入曲）。李白在诗歌的艺术成就被认为是中国浪漫主义诗歌的巅峰。诗作在全唐诗收录于卷161至卷185。有《李太白集》传世。杜甫曾经这样评价过李白的文章：“笔落惊风雨，诗成泣鬼神”、“白也诗无敌，飘然思不群”。
生平
早年

> source_documents/doc.txt:
中年
李白曾经在唐玄宗天宝元年（742年）供奉翰林。有一次皇帝因酒酣问李白说：“我朝与天后（武后）之朝何如？”白曰：“天后朝政出多门，国由奸幸，任人之道，如小儿市瓜，不择香味，惟拣肥大者；我朝任人如淘沙取金，剖石采用，皆得其精粹者。”玄宗听后大笑不止[8][9]。但是由于他桀骜不驯的性格，所以仅仅不到两年他就离开了长安。据说是因为他作的《清平调》得罪了当时宠冠后宫的杨贵妃（因李白命“力士脱靴”，高力士引以为大耻，因而以言语诱使杨贵妃认为“可怜飞燕倚新妆”几句是讽刺她）而不容于宫中[注 3]。天宝三年（745年）“恳求还山，帝赐金放还”，离开长安。
后在洛阳与另两位著名诗人杜甫、高适相识，并结为好友。
晚年
天宝十一年（752年）李白年届五十二岁，北上途中游广平郡邯郸、临洺、清漳等地。十月，抵幽州。初有立功边疆思想，在边地习骑射。后发现安禄山野心，登黄金台痛哭。不久即离幽州南下。

> source_documents/doc.txt:
李白[注 1]（701年5月19日—762年11月30日），字太白，号青莲居士，中国唐朝诗人。李白自言祖籍陇西成纪（今甘肃静宁西南），汉飞将军李广后裔，西凉武昭王李暠之后，与李唐皇室同宗。
一说其幼时内迁，寄籍剑南道绵州昌隆（今四川省江油市青莲镇）。一说先人隋末被窜于碎叶，出生于碎叶，属唐安西都护府（今吉尔吉斯斯坦共和国楚河州托克马克市）。有“诗仙”、“诗侠”、“酒仙”、“谪仙人”等称呼，活跃于盛唐[1]，为杰出的浪漫主义诗人。与杜甫合称“李杜”[注 2]。被贺知章呼为“天上谪仙”、“李谪仙”。

> source_documents/doc.txt:
李阳冰在《草堂集序》中说李白是病死的[11]；皮日休在诗作中记载，李白是患“腐胁疾”而死的[12]。
《旧唐书》则记载，李白流放虽然遇赦，但因途中饮酒过度，醉死于宣城。中国民间有“太白捞月”的传说：李白在舟中赏月，饮酒大醉，想要跳下船至水里捞月而溺死[13][14][15]；在民间的求签活动中亦有“太白捞月”一签文，乃是下下签[16]。
作品
李白一生创作大量的诗歌，绝大多数已散佚[17]，流传至今的只有九百多首。他的诗歌创作涉及的中国古典诗歌的题材非常广泛，而且在许多题材都有名作出现，而且因为际遇的不同，每个时期的诗风都有所不同。
```

The whole process is not very fast, it took about half a minute to give the related result, and 4 data sources will be given.

Enter `exit` to end the script run.


#### Advanced configuration

##### Use more threads for acceleration

`privateGPT.py` actually calls the interface of llama-cpp-python, so if you do not make any code modifications, the default decoding strategy is used. Open `privateGPT.py` and find the following statement (around lines 30-35, it varies depending on different versions).

```
llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
```

Here is the definition of the LlamaCpp model. More custom parameters can be passed in according to the interface definition of llama-cpp-python. The following is an example, which additionally increases the number of decoding threads, which helps to improve the decoding speed (please configure according to the actual number of physical cores).

```
llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, n_threads=8)
```

A few lines after the above definition, LangChain's RetrievalQA will be used for interaction. For the specific definition and configuration method, please refer to the LangChain documentation.

##### Use Alpaca prompt template

If you are using Alpaca models, you can also pass the prompt template before generation. For example, near line 39, you can modify the following code

```
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
```

to

```
    prompt_template = ("Below is an instruction that describes a task. "
                      "Write a response that appropriately completes the request.\n\n"
                      "### Instruction:\n{context}\n\n{question}\n\n### Response: ")
    from langchain import PromptTemplate
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff",
        retriever=retriever,
        return_source_documents= not args.hide_source,
        chain_type_kwargs={"prompt":PROMPT})
```

#### Optimize LangChian

In `privateGPT.py`, the default chain type is `stuff`. However, it is not suitable for long documents. You can switch to `refine` or `map_reduce` chain. Please refer to [LangChain example](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/langchain_demo/langchain_qa.py) . For instance, if using `refine`, users should first define two prompt template:

```
    refine_prompt_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "这是原始问题: {question}\n"
        "已有的回答: {existing_answer}\n"
        "现在还有一些文字，（如果有需要）你可以根据它们完善现有的回答。"
        "\n\n"
        "{context_str}\n"
        "\\nn"
        "请根据新的文段，进一步完善你的回答。\n\n"
        "### Response: "
    )

    initial_qa_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "以下为背景知识：\n"
        "{context_str}"
        "\n"
        "请根据以上背景知识, 回答这个问题：{question}。\n\n"
        "### Response: "
    )
```

and modify the code around line 39:

```python
    from langchain import PromptTemplate
    refine_prompt = PromptTemplate(
        input_variables=["question", "existing_answer", "context_str"],
        template=refine_prompt_template,
    )
    initial_qa_prompt = PromptTemplate(
        input_variables=["context_str", "question"],
        template=initial_qa_template,
    )
    chain_type_kwargs = {"question_prompt": initial_qa_prompt, "refine_prompt": refine_prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="refine",
        retriever=retriever, return_source_documents= not args.hide_source,
        chain_type_kwargs=chain_type_kwargs)
```
