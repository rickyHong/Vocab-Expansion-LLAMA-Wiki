（正在建设中，请勿参考）
[privateGPT](https://github.com/imartinez/privateGPT) 是基于[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)和[LangChain](https://github.com/hwchase17/langchain)等的一个开源项目，旨在提供本地化文档分析并利用大模型来进行交互问答的接口。用户可以利用privateGPT对本地文档进行分析，并且利用GPT4All或llama.cpp兼容的大模型文件对文档内容进行提问和回答，确保了数据本地化和私有化。

由于本项目是基于LLaMA的相关衍生模型，本文以llama.cpp种的GGML格式模型为例介绍privateGPT的使用方法。

更详细的内容和用法请参考privateGPT官方目录：https://github.com/imartinez/privateGPT

### 前置准备：安装llama-cpp-python

由于privateGPT中使用了llama.cpp中的GGML模型，这里需要提前安装llama-cpp-python扩展。以下安装方式并没有使用任何加速库。

```bash
pip install llama-cpp-python
```

如果希望安装OpenBLAS/cuBLAS/CLBlast适配的版本，请参考：https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast

#### Mac M系列芯片用户必读

**必须确保当前安装环境中的python是支持arm64架构的，否则执行速度会慢10x以上。**
测试方法是安装llama-cpp-python之后，执行下述python命令，其中模型路径请替换成你本地的llama.cpp支持的GGML模型文件。

```
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="./models/7B/ggml-model.bin")
```

显示`NEON = 1`则表示正常，`NEON = 0`则表示并没有按arm64架构正确安装。下面给出的是支持ARM NEON加速的日志示例。

```
system_info: n_threads = 8 / 10 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 0 | VSX = 0 |
```

#### 如何安装arm64适配的python？

如果你使用conda，则可以使用以下命令创建相关环境，这里选择Python 3.10以满足privateGPT的要求。

```bash
CONDA_SUBDIR=osx-arm64 conda create -n privategpt python=3.10 -c conda-forge
```

### Step 1: 克隆目录并安装依赖包

在**正确安装**llama-cpp-python之后，则可以继续安装privateGPT，具体命令如下（注意python>=3.10）。

```bash
git clone https://github.com/imartinez/privateGPT.git
cd privateGPT
pip3 install -r requirements.txt
```

### Step 2: 修改配置文件

在privateGPT根目录下创建一个名为`.env`的配置文件，写好的配置文件示例：

```
MODEL_TYPE=LlamaCpp
PERSIST_DIRECTORY=db
MODEL_PATH=/your-path-to-ggml-model/ggml-model-q5_1.bin
MODEL_N_CTX=2048
EMBEDDINGS_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

- MODEL_TYPE：填写LlamaCpp
- PERSIST_DIRECTORY：填写分析文件存放位置，这里会在privateGPT根目录创建一个`db`目录
- MODEL_PATH：指向大模型存放位置，这里指向的是llama.cpp支持的GGML文件
- MODEL_N_CTX：大模型的最大token限制，设置为2048
- EMBEDDINGS_MODEL_NAME：SentenceTransformers词向量模型位置，可以指定HuggingFace上的路径（会自动下载），其他官方支持的模型可参考：https://www.sbert.net/docs/pretrained_models.html

### Step 3: 分析本地文件

privateGPT支持以下常规文档格式分析，例如（仅列举了最常用的）：

- Word文件：`.doc`，`.docx`
- PPT文件：`.ppt`，`.pptx`
- PDF文件：`.pdf`
- 纯文本文件：`.txt`
- CSV文件：`.csv`

- Markdown文件：`.md`
- 电子邮件文件：`.eml`，`.msg`

将需要分析的文档（不限于单个文档）放到privateGPT根节点下的`source_documents`目录下。这里以[本项目的LangChain示例数据](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/langchain_demo/doc.txt)为例进行介绍。目录结构类似：

```
> ls source_documents
doc.txt
```

下一步，运行ingest命令对文档进行分析。

```
python ingest.py
```

输出如下（笔者使用M1 Max，解析只经历了几秒钟）。需要注意的是首次使用会下载（如果给出的是huggingface地址，而不是本地路径）配置文件中的词向量模型。

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

⚠️ 注意：如果`db`目录中已经有相关分析文件，则会对数据文件进行积累。如果只想针对当前文档进行解析，请清空`db`目录后再ingest。

### Step 4: 在本地对文档进行提问

在上一步分析文档结束后，可运行以下命令开始对文档提问：

```
python privateGPT.py
```

出现以下提示之后即可输入问题：

```
Enter a query: 
```

例如询问`李白的诗是什么风格？`，结果如下：

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

整个过程并不是非常快，大概等了半分钟左右给出了相关结果，并且会给出4个数据来源。

输入`exit`则可结束脚本运行。

#### 高级配置

`privateGPT.py`实际是调用了llama-cpp-python的接口，因此如果不做任何代码修改则采用的默认解码策略。打开`privateGPT.py`查找以下语句（大约30-35行左右，根据不同版本有所不同）。

```python
llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
```

这里即是LlamaCpp模型的定义，可根据llama-cpp-python的接口定义传入更多自定义参数，以下是其中一个示例，额外增加了解码线程数量，有助于提升解码速度（请根据实际物理核心数酌情配置）。

```python
llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, n_threads=8)
```

在上述定义后几行会使用LangChain的RetrievalQA进行交互，具体定义和配置方式请参考LangChain的文档。

```
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
```

