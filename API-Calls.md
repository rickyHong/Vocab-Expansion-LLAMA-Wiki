> This document is a mirror of `scripts/openai_server_demo/README.md`, provided by @sunyuhan19981208 in this PR.

> For more detailed OPENAI API information, visit: <https://platform.openai.com/docs/api-reference>

This is a simple server DEMO implemented using fastapi that imitates the style of the OPENAI API. You can use this API DEMO to quickly build a personal website based on the large Chinese model and other interesting WEB DEMOs.

## Deployment Method

Install dependencies
``` shell
pip install fastapi uvicorn shortuuid
```

Start script
``` shell
python scripts/openai_server_demo/openai_api_server.py --base_model /path/to/base_model --lora_model /path/to/lora_model --gpus 0,1
```

### Parameter Description

`--base_model {base_model}`: The directory containing the LLaMA model weights and configuration files in HF format. It can be a merged Chinese Alpaca or Alpaca Plus model (in this case, `--lora_model` is not required), or the original LLaMA model in HF format after conversion (you need to provide `--lora_model`).

`--lora_model {lora_model}`: The directory where the Chinese Alpaca LoRA decompressed files are located, or the model call name using the 🤗 Model Hub. If this parameter is not provided, only the model specified by --base_model will be loaded.

`--tokenizer_path {tokenizer_path}`: The directory where the corresponding tokenizer is stored. If this parameter is not provided, its default value is the same as `--lora_model`; if the `--lora_model` parameter is also not provided, its default value is the same as --base_model.

`--only_cpu`: Use only CPU for inference.

`--gpus {gpu_ids}`: Specifies the GPU device number to use, default is 0. If using multiple GPUs, separate with commas, like 0,1,2.

`--load_in_8bit`: Use 8bit model for inference to save video memory, but may affect the model effect.

## API Documentation

### Text Relay (completion)

> For the Chinese translation of completion, Professor Li Hongyi translates it as text relay <https://www.youtube.com/watch?v=yiY4nPOzJEg>

The most basic API interface, input a prompt, and output the text relay (completion) result of the language large model.

The API DEMO has a built-in alpaca prompt template, and the prompt will be incorporated into the alpaca instruction template. The prompt input here should be more like an instruction rather than a dialogue.

#### Quick experience completion interface

Request command:

``` shell
curl http://localhost:19327/v1/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "prompt": "Tell me where the capital of China is"
  }'
```

json return body:

``` json
{
    "id": "cmpl-3watqWsbmYgbWXupsSik7s",
    "object": "text_completion",
    "created": 1686067311,
    "model": "chinese-llama-alpaca",
    "choices": [
        {
            "index": 0,
            "text": "The capital of China is Beijing."
        }
    ]
}
```

#### Advanced Parameters of the Completion Interface

Request command:

``` shell
curl http://localhost:19327/v1/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "prompt": "Tell me the strengths and weaknesses of China and the United States respectively",
    "max_tokens": 90,
    "temperature": 0.7,
    "num_beams": 4,
    "top_k": 40
  }'
```

json return body:

``` json
{
    "id": "cmpl-PvVwfMq2MVWHCBKiyYJfKM",
    "object": "text_completion",
    "created": 1686149471,
    "model": "chinese-llama-alpaca",
    "choices": [
        {
            "index": 0,
            "text": "The strengths of China lie in its rich culture and history, while the strengths of the United States lie in its advanced technology and economic system."
        }
    ]
}
```

#### Explanation of Advanced Parameters of the Completion Interface

> For more detailed information about Decoding strategies, you can refer to <https://towardsdatascience.com/the-three-decoding-methods-for-nlp-23ca59cb1e9d>. This article details three Decoding strategies that LLaMA will use: Greedy Decoding, Random Sampling, and Beam Search. These decoding strategies are the basis for advanced parameters such as top_k, top_p, temperature, num_beam, etc.

`prompt`: The prompt for generating the text relay (completion).

`max_tokens`: The token length of the newly generated sentence.

`temperature`: A sampling temperature between 0 and 2. Higher values such as 0.8 make the output more random, while lower values such as 0.2 make the output more deterministic. The higher the temperature, the greater the probability that random sampling will be used for decoding.

`num_beams`: This parameter is used in beam search when the search strategy is beam search. When num_beams=1, it is actually greedy decoding.

`top_k`: In random sampling, the top_k high-probability tokens will be sampled as candidate tokens.

`top_p`: In random sampling, tokens with a cumulative probability exceeding top_p will be sampled as candidate tokens. The lower it is, the more randomness there is. For example, when top_p is set to 0.6, if the probability of the top 5 tokens is [0.23, 0.20, 0.18, 0.11, 0.10], the cumulative probability of the top three tokens is 0.61, then the fourth token will be filtered out, and only the top three tokens will be sampled as candidate tokens.

`repetition_penalty`: Repetition penalty, for more details you can refer to this article: <https://arxiv.org/pdf/1909.05858.pdf>.

### Chat (Chat Completion)

The chat interface supports multi-turn conversations.

#### Quick Experience with the Chat Interface

Request command:

``` shell
curl http://localhost:19327/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "messages": [
      {"role": "user","message": "Tell me some stories about Hangzhou"}
    ],
    "repetition_penalty": 1.0
  }'
```

json return body:

``` json
{
    "id": "chatcmpl-5L99pYoW2ov5ra44Ghwupt",
    "object": "chat.completion",
    "created": 1686143170,
    "model": "chinese-llama-alpaca",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "user",
                "content": "Tell me some stories about Hangzhou"
            }
        },
        {
            "index": 1,
            "message": {
                "role": "assistant",
                "content": "Sure, do you have any specific preferences about Hangzhou?"
            }
        }
    ]
}
```

#### Multi-turn Conversation

Request command:

``` shell
curl http://localhost:19327/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{   
    "messages": [
      {"role": "user","message": "Tell me some stories about Hangzhou"},
      {"role": "assistant","message": "Sure, do you have any specific preferences about Hangzhou?"},
      {"role": "user","message": "I am quite fond of West Lake, could you tell me about it?"}
    ],
    "repetition_penalty": 1.0
  }'
```

json return body:

``` json
{
    "id": "chatcmpl-hmvrQNPGYTcLtmYruPJbv6",
    "object": "chat.completion",
    "created": 1686143439,
    "model": "chinese-llama-alpaca",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "user",
                "content": "Tell me some stories about Hangzhou"
            }
        },
        {
            "index": 1,
            "message": {
                "role": "assistant",
                "content": "Sure, do you have any specific preferences about Hangzhou?"
            }
        },
        {
            "index": 2,
            "message": {
                "role": "user",
                "content": "I am quite fond of West Lake, could you tell me about it?"
            }
        },
        {
            "index": 3,
            "message": {
                "role": "assistant",
                "content": "Indeed, West Lake is one of the most famous attractions in Hangzhou, it is known as 'Paradise on Earth'."
            }
        }
    ]
}
```

#### Explanation of Advanced Parameters of the Chat Interface

`prompt`: The prompt for generating the text relay (completion).

`max_tokens`: The token length of the newly generated sentence.

`temperature`: A sampling temperature between 0 and 2. Higher values such as 0.8 make the output more random, while lower values such as 0.2 make the output more deterministic. The higher the temperature, the greater the probability that random sampling will be used for decoding.

`num_beams`: This parameter is used in beam search when the search strategy is beam search. When num_beams=1, it is actually

 greedy decoding.

`top_k`: In random sampling, the top_k high-probability tokens will be sampled as candidate tokens.

`top_p`: In random sampling, tokens with a cumulative probability exceeding top_p will be sampled as candidate tokens. The lower it is, the more randomness there is. For example, when top_p is set to 0.6, if the probability of the top 5 tokens is [0.23, 0.20, 0.18, 0.11, 0.10], the cumulative probability of the top three tokens is 0.61, then the fourth token will be filtered out, and only the top three tokens will be sampled as candidate tokens.

`repetition_penalty`: Repetition penalty, for more details you can refer to this article: <https://arxiv.org/pdf/1909.05858.pdf>.

### Text Embeddings

Text embeddings have many uses, including but not limited to question answering based on large documents, summarizing the content of a book, finding the most similar memory for the large language model according to the current user input, and so on.

Request command:

``` shell
curl http://localhost:19327/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The weather is really nice today"
  }'
```

json return body:

``` json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                0.003643923671916127,
                -0.0072653163224458694,
                0.0075545101426541805, 
                ....,
                0.0045851171016693115
            ],
            "index": 0
        }
    ],
    "model": "chinese-llama-alpaca"
}
```

The length of the embedding vector is the same as the hidden size of the model used. For example, when using a 7B model, the length of the embedding is 4096.
