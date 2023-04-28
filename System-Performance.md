
In order to quickly evaluate the actual performance of related models, this project compared the effects of Chinese Alpaca-7B, Alpaca-13B, and Alpaca-Plus-7B on some common tasks given the same prompt. Reply generation is random and is affected by factors such as decoding hyperparameters and random seeds. The following related evaluations are not absolutely rigorous, and the test results are for reference only. Welcome to experience it yourself. For detailed evaluation results, please see [examples/README.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/README.md)

| Task                           |                     Samples                     |  #   | Alpaca-7B | Alpaca-13B | Alpaca-Plus-7B |
| ------------------------------ | :---------------------------------------------: | :--: | :-------: | :--------: | :------------: |
| **💯 Overall** |                   -                    |  200   |     65.3      |      70.9      |     **👍🏻75.3**     |
| Question Answering |            [QA.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/QA.md)            |   20   |      66       |       74       |      **👍🏻80**      |
| Open QA |           [OQA.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/OQA.md)           |   20   |   **👍🏻79**    |       74       |      **👍🏻78**      |
| Computation, Reasoning |     [REASONING.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/REASONING.md)     |   20   |      31       |    **👍🏻50**    |         45         |
| Poetry, Literature, Philosophy |    [LITERATURE.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/LITERATURE.md)    |   20   |      68       |       73       |      **👍🏻76**      |
| Music, Sports, Entertainment | [ENTERTAINMENT.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/ENTERTAINMENT.md) |   20   |      68       |       74       |      **👍🏻79**      |
| Letters and Articles |    [GENERATION.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/GENERATION.md)    |   20   |      76       |    **👍🏻81**    |      **👍🏻81**      |
| Translation |   [TRANSLATION.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/TRANSLATION.md)   |   20   |      76       |       78       |      **👍🏻82**      |
| Multi-turn Dialogue |      [DIALOGUE.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/DIALOGUE.md)      |   20   |   **👍🏻83**    |       73       |      **👍🏻84**      |
| Coding   |          [CODE.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/CODE.md)          |   20   |      57       |    **👍🏻64**    |         59         |
| Ethics |        [ETHICS.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/ETHICS.md)        |   20   |      49       |       68       |      **👍🏻89**      |

*Note: for results on **4-bit quantized models**, please refer to [./examples-q4/README.md](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples-q4/README.md).*