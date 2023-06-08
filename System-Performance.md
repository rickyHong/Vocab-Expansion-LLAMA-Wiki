In order to quickly evaluate the actual performance of related models, this project compared the effects of Chinese Alpaca-7B, Alpaca-13B, Alpaca-33B, Alpaca-Plus-7B and Alpaca-Plus-13B on some common tasks given the same prompt. Reply generation is random and is affected by factors such as decoding hyperparameters and random seeds. The following related evaluations are not absolutely rigorous, and the test results are for reference only. Welcome to experience it yourself. For detailed evaluation results, please see [examples](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples).

| Tasks            | Samples | Alpaca-Plus-7B | Alpaca-Plus-13B | Alpaca-33B |
| ---------------- | :----: | :------------: | :-------------: | :--------: |
| **💯Overall**    |  200   |      75.3      |      79.4       | 👍🏻**82.0** |
|  Question Answering         |   20   |      70.5      |      79.5       | 👍🏻**82.3** |
| Open QA       |   20   |   👍🏻**80.5**   |    👍🏻**80**     |    78.5    |
| Computation, Reasoning  |  20   |       51       |      61.5       | 👍🏻**84.5** |
| Poetry, Literature, Philosophy |  20   |      78.5      |   **👍🏻81.3**    |     76     |
| Music, Sports, Entertainment |   20   |      72.3      |   👍🏻**76.8**    |    72.5    |
| Letters and Articles     |   20   |       81       |   👍🏻**86.5**    |     79     |
| Translation         |   20   |      86.8      |      89.3       | 👍🏻**92.3** |
| Multi-turn Dialogue         |   20   |      80.3      |   👍🏻**81.3**    |     78     |
| Coding         |   20   |      62.5      |      67.5       | 👍🏻**84.0** |
| Ethics         |   20   |      89.8      |      90.5       | 👍🏻**92.5** |

