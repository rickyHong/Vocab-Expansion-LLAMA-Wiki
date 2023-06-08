In order to quickly evaluate the actual performance of related models, this project compared the effects of Chinese Alpaca-7B, Alpaca-13B, Alpaca-Plus-7B and Alpaca-Plus-13B on some common tasks given the same prompt. Reply generation is random and is affected by factors such as decoding hyperparameters and random seeds. The following related evaluations are not absolutely rigorous, and the test results are for reference only. Welcome to experience it yourself. For detailed evaluation results, please see [examples](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples).

| Tasks            | Samples | Alpaca-13B | Alpaca-Plus-7B | Alpaca-Plus-13B |
| ---------------- | :----: | :--------: | :------------: | :-------------: |
| **💯Overall**    |  200   |    74.3    |      78.2      |   **👍🏻80.8**    |
|  Question Answering         |   20   |     70     |       74       |    **👍🏻79**     |
| Open QA       |   20   |     77     |       77       |       77        |
| Computation, Reasoning  |   20   |     61     |       61       |       60        |
| Poetry, Literature, Philosophy |   20   |     65     |    **👍🏻76**    |    **👍🏻76**     |
| Music, Sports, Entertainment |   20   |     68     |       73       |    **👍🏻80**     |
| Letters and Articles     |   20   |     83     |       82       |    **👍🏻87**     |
| Translation         |   20   |     84     |       87       |    **👍🏻90**     |
| Multi-turn Dialogue         |   20   |     88     |       89       |       89        |
| Coding         |   20   |     65     |       64       |    **👍🏻70**     |
| Ethics       |   20   |     82     |    **👍🏻99**    |    **👍🏻100**    |

