### ⚠️ 由于上游llama.cpp重大更新，目前LlamaChat暂时无法加载新版ggml模型。

LlamaChat: https://github.com/alexrozanski/LlamaChat

LlamaChat提供了一种面向macOS系统的类LLaMA模型的交互图形界面。下面以中文Alpaca 7B模型为例介绍相应的启动步骤。

⚠️ 注意：LlamaChat需要macOS 13 Ventura系统，配备英特尔CPU或者Apple M系列芯片。

### 第一步：下载最新版本的LlamaChat

选择最新版的`.dmg`文件即可。

链接：https://github.com/alexrozanski/LlamaChat/releases

### 第二步：安装LlamaChat

只需将LlamaChat拖入Applications即可（其他文件夹亦可）。

<img width="650" alt="image" src="https://user-images.githubusercontent.com/16095339/232356480-f545f93e-23f1-4b44-97d9-14c2a4722eee.png">

### 第三步：配置模型

按照向导配置模型即可。在本实例中，选择Alpaca模型。
<img width="620" alt="image" src="https://user-images.githubusercontent.com/16095339/232356597-22e75440-576b-42a4-a2ca-3d442ab28833.png">

给模型取一个名字，头像自行选择，下面的`Format`下拉列表选择PyToch格式（`.pth`扩展名）或者GGML格式（`.bin`扩展名），指定模型路径和模型大小。其中GGML格式就是llama.cpp中转换得到的模型格式，具体参考[llama.cpp转换](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/llama.cpp量化部署)。

⚠️ LlamaChat暂不支持最新的量化方法，例如Q5或者Q8。

<img width="620" alt="image" src="https://user-images.githubusercontent.com/16095339/232356838-1179e76c-e19d-4ffc-afdb-d2495eb5d657.png">

### 第四步：聊天交互

添加模型成功之后即可和模型进行交互。点击聊天框左侧的图标可以结束当前轮次的对话（清除上下文缓存，但不删除相应的聊天记录）。

（提示：生成第一句话时需要加载模型，之后速度就会恢复正常了）

<img width="1033" alt="image" src="https://user-images.githubusercontent.com/16095339/232357193-c4f5b1ac-437c-4fb3-9eb4-9a62743775d3.png">

### 自定义设置

v1.2版本支持了自定义参数，提供了更加灵活的解码方式，可根据实际情况自行修改。

<img width="1033" alt="image" src="https://user-images.githubusercontent.com/16095339/233646243-100ea9c5-8a59-4bca-a530-8afbf66ad98c.png">
