### ⚠️ Due to breaking changes in llama.cpp, LlamaChat cannot load the latest ggml models.

LlamaChat: https://github.com/alexrozanski/LlamaChat

LlamaChat provides an interactive graphical interface for LLaMA-like models on macOS systems. The following instructions demonstrate the setup process using the Chinese Alpaca 7B model as an example.

⚠️ Warning: LlamaChat requires macOS 13 Ventura, and either an Intel or Apple Silicon processor.

### Step 1: Download the latest version of LlamaChat

Choose the latest .dmg file.

Link: https://github.com/alexrozanski/LlamaChat/releases

### Step 2: Install LlamaChat

Simply drag LlamaChat into the Applications folder (or any other folder).

<img width="650" alt="image" src="https://user-images.githubusercontent.com/16095339/232356480-f545f93e-23f1-4b44-97d9-14c2a4722eee.png">

### Step 3: Configure the model

Follow the wizard to configure the model. In this example, choose the Alpaca model.
<img width="620" alt="image" src="https://user-images.githubusercontent.com/16095339/232356597-22e75440-576b-42a4-a2ca-3d442ab28833.png">

Name the model, choose an avatar, and select the appropriate format from the Format dropdown list: PyTorch format (.pth extension) or GGML format (.bin extension). Specify the model path and model size. The GGML format is the model format obtained from converting with llama.cpp. For more information, refer to [llama.cpp conversion](./llama.cpp-Deployment).

⚠️ LlamaChat does not support the latest quantization format, such as Q5, Q8.

<img width="620" alt="image" src="https://user-images.githubusercontent.com/16095339/232356838-1179e76c-e19d-4ffc-afdb-d2495eb5d657.png">

### Step 4: Chat!

After successfully adding the model, you can interact with it. Click on the icon to the left of the chat box to end the current round of conversation (this clears the context cache but does not delete the corresponding chat history).

(Note: Loading the model is necessary for generating the first sentence, but the speed will return to normal afterward.)

<img width="1033" alt="image" src="https://user-images.githubusercontent.com/16095339/232357193-c4f5b1ac-437c-4fb3-9eb4-9a62743775d3.png">


### Settings

LlamaChat v1.2 introduces customized settings for inference. Take a look and try.

<img width="1033" alt="image" src="https://user-images.githubusercontent.com/16095339/233646243-100ea9c5-8a59-4bca-a530-8afbf66ad98c.png">
