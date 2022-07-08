# pytorch_Chinese_Generate
基于pytorch的中文文本生成。

时下主流的预训练框架可以分为三种：
- autoregressive 自回归模型的代表是GPT，本质上是一个从左到右的语言模型，常用于无条件生成任务（unconditional generation）。
- autoencoding 自编码模型是通过某个降噪目标（如掩码语言模型）训练的语言编码器，如BERT、ALBERT、DeBERTa。自编码模型擅长自然语言理解任务（natural language understanding tasks），常被用来生成句子的上下文表示。
- encoder-decoder 则是一个完整的Transformer结构，包含一个编码器和一个解码器，以T5、BART为代表，常用于有条件的生成任务 （conditional generation）。

这里相当于是将bert改造为一种encoder-decode结构。

### 参考
> 理论：https://zhuanlan.zhihu.com/p/532851481
