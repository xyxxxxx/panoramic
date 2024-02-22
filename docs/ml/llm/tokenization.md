# 分词

!!! info "参考"
    * [Youtube - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

Karpathy：“分词是 LLM 最无趣的部分，并且其中有很多坑，但是我们绕不开它 :(”。

## 训练 tokenizer（BPE 算法）

## 观察分词结果

### tiktoken

!!! info "参考"
    * [在线 tiktoken 分词](https://tiktokenizer.vercel.app/)

tiktoken 是一个快速的 BPE 分词器，用于 OpenAI 的 LLM。

不同文本模态的分词结果呈现以下特点：

* 英文：分词效果最好，常见词可以被划分为单个 token，词缀可以被正确划分。但是同一个单词有无空格、（每个字母）大写或小写都会被识别为多个 token。
* 非英文：分词效果不如英文。因为 tokenizer 训练集中的英文语料最多，得到的词汇表中的英文 token 也最多，非英文的 token 则较少。这导致对于表示相同语义的英文文本和非英文文本，非英文文本分词后的 token 数量更多，Transformer 的计算量更大，正确 attend 的难度更大；语义更零碎（一个语义可能被拆散到更多 token 中），Transformer 正确拼接的难度也更大；对于有限上下文长度的利用效率也更低。
* 算术：除单个数字字符外，数字被划分为多少个 token 是相当随机的，Transformer 想要正确执行算术运算必须正确拼接这些数字。
* 代码：

tokenizer 的迭代呈现以下特点：

* 新的 tokenizer 词汇表规模更大。编码相同的文本，新的 tokenizer 产生的 token 数量更少。
* 对于代码有所改进，例如手动添加了 3/7/11/… 个空格的 token。

!!! note "注意"
    GPT-2 的词汇表规模约为 50k，GPT-4 约为 100k，当前实践普遍将词汇表规模定在 50-100k 区间。进一步扩大词汇表规模不一定能继续提升 LLM 的性能，因为嵌入表的规模也扩大，训练难度增加。需要为词汇表规模找到一个平衡点。
