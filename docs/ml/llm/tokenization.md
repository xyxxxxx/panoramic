# 分词

!!! abstract "参考"
    * [Youtube - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

Karpathy：“分词是 LLM 最无趣的部分，并且其中有很多坑，但是我们绕不开它 :(”。

## tokenizer 和 LLM 的关系

![](https://s2.loli.net/2024/02/29/7QLCunFrbpgsGMk.png)

tokenizer 负责编码 Unicode 码点序列（文本）为 token 序列，和解码 token 序列为 Unicode 码点序列（文本），这两种序列都可以被视为整数序列。LLM 只处理 token 序列，它不知道任何 token 序号所对应的 Unicode 码点（序列）或字节（序列）。

训练时先训练 tokenizer，使用 BPE 算法和单独的训练数据集；再训练 LLM，嵌入表规模应与词汇表规模一致。

## tokenizer 的代码实现

下面的 Python 代码实现了 BPE 算法：

```python
# 训练数据
text = 'Long long text...'
tokens = text.encode('utf-8')    # 字节串
tokens = list(tokens)            # 转换为范围在 0 到 255 之间的整数列表

# 训练 tokenizer
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """替换 ids 中出现的所有相邻的 pair 为 idx"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else: 
            new_ids.append(ids[i])
            i += 1
    return new_ids

VOCAB_SIZE = 500
num_merges = VOCAB_SIZE - 256
ids = tokens.copy()

merges = {}                                        # 合并规则
vocab = {idx: bytes([idx]) for idx in range(256)}  # 词汇表
for i in range(num_merges):
    stats = get_stats(ids)
    top_pair = max(stats, key=stats.get)  # `max(stats)` 返回最大的键
                                          # `key=stats.get` 表示取相应键的值进行排序
    idx = 256 + i
    ids = merge(ids, top_pair, idx)
    merges[top_pair] = idx
    vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

# 编码
def encode(text):
    ids = list(text.encode('utf-8'))

    # 简明的实现
    # for pair, idx in merges.items():
    #     ids = merge(ids, pair, idx)

    # fancy 的实现
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))  # 获取最优先合并的 pair
        if pair not in merges:
            break
        idx = merges[pair]
        ids = merge(ids, pair, idx)

    return ids

# 解码
def decode(ids):
    tokens = b''.join([vocab[idx] for idx in ids])
    text = tokens.decode('utf-8', errors='replace')
    return text
```

!!! info "信息"
    训练 tokenizer 完成后，保存合并规则和词汇表即可进行后续编解码：前者用于编码，后者用于解码。

!!! question "为什么使用 UTF-8 编码？"
    从 UTF-8 到 UTF-16 再到 UTF-32，编码相同的 Unicode 码点/字符串得到的字节串长度增加，为了将相同的 token 添加到词汇表所需要的合并次数也增加。这意味着过渡 token（不能被解码）的数量增加，从而挤占非过渡 token 的空间；tokenizer 需要处理更长的字节串，进行更多的合并操作，从而损害性能。

[GPT-2 的论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)中写道：……然而，直接对字节序列应用 BPE 会由于 BPE 使用基于贪婪频率的启发式方法构建词汇表，导致不理想的合并。我们观察到 BPE 包含了许多常见词的不同版本，比如 `dog`，因为它们以多种变体出现，如 `dog`、`dog!`、`dog?` 等。这导致有限的词汇表空间和模型容量被次优地分配。为了避免这种情况，我们阻止 BPE 跨字符类别合并任何字节序列。我们将空格作为一个例外，这显著提高了压缩效率，同时只增加了最小的词的碎片化。（总结：空格前缀能够极大地提升英文文本的压缩效率，因而保留；标点符号后缀对于压缩效率的提升不大，因而移除。）

在 [tokenizer 的推理代码](https://github.com/openai/gpt-2/blob/master/src/encoder.py)中，文本按照以下正则表达式模式被拆分为多个子串，每个子串分别编码，最后再将结果拼接起来。这样保证了合并不会跨越模式的边界，从而达成了上述目标。训练数据也应该按照相同的模式被拆分，否则 tokenizer 学习了跨越模式边界的合并却使用不上，造成词汇表空间的浪费。然而 GPT-2 及其 tokenizer 的训练代码未被开源，具体细节尚不清楚。

```python
import regex as re

gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
re.findall(gpt2pat, 'Long long text...')
```

对于该模式解释如下：

* `'s|'t|'re|'ve|'m|'ll|'d`：匹配英文缩写。（然而该模式并不完备，未考虑到使用另一种撇号 `’` 或字母大写的情况）
* ` ?\p{L}+`：匹配一个或多个文字（可以来自任何语言），前面可能有一个空格。
* ` ?\p{N}+`：匹配一个或多个数字，前面可能有一个空格。
* ` ?[^\s\p{L}\p{N}]+`：匹配一个或多个非空白字符、非文字、非数字的字符（即标点符号或特殊符号），前面可能有一个空格。
* `\s+(?!\S)`：匹配一个或多个空白字符，但后面不能有一个非空白字符（即不匹配最后一个空白字符）。
* `\s+`：匹配末尾的一个或多个空白字符。

GPT-3 及之后的模型都未被开源，但我们仍然可以在[这里](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)窥见它们的正则表达式模式，例如 GPT-4 的模式如下：

```python
r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

对于该模式解释如下：

* `'(?i:[sdmt]|ll|ve|re)`：匹配英文缩写，非捕获组 `(?i: ... )` 表示忽略大小写匹配。（仍未考虑到使用另一种撇号 `’` 的情况）
* `[^\r\n\p{L}\p{N}]?+\p{L}+`：非贪婪地匹配一个或多个非文字、非数字、非回车符、非换行符的字符，再匹配一个或多个文字。
* `\p{N}{1,3}`：匹配 1-3 个数字字符。
* ` ?[^\s\p{L}\p{N}]++[\r\n]*`：匹配一个或多个非空白字符、非文字、非数字的字符（即标点符号或特殊符号），前面可能有一个空格，后面可能有任意个回车符或换行符。
* `\s*[\r\n]`：匹配任意个空白字符，后面有一个回车符或换行符。
* `\s+(?!\S)`：同上。
* `\s+`：同上。

## 特殊 token

一些特殊 token 可能会被手动添加（而不是通过 BPE 算法）到词汇表中，用于对输入和输出 token 序列进行控制。tokenizer 需要单独识别这些特殊 token。

下面是一些常见的特殊 token：

* tiktoken：
    * `<|endoftext|>`：文本结束标记，用于标识模型训练数据或生成文本的结束位置。
    * `<|im_start|>`：对话中输入消息（input message）的开始标记，通常后面会跟角色名称，例如 `<|im_start|>system`、`<|im_start|>user`、`<|im_start|>assistant`。
    * `<|im_end|>`：对话中输入消息的结束标记。
    * `<|fim-prefix|>`：FIM（Fill-in-the-Middle）任务中的前缀标记，用于标识需要填充的文本的上文起始位置。
    * `<|fim-middle|>`：FIM 任务中的中间部分标记，用于标识需要填充的文本的位置。
    * `<|fim-suffix|>`：FIM 任务中的后缀标记，用于标识需要填充的文本的下文结束位置。
    * `<|fim-pad|>`：FIM 任务中的填充标记，用于对齐或填充文本。
    * `<|object_ref_start|>` 和 `<|object_ref_end|>`：用于标识对象引用的开始和结束。
    * `<|box_start|>` 和 `<|box_end|>`：用于标识边界框（bounding box）的开始和结束。
    * `<|quad_start|>` 和 `<|quad_end|>`：用于标识四边形区域的开始和结束。
    * `<|vision_start|>` 和 `<|vision_end|>`：用于标识视觉数据的开始和结束。
    * `<|vision_pad|>`：视觉数据的填充标记。
    * `<|image_pad|>`：图像数据的填充标记。
    * `<|video_pad|>`：视频数据的填充标记。
    * `<tool_call>` 和 `</tool_call>`：用于标记工具调用的开始和结束。
* sentencepiece：
    * `<unk>`：表示未知。
    * `<s>`：标识序列开始。
    * `</s>`：标识序列结束。
    * `<pad>`：表示填充。

## 分词工具

### tiktoken

!!! abstract "参考"
    * [在线 tiktoken 分词](https://tiktokenizer.vercel.app/)

[tiktoken](https://github.com/openai/tiktoken) 是一个快速的 BPE tokenizer，用于 OpenAI 的 LLM。

不同文本模态的分词结果呈现以下特点：

* 英文：分词效果最好，常见词可以被划分为单个 token，词缀可以被正确划分。但是同一个单词前方有无空格、（每个字母）大写或小写都会被识别为不同的 token。
* 非英文：分词效果不如英文。因为 tokenizer 训练集中的英文语料最多，得到的词汇表中的英文 token 也最多（因而 dense token 较多），非英文的 token 则较少（因而 dense token 较少）。这导致对于表示相同语义的英文文本和非英文文本，非英文文本分词后的 token 数量更多，Transformer 的计算量更大，正确 attend 的难度更大；语义更零碎（一个语义可能被拆散到更多 token 中），Transformer 正确拼接的难度也更大；对于有限上下文长度的利用效率也更低。
* 算术：除单个数字字符外，数字被划分为多少个 token 是相当随机的，Transformer 想要正确执行算术运算必须正确拼接这些数字。
* 代码：

tokenizer 的迭代呈现以下特点：

* 新的 tokenizer 词汇表规模更大。编码相同的文本，新的 tokenizer 产生的 token 数量更少。
* 针对代码有所改进，例如手动添加了 3/7/11/… 个空格的 token。

!!! note "注意"
    GPT-2 的词汇表规模约为 50k，GPT-4 约为 100k，当前实践普遍将词汇表规模定在 50-100k 区间。进一步扩大词汇表规模不一定能提升 LLM 的表现，因为：
    
    * 嵌入表的规模扩大，使得 LLM 的计算量增加。
    * 在 LLM 的训练数据中出现频率低（甚至从未出现过）的 token 对应的向量可能 undertrained。请参阅 [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)。
    * 一些 token 过于 dense，LLM 正确处理的难度增加。

下列 LLM 采用 tiktoken：

* GPT 系列
* Pythia 系列
* Qwen 系列

### sentencepiece

[sentencepiece](https://github.com/google/sentencepiece) 能够高效地训练和推理 BPE tokenizer，用于 Llama 和 Mistral 系列模型。

和 tiktoken 的主要区别在于，tiktoken 先对文本进行 UTF-8 编码，再对字节串进行 BPE 编码，而 sentencepiece 直接对文本（Unicode 码点序列）进行 BPE 编码。对于罕见的 Unicode 码点，可选择将其映射到特殊 token `<|unk|>`，或对其进行 UTF-8 编码，转换为字节 token（序列）。

一个完整的示例请参阅[这里](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing#scrollTo=rSv1vfIVOhvr)。

Llama tokenizer 的分词结果呈现以下特点：

* 算术：每个数字字符一个 token，Transformer 想要正确执行算术运算必须正确拼接这些数字字符。

下列 LLM 采用 sentencepiece：

* Llama 系列
* Mistral 系列
* ChatGLM 系列

## 展望

* 一种方案是使用 Unicode（目前 ~150k 个码点）作为初始词汇表，应用 BPE 进一步合并，最终的词汇表规模定在 200-250k 区间。（这一方案曾在 GPT-2 的论文中遭否决，但如今这一方案完全可行）
* 一种研究方向是构建不需要分词的 byte-level LM 架构，例如 [2305.07185](https://arxiv.org/abs/2305.07185)。
