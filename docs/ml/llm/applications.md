# 应用

LLM 应用广泛，但不同的应用类型之间并没有明显的界限。这里选取一些主题进行讨论。

## RAG

RAG（retrieval-augmented generation）这一概念由论文 [[2005.11401](https://arxiv.org/abs/2005.11401)] 提出，其为预训练的参数化记忆生成模型（预训练 transformer）赋予了非参数化记忆（向量索引），并将其用在知识密集型的任务上。具体方法如下：

![](../../assets/ml/llm/rag.png)

该方法本来是一个微调方法，对组合架构进行端到端的训练。亦即，使用成对的 QA 数据，最小化 $\sum_i -\log p(y_i|x_i)$，来同时微调生成回复的 LM（BART）和编码查询文本的 LM（BERT）（若要微调编码文档的 LM（BERT），则需要定期重新编码文档，开销较大，原论文发现其对于模型表现提升不大，于是固定其参数）。

对于外挂知识库的优点，原论文提到：可以直接扩展或修改（位于向量索引的）知识；可以对被检索到的知识作进一步的检查，减少 LLM 的幻觉。

如今的知识库问答应用都是基于这一方法，但不进行训练，只进行推理。典型应用：

* [quivr](https://github.com/QuivrHQ/quivr)
* [DocsGPT](https://github.com/arc53/DocsGPT)
* [perplexity](https://www.perplexity.ai/)
* [Search with Lepton](https://github.com/leptonai/search_with_lepton)

## 聊天机器人

用结构化的模板引导 LLM 生成与用户聊天的内容。例如下面展示了传入 Llama-2-7b-chat-hf 模型的 token 序列：

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
>>> tokenizer.use_default_system_prompt = False
>>> conversation = [
{"role": "system", "content": "You are a helpful assistant."}
{"role": "user", "content": "Hello, how are you?"},
{"role": "assistant", "content": "I'm doing great. How can I help you today?"},
{"role": "user", "content": "I'd like to show off how chat templating works!"},
 ]
>>> tokenizer.apply_chat_template(conversation, tokenize=False)
"<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nHello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]"
>>> tokenizer.apply_chat_template(conversation, return_tensors="pt")
tensor([[    1,   518, 25580, 29962,  3532, 14816, 29903,  6778,    13,  3492,
           526,   263,  8444, 20255, 29889,    13, 29966,   829, 14816, 29903,
          6778,    13,    13, 10994, 29892,   920,   526,   366, 29973,   518,
         29914, 25580, 29962,   306, 29915, 29885,  2599,  2107, 29889,  1128,
           508,   306,  1371,   366,  9826, 29973, 29871,     2,     1,   518,
         25580, 29962,   306, 29915, 29881,   763,   304,  1510,  1283,   920,
         13563,  1350,   572,  1218,  1736, 29991,   518, 29914, 25580, 29962]])

# 其中：
# * INST 指 instruction，[INST] 和 [/INST] 标签包裹用户消息
# * <s> 和 </s> 标签包裹用户和 LLM 的一个聊天回合
# * <<SYS>> 和 <</SYS>> 标签包裹系统消息，放在第一个用户消息前，[INST] 标签内
# * 只有 <s> 和 </s> 是特殊 token
```

通过 RLHF 微调使 LLM 对齐。

典型应用：

* ChatGPT
* Claude
* Gemini
* ……（不计其数）

### 提示工程

!!! info "参考"
    * [提示工程指南](https://www.promptingguide.ai)

在与 LLM 聊天（使用 LLM 生成文本）时，输入的格式和内容是自由的，而输入会影响 LLM 的行为（输出的分布）。精心或巧妙构建出的提示词（prompt）可以让 LLM 更加贴合用户的需求，提升特定能力，或处理特定形式的任务等。

提示工程是 LLM 领域的一个新兴的子学科。一些人认为提示工程最终会消失；我认为提示工程本身作为优化模型表现的方法总会存在：输入不同，输出的分布必然不同，若对其计算量化指标，则必然有高有低，总会有优化的空间。与此同时，模型本身的能力、应用/系统的工作流也会影响模型表现，应结合实际需求，综合考虑不同部分的优化难度、优化效果（这些仍在发展中）选择方案。

但不应让用户自己来做提示工程，这会损害使用体验，自然的提示词（类似人与人的交流）一定是体验最好的。

下面是一些 prompting 方法：

* zero-shot（基线）
* few-shot（出自 GPT-3 的论文）[[2005.14165](https://arxiv.org/abs/2005.14165)]
* generate knowledge prompting（LLM 先生成与问题相关的知识，再参考其进行回复）[[2110.08387](https://arxiv.org/abs/2110.08387)]
* Chain-of-Thought（CoT）[[2201.11903](https://arxiv.org/abs/2201.11903)]
* self-consistency（采样多个答案，选取最一致的答案）[[2203.11171](https://arxiv.org/abs/2203.11171)]
* Auto-CoT（聚类选出多样的问题，LLM 对它们生成 CoT 过程）[[2210.03493](https://arxiv.org/abs/2210.03493)]
* Re3（生成长篇故事）[[2210.06774](https://arxiv.org/abs/2210.06774)]
* automatic prompt engineer（提供输入输出示例，LLM 生成多个 prompt，评估选出效果最好的）[[2211.01910](https://arxiv.org/abs/2211.01910)]
* active prompt（选出 LLM 多次回复最不一致的问题，人类标注 CoT 过程）[[2302.12246](https://arxiv.org/abs/2302.12246), [2305.08291](https://arxiv.org/abs/2305.08291)]
* Tree-of-Thought(ToT)（将思维链扩展为思维树，LLM 每一步先生成几个备选答案，再检查每个答案是否正确；可以使用深度或广度优先的搜索策略）[[2305.10601](https://arxiv.org/abs/2305.10601)]
* Graph-of-Thought(GoT)（将思维树扩展为思维图，）[[2308.09687](https://arxiv.org/abs/2308.09687)]
* emotion prompt（对模型进行情绪勒索）[[2307.11760](https://arxiv.org/abs/2307.11760)]

!!! note "注意"
    prompting 方法对于更新、更强的 LLM 可能会失效。

下面是一些 prompting 小窍门：

* 对 LLM 礼貌没用，但说无妨。
* 尽量使用肯定的指示（“做什么”）而不是否定的指示（“不要做什么”）。
* 对 LLM 说“我会为一个更好的答案付x元小费”是有用的。
* 对 LLM 说“（如果你不能完成任务）你将会受到惩罚”是有用的。

提示注入（或提示攻击）是提示工程的一种恶意利用，其通过巧妙的提示诱导 LLM 产生有害或不希望的输出（例如变更任务、突破限制、泄露机密或隐私信息等）。

## 智能体

对于智能体（agent）的定义，就和对于通用人工智能（AGI）的定义一样莫衷一是。综合现有的观点，智能体应能够：

* 独立地（不需要人为干预）完成一项任务
* 完成多阶段的任务
* 自主地（预判用户的需求）完成任务
* 使用工具

下面是用于构建 LLM 智能体的框架：

* ReAct（）[]
* AutoGen（多智能体对话）[]

### 工具使用

下列工具可以提供给 LLM 使用：

* 计算器
* 程序运行环境
* 向量数据库、搜索引擎（RAG）
* 其他模型，例如图像/语音/视频生成模型
* 其他应用程序

下面是一些让 LLM 使用工具的实践：

* WebGPT（）[[2112.09332](https://arxiv.org/abs/2112.09332)]
* Toolformer（）[[2302.04761](https://arxiv.org/abs/2302.04761)]
* any tool（）[[2402.04253](https://arxiv.org/abs/2402.04253)]
