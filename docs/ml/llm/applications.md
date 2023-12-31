# 应用

LLM 应用广泛，但不同的应用类型之间并没有明显的界限。这里选取一些主题进行讨论。

## RAG

RAG（retrieval-augmented generation）这一概念由论文 [[2005.11401](https://arxiv.org/abs/2005.11401)] 提出，其为预训练的参数化记忆生成模型（预训练 transformer）赋予了非参数化记忆（向量索引），并将其用在知识密集型的任务上。具体方法如下：

![](../../assets/ml/llm/rag.png)

该方法本来是一个微调方法，对组合架构进行端到端的训练。亦即，使用成对的 QA 数据，最小化 $\sum_i -\log p(y_i|x_i)$，来同时微调生成回复的 LM（BART）和编码/嵌入查询文本的 LM（BERT）（若要微调编码/嵌入文档的 LM（BERT），则需要定期重新编码/嵌入文档，开销较大，原论文发现其对于模型表现提升不大，于是固定其参数）。

对于外挂知识库的优点，原论文提到：可以直接扩展或修改（位于向量索引的）知识，可以对被检索到的知识作进一步的检查，减少 LLM 的幻觉。

如今的知识库问答应用都是基于这一方法，但不进行训练，只进行推理。典型应用：

* [quivr](https://github.com/StanGirard/quivr)
* [DocsGPT](https://github.com/arc53/DocsGPT)

## 聊天机器人

用结构化的 prompt 引导 LLM 生成聊天内容。

微调以使得 LLM 对齐。

典型应用：

* ChatGPT
* Claude
* Bard
* ……（不计其数）

## 智能体

智能体、代理、助手

ReAct

AutoGen
