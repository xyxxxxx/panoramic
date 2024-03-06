# 训练和推理

## 预训练

## 微调

### 有监督微调

### RLHF

PPO

DPO

### RLAIF

## 提示工程

!!! info "参考"
    * [提示工程指南](https://www.promptingguide.ai)

在与 LLM 聊天（使用 LLM 生成文本）时，输入的格式和内容是自由的，而输入会影响 LLM 的行为（输出的分布）。精心或巧妙构建出的提示词可以让 LLM 更加贴合用户的需求，提升特定能力，或处理特定形式的任务等。

提示工程是 LLM 领域的一个新兴的子学科。一些人认为提示工程最终会消失；我认为提示工程本身作为优化模型表现的方法总会存在：输入不同，输出的分布必然不同，若对其计算量化指标，则必然有高有低，总会有优化的空间。与此同时，模型本身的能力、应用/系统的工作流也会影响模型表现，应结合实际需求，综合考虑不同部分的优化难度、优化效果（这些仍在发展中）选择方案。

但不应让用户自己来做提示工程，这会损害使用体验，自然的提示词（类似人与人的交流）一定是体验最好的。

下面是一些 prompting 方法：

* zero-shot（基线）
* few-shot
* Chain-of-Thought(CoT)[[2201.11903]](https://arxiv.org/abs/2201.11903)
* Auto-CoT（聚类选出多样的问题，LLM 对它们生成 CoT 过程）[[2210.03493](https://arxiv.org/abs/2210.03493)]
* generate knowledge prompting（LLM 先生成与问题相关的知识，再参考其进行回复）[[2110.08387](https://arxiv.org/abs/2110.08387)]
* Tree-of-Thought(ToT) prompting（将思维链扩展为思维树，可以使用多种搜索策略）[[2305.10601](https://arxiv.org/abs/2305.10601)]
* automatic prompt engineer（提供输入输出示例，LLM 生成多个 prompt，评估选出效果最好的）[[2211.01910](https://arxiv.org/abs/2211.01910)]
* active prompt（选出 LLM 多次回复最不一致的问题，人类标注 CoT 过程）[[2302.12246](https://arxiv.org/abs/2302.12246), [2305.08291](https://arxiv.org/abs/2305.08291)]

提示注入（或提示攻击）是提示工程的一种恶意利用，其通过巧妙的提示诱导 LLM 产生有害或不希望的输出（例如变更任务、突破限制、泄露机密或隐私信息等）。

## PEFT

PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）方法仅微调少量模型参数，显著降低计算和存储成本，却能够实现与全参数微调相当的性能。

## 推理

一些技术：

* 量化（到 4-bit 量化）
* Speculative decoding
* 自一致性（self-consistency）[[2203.11171]](https://arxiv.org/abs/2203.11171) 方法让 LLM 采样多个回复，然后对于答案行多数决。
