# 训练和推理

## 预训练

## 微调

对于微调的数据，质量比数量更重要，换言之，在精不在多。

### 有监督微调

### RLHF

PPO

DPO

### RLAIF

* Constitutional AI（宪法 AI，）[[2212.08073]](https://arxiv.org/abs/2212.08073)

## PEFT

PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）方法仅微调少量模型参数，显著降低计算和存储成本，却能够实现与全参数微调相当的性能。

## 推理

一些降低推理成本、提高推理速度、改进推理结果的技术：

* 量化（到 4-bit 量化）
* Speculative decoding
* self-consistency（采样多个答案，选取最一致的答案）[[2203.11171]](https://arxiv.org/abs/2203.11171)
* LLM cascade（顺序调用从弱到强，同时也是成本从低到高的多个 LLM，当答案足够可靠时返回给用户，并取消后续调用）[[2305.05176](https://arxiv.org/abs/2305.05176)]
