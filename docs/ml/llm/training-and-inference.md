# 训练和推理

## 预训练

## 微调

### 有监督微调

### RLHF

PPO

DPO

### RLAIF

* Constitutional AI（宪法 AI，）[[2212.08073]](https://arxiv.org/abs/2212.08073)

## PEFT

PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）方法仅微调少量模型参数，显著降低计算和存储成本，却能够实现与全参数微调相当的性能。

## 推理

一些技术：

* 量化（到 4-bit 量化）
* Speculative decoding
* 自一致性（self-consistency）[[2203.11171]](https://arxiv.org/abs/2203.11171) 方法让 LLM 采样多个回复，然后对于答案行多数决。
