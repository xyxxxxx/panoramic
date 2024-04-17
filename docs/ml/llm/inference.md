## 推理

一些降低推理成本、提升推理速度的技术：

* 量化（到 4-bit 量化）
* Speculative decoding

一些改进推理结果的技术：

* self-consistency（采样多个答案，选取最一致的答案）[[2203.11171]](https://arxiv.org/abs/2203.11171)
* LLM cascade（顺序调用从弱到强，同时也是成本从低到高的多个 LLM，当答案足够可靠时返回给用户，并取消后续调用）[[2305.05176](https://arxiv.org/abs/2305.05176)]
