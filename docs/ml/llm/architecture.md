# 架构

GPT 系列模型的基础架构请参阅 [LLM Visualization](https://bbycroft.net/llm)。

### 位置嵌入

输入嵌入包括 token 嵌入和位置嵌入。这是因为自注意力机制（乃至整个 transformer）并不存在处理位置等信息的机制，token 的所有信息都必须写入到输入向量中。

位置嵌入考虑绝对位置和相对位置，但 token 的绝对位置并不重要——token 并不会因为它的绝对位置而被赋予什么含义；换个角度想，训练中一个 token、句子或段落出现在上下文窗口中的位置是随机的，绝对位置实际上没有什么意义。并且在越来越长的上下文窗口中，对于绝对位置的编码也难以形成差异。

[2104.09864](https://arxiv.org/abs/2104.09864) 提出的 RoPE（Rotary Position Embedding）采用了相对位置，并且在计算上简单高效。

RoPE 实质上就是，对查询向量 $\pmb q_m$ 和键向量 $\pmb k_m$ 的元素进行两两分组，每组视作一个二维向量，然后左乘矩阵

$$
\begin{bmatrix}
\cos m\theta & -\sin m\theta\\
\sin m\theta &  \cos m\theta
\end{bmatrix}
$$

即在实平面中逆时针旋转一个角度 $m\theta$。之后对 $\pmb q_m$ 和 $\pmb k_n$ 计算点积，可以视作相应组作点积再求和，因此对于相距越远的 $m$ 和 $n$，旋转角度差 $(m-n)\theta$ 越大，点积 $\pmb q_{m,[j]}\cdot\pmb k_{n,[j]}=\vert\pmb q_{m,[j]}\vert\vert\pmb k_{n,[j]}\vert\cos\alpha$ 因为夹角 $\alpha$ 变化而产生的变化越大，其中 $[j]$ 表示第 $j$ 组。

### 注意力头

* MHA（多头注意力）（多个 q、k、v 头）[[1706.03762](https://arxiv.org/abs/1706.03762)]
* MQA（多查询注意力）（多个 q 头，一个 k、v 头；大幅提升计算速度，但造成模型表现下降以及训练不稳定）[[1911.02150](https://arxiv.org/abs/1911.02150)]
* GQA（分组查询注意力）（每组多个 q 头，一个 k、v 头；计算速度接近 MQA，模型表现接近 MHA）[[2305.13245](https://arxiv.org/abs/2305.13245)]

![](../../assets/ml/llm/gqa.png)

## 上下文长度

https://arxiv.org/pdf/2311.12351.pdf

## 多模态

## MOE
