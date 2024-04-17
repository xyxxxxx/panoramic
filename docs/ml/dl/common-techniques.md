# 常用技术

## 激活函数

* [2002.05202](https://arxiv.org/abs/2002.05202v1) 提出 GLU 层的变体用于替换 Transformer 两层 FFN 的第一层和中间的激活函数，实验显示，不论是预训练还是微调（在 GLUE 和 SuperGLUE 上进行基准测试），带有 GLU 变体的 FFN 都比原始的 FFN 表现更好。Google 的 PaLM 和 Meta 的 LLaMA（以及更多的语言模型）都使用了 SwiGLU。请参阅 [SwiGLU: GLU Variants Improve Transformer (2020)](https://kikaben.com/swiglu-2020/)。

## 正则化

正则化是一类通过限制模型复杂度，从而避免过拟合、提高泛化能力的方法。

* l1 和 l2 正则化（在损失函数中添加参数的 l1 范数或 l2 范数的平方）
* 权重衰减（weight decay）（每次更新参数前衰减参数，对于 vanilla SGD 等价于 l2 正则化）
* 提前停止（early stop）
* 丢弃（dropout）（训练中以固定概率清零张量中的一些元素）
    * PyTorch 的丢弃层（`torch.nn.Dropout`）会对丢弃后的剩余元素乘以放大系数 `1 / (1-p)`

## 归一化

首先说明这里的张量元素分组方法。对于多维张量，例如形状为 `(a, b, c, d)` 的向量，选取轴 0、2 作为组，表示轴 0、2 对应组的元素，轴 1、3 对应组的索引，即每个组包含 `a * c` 个元素，共有 `b * d` 个组。

* 批次归一化（batch normalization，BN）（选取轴 0 和可选的其他轴作为组，组内进行归一化）[[1502.03167](https://arxiv.org/abs/1502.03167), [1805.11604](https://arxiv.org/abs/1805.11604)]
* 层归一化（layer normalization，LN）（选取最后 N 个轴作为组，组内进行归一化）[[1607.06450](https://arxiv.org/abs/)]
* ……

!!! question "归一化为什么会对优化有帮助？"
    请参阅此[视频链接](https://youtu.be/BABPWOkSbLE?t=1552)。

!!! note "注意"
    PyTorch 的批次归一化层（包括 `torch.nn.BatchNorm1d`、`torch.nn.BatchNorm2d` 和 `torch.nn.BatchNorm3d`）总是选取轴 0 和可能存在的轴 2、3、4 作为组，共有 C（轴 1 的规模）个组。

PyTorch 演示：

```python
>>> input = torch.arange(24).reshape((2, 3, 4)).to(torch.float32)
>>> input
tensor([[[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]],

        [[12., 13., 14., 15.],
         [16., 17., 18., 19.],
         [20., 21., 22., 23.]]])

>>> nn.BatchNorm1d(12, affine=False)(input.reshape(2, 12))  # 合并轴 1、2，选取轴 0 作为组 
tensor([[-1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
         -1.0000, -1.0000, -1.0000, -1.0000],
        [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,
          1.0000,  1.0000,  1.0000,  1.0000]])
>>> nn.BatchNorm2d(1, affine=False)(input.unsqueeze(1))     # 插入轴 1，选取轴 0、2、3 作为组
tensor([[[[-1.6613, -1.5169, -1.3724, -1.2279],
          [-1.0835, -0.9390, -0.7945, -0.6501],
          [-0.5056, -0.3612, -0.2167, -0.0722]]],


        [[[ 0.0722,  0.2167,  0.3612,  0.5056],
          [ 0.6501,  0.7945,  0.9390,  1.0835],
          [ 1.2279,  1.3724,  1.5169,  1.6613]]]])

>>> F.layer_norm(input, [4])        # 选取轴 2 作为组
tensor([[[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]],

        [[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]]])
>>> F.layer_norm(input, [3, 4])     # 选取轴 1、2 作为组
tensor([[[-1.5933, -1.3036, -1.0139, -0.7242],
         [-0.4345, -0.1448,  0.1448,  0.4345],
         [ 0.7242,  1.0139,  1.3036,  1.5933]],

        [[-1.5933, -1.3036, -1.0139, -0.7242],
         [-0.4345, -0.1448,  0.1448,  0.4345],
         [ 0.7242,  1.0139,  1.3036,  1.5933]]])
```

## 梯度裁剪

## 混合精度训练

!!! info "参考"
    * [What Every User Should Know About Mixed Precision Training in PyTorch](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)

现代神经网络的高效训练通常依赖于使用较低精度的数据类型。在 A100 GPU 上，FP16 矩阵乘法和卷积的峰值性能比 FP32 的峰值性能快 16 倍。由于 FP16 和 BF16 数据类型的大小仅为 FP32 的一半，它们可以使受带宽限制的内核性能提升一倍，并减少训练网络所需的内存，从而允许使用更大的模型、更大的批次或更大的输入，进而提高性能。

### 训练精度

常用浮点类型的位数分配如图：

![](../../assets/ml/dl/common-techniques/precisions.png)

**FP32**：在 PyTorch 中，默认情况下浮点张量和模块都以 FP32 精度创建，但这是一个历史遗留问题，并不能代表大多数现代深度学习网络的训练需求。网络很少需要这么高的数值精度。

**FP16 & BF16**：这两种低精度浮点数据类型通常具有相似的速度，但有些网络可能只能在其中一种类型上收敛。如果一个网络需要更高的精度，可能需要使用 FP16，如果一个网络需要更大的动态范围，可能需要使用 BF16，其动态范围与 FP32 相等。如果观察到溢出等问题，建议尝试使用 BF16。BF16 仅在 Ampere 及后续架构的 CUDA 设备上可用。

**TF32**：TF32 是 NVIDIA Ampere GPU 新引入的数学模式（而非数据类型），其在第三代 Tensor Core 上执行 FP32 浮点数的矩阵相乘和卷积运算。计算流程如上图所示，和全 FP32 精度计算的区别仅在于将输入在相乘之前舍入到 TF32 精度。

![](../../assets/ml/dl/common-techniques/tf32.png)

TF32 相比 FP32 有相同的数值范围但更低的数值精度。根据 NVIDIA 的研究，大多数模型训练都不会受到影响，并且显示出与 FP32 精度训练相同的收敛性和准确率。默认情况下，PyTorch 仅为卷积启用 TF32 模式，而没有矩阵乘法。建议也为矩阵乘法启用此设置，除非网络需要全 FP32 精度。TF32 仅在 Ampere 及后续架构的 CUDA 设备上可用。

!!! info "TF32"
    * [TensorFloat-32 in the A100 GPU Accelerates AI Training, HPC up to 20x](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
    * [Accelerating AI Training with NVIDIA TF32 Tensor Cores](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)

### 混合精度训练



## 量化
