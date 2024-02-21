# 洞见

* [2402] OpenAI 预告视频生成模型 [Sora](https://openai.com/sora)，远超一众竞品，迅速引爆 AI 社群。Sora 拥有以下能力：

    * 能够生成多种长度（最长达 1min）、比例和分辨率的视频（和图像）。
    * 不仅支持文生视频，还支持根据图像和视频（双向）扩展视频，编辑图片和视频。
    * 通常（但不总是）能够保持物体的一致性。
    * 指令遵循较好。
    * 模拟复杂的物理场景时仍存在问题。

    技术细节请参阅[技术报告](https://openai.com/research/video-generation-models-as-world-simulators)以及下列论文：

    * [2010.11929](https://arxiv.org/abs/2010.11929)
    * [2103.15691](https://arxiv.org/abs/2103.15691)
    * [2212.09748](https://arxiv.org/abs/2212.09748)
    * [2307.06304](https://arxiv.org/abs/2307.06304)

    OpenAI 发现，在大规模训练时，视频模型展现出许多有趣的涌现能力。这些能力使得 Sora 能够模拟现实世界中人类、动物和环境的某些方面。可以看到，在引入 Transformer 模型后，视频模型和语言模型一样，具有 high scalability，并且在扩大规模后涌现出了新的能力。接下来的研究会朝着这个方向继续前进。

    视频模型的“GPT-3 时刻”已经到来，接下来就是不断的迭代。预期今年内将有比较好的开源视频模型和 web UI。

* [23xx] Stable Diffusion

    https://stable-diffusion-art.com/how-stable-diffusion-work/
