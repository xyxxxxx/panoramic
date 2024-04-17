# 评估

完整、全面、公正地评估 LLM 的能力对于我们正确理解、改进和应用 LLM 有着十分重要的意义，但在当下却是一项困难的任务。

## 评估维度

相比于过往的语言模型仅限制于单一特定任务，LLM 拥有能够解决多种任务的强大能力，对于每一种任务类型，或者说能力维度都有必要进行评估。不仅如此，人类对于 LLM 还有更高的要求，例如：对齐人类的价值观，遵循人类的指令和意图，以及安全可靠地运行。LLM 最终能在哪些应用场景落地，在特定应用场景可以为人类提供多大程度的帮助，也是所有人关心的问题。

这里将当前 LLM 的所有评估维度总结如下。当然，不同的人会有不同的分类标准和关注点，并且 LLM 的能力和多样性仍在快速发展中，新的评估维度还会不断出现。

**知识和能力（Knowledge and Capabilities）**

<table>
  <tr>
    <td rowspan="3">自然语言能力</td>
    <td>自然语言理解</td>
    <td>是否（或在多大程度上，下同）能够理解输入文本中的含义、情感等信息，任务包括情感分析、文本分类、命名实体检测、语义理解等。</td>
  </tr>
  <tr>
    <td>自然语言生成</td>
    <td>是否能够在理解输入文本的基础上，按照要求生成输出文本，任务包括总结、对话、问答、翻译、写作/创作/头脑风暴等。</td>
  </tr>
  <tr>
    <td>多语言</td>
    <td>在不同语言的相同任务上是否表现相当。</td>
  </tr>
  <tr>
    <td rowspan="4">推理能力</td>
    <td>常识推理</td>
    <td>是否能够运用人类的常识进行推理。</td>
  </tr>
  <tr>
    <td>逻辑推理</td>
    <td>是否能够进行逻辑推理。</td>
  </tr>
  <tr>
    <td>多跳推理</td>
    <td>是否能够捕获并连接多个信息片段以进行推理。</td>
  </tr>
  <tr>
    <td>数学推理</td>
    <td>是否能够进行数学抽象、推理和计算并解决数学问题。</td>
  </tr>
  <tr>
    <td rowspan="3">多模态能力</td>
    <td>特殊文本</td>
    <td>是否能够理解和生成（Markdown、PDF 等格式中的）列表、链接、表格、代码、数学公式等特殊文本。</td>
  </tr>
  <tr>
    <td>视觉</td>
    <td>是否能够阅读、理解和生成图像（和视频）。</td>
  </tr>
  <tr>
    <td>语音</td>
    <td>是否能够接收、理解和生成语音。（可以借助外部模型）</td>
  </tr>
  <tr>
    <td rowspan="2">知识</td>
    <td>世界/学科/领域知识</td>
    <td>对于各个学科领域的专业知识有何种程度的掌握。</td>
  </tr>
  <tr>
    <td>常识/社会知识</td>
    <td>对于人类的日常生活、社交交往、文化传统等方面有何种程度的了解。</td>
  </tr>
</table>

**对齐（Alignment）**

<table>
  <tr>
    <td>指令遵循</td>
    <td>是否准确遵循用户的指令，满足用户的需求。</td>
  </tr>
  <tr>
    <td>道德和伦理</td>
    <td>是否会生成违反人类道德和伦理的内容。</td>
  </tr>
  <tr>
    <td>法律法规</td>
    <td>是否会生成违反当地法律法规的内容。</td>
  </tr>
  <tr>
    <td>立场、偏见和刻板印象</td>
    <td>是否在政治、社会、文化等议题中采取客观中立的立场，是否对于特定人群存在偏见或刻板印象。</td>
  </tr>
  <tr>
    <td>有害信息</td>
    <td>是否会（主动或在人类要求下）产生冒犯、侮辱、仇恨、暴力、淫秽等有害信息。</td>
  </tr>
  <tr>
    <td>隐私保护</td>
    <td>是否会（主动或在人类要求下）泄漏其他人的隐私信息。</td>
  </tr>
  <tr>
    <td>真实性</td>
    <td>是否会编造虚假信息/产生幻觉（hallucination）。</td>
  </tr>
</table>

**安全和可靠性（Safety and Reliability）**

<table>
  <tr>
    <td>对抗提示词攻击</td>
    <td>对抗提示词攻击（在字、词、句或语义级别对提示词施加一个扰动，诱导 LLM 产生不正确的回复）的健壮性。</td>
  </tr>
  <tr>
    <td>对抗越狱攻击</td>
    <td>对抗越狱攻击（使用特定的提示词诱导 LLM 突破对齐限制）的健壮性。</td>
  </tr>
</table>

**领域和应用**

在教育、医疗、金融、法律、科学研究、程序开发、游戏开发、心理学等专业领域中 LLM 展现了广阔的应用前景，例如在教育领域 LLM 可以解答学生的问题、评估学生的能力，在医疗领域 LLM 可以回答医疗咨询、辅助医生诊断。LLM 在每个领域的评估维度与该领域的实际工作内容和特性有关，并且仍然在探索之中。目前的一些评估工作主要包含：构造一些不同形式的领域任务作为基准测试，或将标准化职业资格考试或作为基准测试。

此外，LLM 利用工具的能力也是重要的评估维度。基于能够使用多种工具的 LLM，我们可以开发出的强大的智能体（agent）。

## 指标

**正确率（accuracy）**

客观题的正确率。

**EM & F1**

精确匹配（Exact Match，EM）： 度量模型生成的答案是否与人工标注的答案完全匹配。如果生成的答案与人工标注的答案一字不差，那么精确匹配得分为 1，否则为 0。

F1 分数： 计算生成的答案与人工标注答案之间的共享词汇的精确度和召回率，F1 分数是精确度和召回率的调和平均值。

**BLEU & ROUGE**

参阅 https://clementbm.github.io/theory/2021/12/23/rouge-bleu-scores.html

* 自动计算生成文本与参考文本之间的基于统计的文本相似度（而非语义相似度），1 代表完全相同，0 代表完全不同。
* 与人类的判断高度一致，是可靠的指标。
* BLEU 为 precision 导向，ROUGE 为 recall 导向。

**胜率**

相对于 benchmark 模型的胜率。

**通过率/成功率**

评估是否能够在一些具体应用中完成任务，例如网购、生成代码、操纵机器人等。

## 基准测试

迄今为止，已经有大量的基准测试（或数据集）被构造出来，分别用于评估 LLM 在某一个到多个维度的性能。这里将按照主题划分，介绍一些常见的基准测试。另外请注意，一个基准测试可以同时评估 LLM 多个维度的性能（例如阅读理解类的基准测试可以同时评估 LLM 的自然语言理解、自然语言生成和多跳推理能力），因此主题分类并不唯一。

**自然语言理解/生成**

在预训练语言模型流行之前，这些基准测试就已经被广泛应用于评估当时较为简单的语言模型在自然语言理解/生成方面的能力，涵盖了情感分析、文本分类、阅读理解等经典任务。当下的 LLM 在绝大多数这些基准测试中都能轻松取得非常出色的表现。

| benchmark   | paper                                                                              | link                                                                                                | 任务             | 指标                           | 说明                                                                                        |
| ----------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------ | ------------------------------------------------------------------------------------------- |
| IMDB        | [acl](https://aclanthology.org/P11-1015/)                                          | [hf](https://huggingface.co/datasets/imdb)                                                          | 情感分析         | acc                            |                                                                                             |
| SST-2       | [acl](https://aclanthology.org/D13-1170/)                                          | [hf](https://huggingface.co/datasets/sst2)                                                          | 情感分析         | acc                            |                                                                                             |
| SQuAD       | [1.0](https://arxiv.org/abs/1606.05250)<br>[2.0](https://arxiv.org/abs/1806.03822) | [hf1.0](https://huggingface.co/datasets/squad)<br>[hf2.0](https://huggingface.co/datasets/squad_v2) | 阅读理解         | EM<br>F1                       | • 语料来自维基百科<br>• 2.0 增加了基于上下文无法回答的问题，测试模型是否知道它不知道        |
| NarrativeQA | [arxiv](https://arxiv.org/abs/1712.07040)                                          | [hf](https://huggingface.co/datasets/narrativeqa)                                                   | 阅读理解         | BLEU<br>ROUGE<br>Meteor<br>MRR | • 语料为完整的小说或电影剧本，或是人类给出的概要<br>• 问题和答案皆为人类生成                |
| CoQA        | [arxiv](https://arxiv.org/abs/1808.07042)                                          | [official](https://stanfordnlp.github.io/coqa/)                                                     | 对话<br>阅读理解 | F1                             | • 语料来自多种领域<br>• 问题、答案皆为人类聊天时产生，解释为人类框选原文得到                |
| RAFT        | [arxiv](https://arxiv.org/abs/2109.14076)                                          | [official](https://raft.elicit.org/)<br>[hf](https://huggingface.co/datasets/ought/raft)            | 文本分类         | acc                            | • 11 个来自现实应用的较为复杂的分类任务<br>• LLM 通过 few-shot 的上下文学习或微调来进行分类 |

**推理**

人们逐渐认识到推理能力是 LLM 完成许多任务的关键能力，同时也是机器智能的重要方面。越来越多与推理相关的数据集被释出，从逻辑推理到常识推理、多跳推理和数学推理，难度也在逐渐升高。

| benchmark     | paper                                         | link                                                                                                     | 任务     | 指标                           | 说明                                               |
| ------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------- | ------------------------------ | -------------------------------------------------- |
| SNLI          | [arxiv](https://arxiv.org/abs/1508.05326)     | [hf](https://huggingface.co/datasets/snli)                                                               | 逻辑推理 | acc                            | • 两句话（前提和假设）的关系是包含、中立或矛盾之一 |
| MultiNLI      | [arxiv](https://arxiv.org/abs/1704.05426)     |                                                                                                          | 常识推理 | acc                            | • 相比 SNLI，在多样性、难度上进行了升级            |
| HotpotQA      | [arxiv](https://arxiv.org/abs/1809.09600)     | [official](https://hotpotqa.github.io/)<br>[hf](https://huggingface.co/datasets/hotpot_qa)               | 多跳推理 | EM<br>F1<br>（对于回答和解释） | • 语料来自维基百科<br>• 涵盖多种多跳推理类型       |
| CommonsenseQA | [arxiv](https://arxiv.org/abs/1811.00937)     | [hf](https://huggingface.co/datasets/commonsense_qa)                                                     | 常识推理 | acc                            |                                                    |
| PIQA          | [arxiv](https://arxiv.org/pdf/1911.11641.pdf) | [hf](https://huggingface.co/datasets/piqa)                                                               | 常识推理 | acc                            |                                                    |
| HybridQA      | [arxiv](https://arxiv.org/abs/2004.07347)     | [hf](https://huggingface.co/datasets/hybrid_qa)                                                          | 多跳推理 | EM<br>F1                       |                                                    |
| HellaSwag     | [arxiv](https://arxiv.org/abs/1905.07830)     | [official](https://rowanzellers.com/hellaswag/)<br>[hf](https://huggingface.co/datasets/Rowan/hellaswag) | 常识推理 | acc                            |                                                    |

**为人类设计的测验题**

这些基准测试搜集了原本为人类设计的标准化考试题目，其中数学科目主要考察 LLM 的数学推理能力，其他科目则主要考察 LLM 在各学科领域方面的知识量。

| benchmark    | paper                                     | link                                                                                                                            | 任务                   | 说明                                                                  |
| ------------ | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------- |
| ARC          | [arxiv](https://arxiv.org/abs/1803.05457) | [hf](https://huggingface.co/datasets/ai2_arc)                                                                                   | 选择题                 | • 三年级到九年级的自然/科学科目，~8k 道问题                           |
| MMLU         | [arxiv](https://arxiv.org/abs/2009.03300) | [hf](https://huggingface.co/datasets/cais/mmlu)                                                                                 | 选择题                 | • 57 个学科<br>• 来源不明                                             |
| MATH         | [arxiv](https://arxiv.org/abs/2103.03874) | [hf](https://huggingface.co/datasets/competition_math)                                                                          | 解答题                 | • 12.5k 道高中数学竞赛题，包含 7 个子科目和 5 级难度                  |
| GSM8K        | [arxiv](https://arxiv.org/abs/2110.14168) | [hf](https://huggingface.co/datasets/gsm8k)                                                                                     | 解答题                 | • 8.5k 道小学数学应用题，表述的自然语言具有多样性                     |
| AGIEval      | [arxiv](https://arxiv.org/abs/2304.06364) | [github](https://github.com/ruixiangcui/AGIEval)                                                                                | 选择题<br>填空题         | • 中英文标准化考试                                                    |
| C-Eval       | [arxiv](https://arxiv.org/abs/2305.08322) | [official](https://cevalbenchmark.com/index.html#home)<br>[blog](https://www.notion.so/C-Eval-6b79edd91b454e3d8ea41c59ea2af873) | 选择题                 | • 对标 MMLU<br>• 52 个学科，~14k 道问题<br>• 使用模拟题手工构建       |
| GAOKAO-bench | [arxiv](https://arxiv.org/abs/2305.12474) | [github](https://github.com/OpenLMLab/GAOKAO-Bench)                                                                             | 选择题<br>填空题<br>解答题 | • 2010-2022 年高考题目                                                |
| CMMLU        | [arxiv](https://arxiv.org/abs/2306.09212) | [github](https://github.com/haonan-li/CMMLU)<br>[hf](https://huggingface.co/datasets/haonan-li/cmmlu)                           | 选择题                 | • 对标 MMLU<br>• 67 个学科或主题（包含中国的本地化主题），~12k 道问题 |

**对齐**

测试 LLM 是否遵循人类的意图或符合人类的价值观，包括遵循指令，不违反道德和法律，不引入既有立场或偏见，不产生有害信息（淫秽、暴力、仇恨等）或虚假信息（幻觉），不泄露隐私等。

| benchmark                | paper                                         | link                                                                                                        | 能力                         | 说明                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------ | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TruthfulQA               | [arxiv](https://arxiv.org/abs/2109.07958)     | [hf](https://huggingface.co/datasets/truthful_qa)                                                           | trustworthiness              | • 问题比较刁钻，包括许多人类具有的误解/错误观念/迷信/幻想/阴谋论等                                                                                                                                                                                                                                                                                                                                                |
| MT-Bench & Chatbot Arena | [arxiv](https://arxiv.org/abs/2306.05685)     | [Chatbot Arena](https://chat.lmsys.org/)                                                                    | 对齐人类偏好/<br>指令遵循        | • 评估 chatbot<br>• MT-Bench 由 80 道高质量两回合问题构成，包含用户 prompt 中常见的 8 种类型：写作、角色扮演、信息提取、推理、数学、代码、知识一（STEM）、知识二（人文社科）<br>• 在 ChatBot Arena 中，由用户自由提出问题<br>• 分别收集人类比较和 GPT-4 比较的结果<br>• 实验表明 GPT-4 与人类对于比较的一致性比人类之间的一致性更高，证明这里 GPT-4 可以替代人类<br>• 指出 LLM 作为裁判的一些问题以及部分解决方案 |
| AlpacaEval               | [arxiv](https://arxiv.org/pdf/2305.14387.pdf) | [official](https://tatsu-lab.github.io/alpaca_eval/)<br>[github](https://github.com/tatsu-lab/alpaca_eval) | 指令遵循                     | • 评估 LLM 遵循指令的能力• 回复与（Davinci003 模型的）参考回复进行比较，由 GPT-4/Claude/ChatGPT 自动评估• 与人类的偏好高度一致                                                                                                                                                                                                                                                                                    |
| SelfAware                | [arxiv](https://arxiv.org/abs/2305.18153)     | [github](https://github.com/yinzhangyue/SelfAware)                                                          | trustworthiness              |                                                                                                                                                                                                                                                                                                                                                                                                                   |
| TrustGPT                 | [arxiv](https://arxiv.org/abs/2306.11507)     |                                                                                                             | 有害信息<br>偏见<br>道德伦理 | • 使用 Social Chemistry 101 作为数据集                                                                                                                                                                                                                                                                                                                                                                            |
| DecodingTrust            | [arxiv](https://arxiv.org/abs/2306.11698)     | [github](https://github.com/AI-secure/DecodingTrust)                                                        | trustworthiness              |                                                                                                                                                                                                                                                                                                                                                                                                                   |

**安全**

测试 LLM 对抗提示词攻击和越狱攻击的能力。

| benchmark   | paper                                     | link                                               | 能力              | 说明 |
| ----------- | ----------------------------------------- | -------------------------------------------------- | ----------------- | ---- |
| AdvGlue     | [arxiv](https://arxiv.org/abs/2111.02840) | [hf](https://huggingface.co/datasets/adv_glue)     | task robustness   |      |
| PromptBench | [arxiv](https://arxiv.org/abs/2306.04528) | [github](https://github.com/microsoft/promptbench) | prompt robustness |      |

**使用工具**

测试 LLM 使用多种外部工具的能力，以及进一步地，作为智能体或人类助手的能力。

| benchmark            | paper                                     | link                                                                    | 能力                      | 说明                                                                                 |
| -------------------- | ----------------------------------------- | ----------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| API-Bank             | [arxiv](https://arxiv.org/abs/2304.08244) |                                                                         | 使用工具/<br>作为 agent   |                                                                                      |
| ToolEval (Toolbench) | [arxiv](https://arxiv.org/abs/2307.16789) | [github](https://github.com/OpenBMB/ToolBench/blob/master/README_ZH.md) | 使用工具                  | • 真实世界的 API 从 RapidAPI 收集<br>• 指令和解答均由 ChatGPT 生成                   |
| GAIA                 | [arxiv](https://arxiv.org/abs/2311.12983) | [hf](https://huggingface.co/datasets/gaia-benchmark/GAIA)               | 使用工具/<br>作为 AI 助手 | • 任务涉及推理、网页浏览、读图和其他工具使用<br>• 人类完成 92% vs GPT-4（带插件）15% |

**综合基准测试**

这些工作尝试多方位或全方位地评估 LLM，采用多场景、多数据集、多指标的方式衡量多个 LLM。它们提供的排行榜（leaderboard）是许多人在想要了解一个模型大致能力水平时首先参考的对象。

BERT 时代：

| benchmark | paper                                                 | link                                                         | 任务 | 说明        |
| --------- | ----------------------------------------------------- | ------------------------------------------------------------ | ---- | ----------- |
| GLUE      | [pdf](https://openreview.net/pdf?id=rJ4km2R5t7)       | [official](https://gluebenchmark.com/)                       |      |             |
| SuperGLUE | [arxiv](https://arxiv.org/abs/1905.00537)             | [official](https://super.gluebenchmark.com/)                 |      |             |
| CLUE      | [acl](https://aclanthology.org/2020.coling-main.419/) | [official](https://www.cluebenchmarks.com/static/index.html) |      | 中文版 GLUE |

LLM 时代：

| benchmark            | paper                                     | link                                                                                                        | 任务                               | 说明                                    |
| -------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------- |
| BIG-bench            | [arxiv](https://arxiv.org/abs/2206.04615) | [github](https://github.com/google/BIG-bench)                                                               | 全面评估                             | • 204 个任务                             |
| BBH                  | [arxiv](https://arxiv.org/abs/2210.09261) | [github](https://github.com/suzgunmirac/BIG-Bench-Hard)<br>[hf](https://huggingface.co/datasets/lukaemon/bbh) | BIG-bench 中 LLM 未能超过人类的任务        | • 23 个任务<br>• 应用 CoT prompting 取得显著性能提升 |
| HELM                 | [arxiv](https://arxiv.org/abs/2211.09110) | [official](https://crfm.stanford.edu/helm/latest/)                                                          | 全面评估                             | • 42 个场景，59 个指标                       |
| Open LLM Leaderboard |                                           | [hf](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)                                      | ARC<br>HellaSwag<br>MMLU<br>TruthfulQA |                                       |

**其他基准测试**

| benchmark | paper                                     | link                                                                                                                  | 能力                  | 说明                                                                                                                                                                                                                      |
| --------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HumanEval | [arxiv](https://arxiv.org/abs/2107.03374) | [github](https://github.com/openai/human-eval)<br>[hf](https://huggingface.co/datasets/openai_humaneval)              | 编写代码              | • 使用 Python 语言根据 docstring 编写函数                                                                                                                                                                                 |<br>• 与 Codex 同时发布
| LongEval  | /                                         | [blog](https://lmsys.org/blog/2023-06-29-longchat/)<br>[github](https://github.com/DachengLi1/LongChat)               | 上下文长度            | • 评估 LLM 处理长上下文的能力，包含两个任务<br>• 任务一是粗粒度的话题检索，需要取回第一个话题的主题，每个话题长 400~600 token<br>• 任务二是细粒度的行检索，需要精准地从某一行取回数字<br>• 容易将测试上下文扩展到任意长度 |
| SWE-bench | [arxiv](https://arxiv.org/abs/2310.06770) | [github](https://github.com/princeton-nlp/SWE-bench)<br>[hf](https://huggingface.co/datasets/princeton-nlp/SWE-bench) | 编写代码/<br>修复问题 | • 从 GitHub 收集的真实 issue                                                                                                                                                                                              |

## 现有问题

### 评估数据泄漏

古德哈特定律（Goodhart’s Law）：一项指标一旦变成了目标，它就不再是个好指标。

在预训练和微调 BERT 系列模型的时代，刷榜 GLUE/SuperGLUE 的现象就已经出现。很多人只为指标上的好看而过拟合数据集，短视近利，不顾泛化能力和用户体验，放弃追求实质的技术进步。现如今这一情况又重演，已有研究[[2310.19341](https://arxiv.org/abs/2310.19341)]和[实验](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam)质疑一些模型在基准测试 GSM8K 上训练过。另有研究[[2311.01964](https://arxiv.org/abs/2311.01964)]证明使用基准测试的数据训练会使得这一测试的分数大幅提高，小模型甚至可以超过参数量是其几十倍的大模型，然而其泛化能力和微调性能都会受到损害。

数据泄漏或许并非有意为之，而是模型的开发人员未能将测试数据从构成复杂的训练语料中清洗出去。但在训练语料不公开的情况下，作为用户我们也无从知晓。

xAI 在发布 Grok 模型时引入了最新的匈牙利全国高中数学期末考试（2023 年 5 月）进行评估，为了防止评估数据泄漏，基准测试 C-Eval 不公开测试集的答案（但这样做又会降低透明度）。或许以后 LLM 也会像人类学生一样，定期地参加闭卷考试？

C-Eval 的[博客](https://www.notion.so/C-Eval-6b79edd91b454e3d8ea41c59ea2af873)中讨论了构建这一基准测试的初衷，以及基准测试的正确和错误使用方法，供读者参考。

### 人类和 GPT-4 评估引入偏差

由于缺少基准测试、基准测试因数据泄漏而失效、基准测试问题的开放性，以及评估对齐人类价值的能力等原因，研究可能采取人类或 GPT-4[[2310.05470](https://arxiv.org/abs/2310.05470)]进行评估。尽管人类和 GPT-4 评估可以更加全面、细致和专业，但也会引入一些偏差。在人类评估中，每个评估者的观点、偏好和文化视角的不同可能会影响评估结果；对于 GPT-4，也可能有产生幻觉或能力不足的问题。

除此之外，人类评估还会花费更多的时间和成本（相对而言，GPT-4 评估花费的时间和成本要低一些）。

### 全面评估开销大

每个 LLM 在发布时通常会公布其在某些基准测试上的成绩，但这远远不能构成全面评估。一些工作（例如 BIG-bench、HELM）对当时的模型进行了全面评估，但由于评估开销大、流程复杂、环境难以复制等原因，之后只有少数 LLM 进行了跟进。

### 一些维度缺少基准测试

一些新兴的评估维度（例如幻觉、遵守法律法规、领域应用等）仍然缺少高质量、被广泛接受的基准测试，需要未来的研究工作进行补充。

### LLM 透明度降低

推出 LLM 的各家大小厂商在商业化的过程中越来越将相关技术视为商业机密，因而对其讳莫如深。OpenAI 的 GPT-2 尚且完全开源，而到了 GPT-3、GPT-4 就只剩对技术细节避而不谈的技术报告。在一份关于 LLM 透明度的报告[[2310.12941](https://arxiv.org/abs/2310.12941)]中，即使排名第一的 LLaMA2 也未公布其训练数据源。

我们呼吁基础模型的发布者公开更多训练相关信息，包括训练预料的构成、评估的环境和代码等。在缺少这些信息的情况下，我们无从分析评估过程中可能存在的问题。

## 讨论

自 2022 年 11 月 ChatGPT 发布以来，不计其数的 LLM 在 2023 年中被发布或开源出来。在 LLM 相关技术与应用取得显著进展的同时，围绕着“最强”、“第一”、“SOTA”的争夺也产生了一些乱象，部分 LLM 声称可以达到的优秀指标无法得到复现，或被质疑是通过刷榜、作弊等手段达到。不透明、不公正、不可靠的评估与比较也使得用户在挑选 LLM 时无所适从。

目前，LLM 难以得到完整、全面、公正的评估，这一研究领域仍在不断发展中。如果在选择 LLM 时感到迷茫，这里建议：

* 根据自己的实际需求，重点关注相关的评估维度和基准测试结果。
* 不要完全相信官方提供的数字，获取多方面的评价信息，多参与社区讨论。
* 将 LLM 的透明度作为选择时的考虑因素之一。
* 多亲自动手试用，还可以建立一个私有的评估数据集或基准测试。
