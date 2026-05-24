# AI 赋格生成器调研报告

更新时间：2026-05-24。

## 目标问题

目标系统输入：

- 调式/调性，例如 `C minor`、`D dorian`。
- 一段主题，可以来自 MIDI、MusicXML，或后续扩展为文本音名+时值。
- 声部数量：三声部或四声部。

目标输出：一首有可辨认赋格过程的 MIDI，至少包含 exposition、若干 episode/middle entry、终止回归和可选 stretto/pedal/coda。

## 1. 是否已有类似东西？

结论：有相近系统，但没有发现一个成熟、开源、可直接满足本项目规格的工具。

商业软件方面，capella 的 `tonica fugata` 已经非常接近：官方说明它可完成 “composition, canon, prelude, fugue, figured bass” 等任务，并明确列出 “Compose a fugue on a subject”。其说明页还提到三/四声部写作、AI 辅助、浏览器/桌面版本和付费授权。它证明任务本身不是幻想，但它是闭源商业软件，不适合作为本项目的可复现研究基础。

开源/研究系统方面，主流可用作品集中在 Bach chorale 或四声部和声补全，而不是完整赋格结构：

- `DeepBach`：ICML 2017，针对 Bach chorale，使用 pseudo-Gibbs sampling，可由用户约束 notes/rhythms/cadences；仓库包含 MuseScore/Flask 交互代码。它的“可约束补全”很适合作为赋格局部生成的参考，但体裁不是赋格。
- `Coconet / Counterpoint by Convolution`：Magenta/Google Bach Doodle 背后的模型。它把复调写作视为 partial score completion，并用 blocked Gibbs sampling 反复重写。该思想很适合做“给定主题 entry 和骨架后补全其他声部”的引擎，但训练语料主要是 Bach chorales。
- `BachBot`：LSTM 生成 Bach 风格音乐/chorale 的项目，技术路线较旧，可参考语料编码和 baseline，但不直接处理赋格形式。
- 一些课程/论文原型做过 automatic fugue generation，例如 Yu Yue/Yue Yang 的遗传算法三声部赋格原型，使用 bundle optimization、声部内/声部间 evaluator 和 Bach 风格目标；也有早期 genetic algorithm counterpoint/fugue 论文。它们说明“规则+搜索”路线可行，但不是现代可用工程产品。

数据方面，最有价值的公开资源是 `humdrum-tools/bach-wtc-fugues`：它包含《平均律键盘曲集》Book I/II 的赋格，并将每个 voice 放在 separate spines，便于声部级分析。`humdrum-tools/bach-wtc` 则提供 WTC 全集 Humdrum edition 和 kernScores 动态转换入口。JSB chorales 数据集可作为基础四声部语法预训练数据，但不是赋格语料。

## 关键参考

- tonica fugata: https://www.capella-software.com/us/index.cfm/products/tonica-fugata/info-tonica-fugata/
- tonica fugata composition notes: https://www.capella-software.com/us/index.cfm/products/tonica-fugata/how-tonica-fugata-composes/
- DeepBach paper: https://arxiv.org/abs/1612.01010
- DeepBach code: https://github.com/Ghadjeres/DeepBach
- Coconet blog: https://magenta.tensorflow.org/coconet
- Coconet paper: https://arxiv.org/abs/1903.07227
- Magenta Coconet code: https://github.com/magenta/magenta/tree/main/magenta/models/coconet
- BachBot: https://github.com/feynmanliang/bachbot
- WTC fugue Humdrum corpus: https://github.com/humdrum-tools/bach-wtc-fugues
- WTC Humdrum corpus: https://github.com/humdrum-tools/bach-wtc
- JSB chorales dataset: https://github.com/czhuang/JSB-Chorales-dataset
- Fugue structure overview: https://www.britannica.com/art/fugue/Elements-of-the-fugue
- Automated Fugue Generation slides/poster: https://www.slideserve.com/adeola/automated-fugue-generation

## 对现有方案的判断

`tonica fugata` 是最接近“产品级”的已有方案，但闭源且不可控。DeepBach/Coconet 是最接近“可借鉴模型机制”的开源方向，但它们解决的是局部复调补全/chorale harmonization，不解决以下赋格核心问题：

- 主题在多个声部、多个调性区域中反复出现。
- real answer 与 tonal answer 的选择。
- countersubject 是否可逆对位，是否能和 subject 在不同声部组合。
- episode 需要从 subject/countersubject 动机中发展，同时承担转调功能。
- 3/4 声部的织体密度、声部进入顺序、终止前回归和高潮安排。

因此，本项目不建议从“纯神经网络端到端生成完整赋格”开始。更合理的定义是：用规则/约束系统保证赋格骨架和对位合法性，用小模型或统计模型负责候选补全、风格排序和局部材料生成。

