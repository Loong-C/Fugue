# 资源清单

更新时间：2026-05-24。

## 首选数据

| 名称 | 用途 | 获取方式 | 备注 |
| --- | --- | --- | --- |
| `humdrum-tools/bach-wtc-fugues` | 主要赋格语料；WTC Book I/II 的赋格，声部分离，适合分析 subject entry、声部范围、对位关系。 | `python scripts/fetch_resources.py` | 原始仓库说明其每个 voice 放在独立 spine，便于声部分析。重新分发前需确认编码授权。 |
| `humdrum-tools/bach-wtc` | 补充 WTC 全集，kernScores/Humdrum 可动态转换格式。 | `python scripts/fetch_resources.py` | 可用于研究调性规划、节拍、材料组织。 |
| `czhuang/JSB-Chorales-dataset` | 预训练/约束学习；4 声部 SATB 的固定网格数据。 | `python scripts/fetch_resources.py` | 不是赋格，但适合训练基础 tonal counterpoint/inpainting 先验。 |

## 可选模型/代码参考

| 名称 | 用途 | 获取方式 | 备注 |
| --- | --- | --- | --- |
| Magenta Coconet checkpoint | 研究 masked/inpainting polyphony。 | `python scripts/fetch_resources.py --include-models` | 体积较大；Magenta 仓库已归档，但方法仍有参考价值。 |
| DeepBach | 研究 pseudo-Gibbs 采样、可约束四声部生成。 | 手动 clone 或后续加入脚本 | 旧 Python/PyTorch 栈，建议只借鉴思想，不直接作为核心依赖。 |

## 本地状态记录

本次运行记录：

- 已下载：`data/raw/humdrum/bach-wtc-fugues/`
- 已下载：`data/raw/humdrum/bach-wtc/`
- 已下载：`data/raw/jsb-chorales-dataset/`
- 已创建：`.venv/`，但普通 pip 安装被 WinError 10013 阻止，无法访问 PyPI。
- 已创建并可用：`.venv-system/`，使用 `--system-site-packages` 复用系统 Python 包，并以 `--no-deps --no-build-isolation` 安装本项目。
- 已验证可用：`fugue --help`、`music21`、`mido`、`pretty_midi`、`numpy`、`pandas`、`typer`、`torch`、`transformers`。
- 当前缺失：`z3-solver`、`python-constraint`、`miditok`。这些需要后续在允许 PyPI/conda 访问的环境中安装，或改用系统已有的 `cvxpy`/`highspy`/自写 beam search 先完成 MVP。

补充状态：

- `.venv/` 中已经可导入 `z3-solver`、`python-constraint`、`miditok`。
- `.venv/pyvenv.cfg` 已设置 `include-system-site-packages = true`，因此同一环境也能访问系统中的 `music21`、`mido`、`pretty_midi`、`torch` 等包。
- 当前生成器没有强依赖 z3；它使用随机 beam-like sampling + rule scoring。这样即使 solver 不可用，核心生成路径仍能运行。
