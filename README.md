# OneModel

用于复现 OneTrans-V2、UniR2 等统一推荐模型的研究项目骨架。参考本机 GenRec 的模块划分，
继续使用 Gin 配置；本项目独立运行，不依赖 GenRec 安装位置。

**当前没有实现任何具体推荐模型。** 网络、论文损失、数据读取、collator、SID 模型及评估
协议均留有明确扩展位置。已提供 Gin 解析、训练/评估入口、通用 Accelerate 训练循环和运行
配置存档。两份模型配置只是组件连接模板，默认参数不代表论文配方。

## 安装

在项目根目录执行，使用 Python 3.10 或更高版本：

```bash
uv venv --python 3.10
uv pip install -e '.[dev]'
source .venv/bin/activate
```

也可使用已有虚拟环境执行 `python -m pip install -e '.[dev]'`。
GPU 环境请先按机器环境安装相应 PyTorch，再安装本项目。

## 检查配置

```bash
python -m onemodel.train configs/onetrans_v2/base.gin --dry-run
python -m onemodel.train configs/unir2/base.gin --dry-run
python -m onemodel.evaluate configs/unir2/base.gin --dry-run
```

`--dry-run` 会检查 Gin 绑定并输出数据/训练参数与组件引用，不构造模型、不读取数据、
不创建训练输出。它不验证模型形状或论文算法正确性。未安装包时可在命令前加 `PYTHONPATH=src`。

命令行覆盖采用 Gin 原生绑定，可重复传入：

```bash
python -m onemodel.train configs/unir2/base.gin --dry-run \
  --gin 'TrainingConfig.seed = 43' \
  --gin 'DataConfig.batch_size = 16' \
  --gin "DataConfig.name = 'my_dataset'"
```

配置中的 include 搜索当前配置目录及其上一级。数据和输出路径相对于启动命令的工作目录，
建议统一从项目根目录启动；机器相关路径通过 `--gin` 或未提交的本地配置覆盖。

## 目录

```text
configs/                  Gin 公共配置与模型实验配置
src/onemodel/
  data/                   schemas、datasets、preprocess、collators
  models/                 onetrans_v2、unir2、tokenizers、输出约定
  modules/                通用网络组件
  trainers/               公共 engine 与模型训练流程
  evaluation/             retrieval、ranking、pipeline
  config.py               Gin 参数与配置检查
  train.py                训练入口
  evaluate.py             评估入口
scripts/                  数据和实验辅助脚本
tests/                    配置、入口与训练基础设施检查
docs/                     架构约定与复现记录
datasets/                 本地数据与 SID 产物（忽略 Git）
outputs/                  每次实验的配置、状态与结果（忽略 Git）
```

## 开始实现

1. 在 `data/preprocess` 和 `data/datasets` 接入数据，保留请求、时间、特征、标签有效性。
2. 在 `data/collators` 实现模型 batch；在 `models/<model>/model.py` 和 `layers.py` 实现网络。
3. 在 `objectives.py` 返回 `{'total': scalar_loss, ...}`；在对应 trainer 中补充论文训练阶段、
   参数组或其他差异。当前模型构造会明确抛出 `NotImplementedError`，写模型时移除该占位。
4. 在 `evaluation` 实现推理和评估。evaluator 接收 `(model, dataloader, *, accelerator, config)`，
   自行管理 `torch.no_grad()`、跨进程聚合、候选去重和 label mask。
5. 更新模型 Gin 配置，移除实际使用路径上的占位后启动训练。

```bash
python -m onemodel.train configs/unir2/base.gin
python -m onemodel.evaluate configs/unir2/base.gin \
  --checkpoint outputs/unir2/<run_id>/epoch_0010
```

训练默认每个 epoch 在验证集评估并保存 Accelerator 状态；独立评估入口默认使用测试集。
CLI 也提供 `onemodel-train` 和 `onemodel-evaluate` 两个同等入口。
详细边界和待实现项见 [架构说明](docs/architecture.md)。

## 验证

```bash
python -m unittest discover -s tests -v
ruff check .
```
