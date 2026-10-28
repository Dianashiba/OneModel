# 开发约定

参考本机 GenRec 的 `models / data / trainers / modules / config` 拆分，使用独立的
`evaluation` 和 `data/collators`，避免 trainer 之间导入数据组装或评估逻辑。

## 组件边界

- `data/preprocess`：读取源数据、时间划分、稳定 ID 映射和特征处理。
- `data/datasets`：读取预处理结果；不要在构造 dataset 时训练 SID 模型。
- `data/collators`：padding、分段信息、请求锚点和监督 mask。
- `models/<paper>`：网络、attention 可见性、共享参数和梯度隔离。
- `models/<paper>/objectives.py`：返回包含标量 `total` 的 loss 字典。
- `models/tokenizers`：SID 构建模型；拟合与导出使用独立脚本和版本化产物。
- `trainers/engine.py`：通用优化循环；论文 trainer 负责阶段切换和参数组等差异。
- `evaluation`：推理流程、候选过滤、SID 还原与指标聚合。

`model(batch) -> ModelOutput` 只是最小约定。各模型可扩展自己的 batch 字段和推理接口，
不要求共享同一个 Transformer 类。通用模块在出现实际复用需求后再提取。

## 两个模型的实现位置

OneTrans-V2 的共享历史、阶段特征和请求锚点从 `UserWindowDataset` 进入；stage visibility、
mixed parameterization 在模型内实现，蒸馏和联合目标在自己的 objectives 中实现。

UniR2 从曝光样本构造用户上下文、SID 轨迹、物品特征和各任务监督 mask；DQ-PCA、排序侧
LoRA 与梯度隔离在模型内实现，生成预热和联合训练在自己的 trainer 中实现。

以上均为待实现项。当前通用循环只提供固定阶段的 AdamW + Accelerate 训练，不包含论文
专用优化器、学习率调度、训练阶段、最佳模型选择、断点续训入口或线上 serving。

## 复现实验约定

1. 固定并共享数据划分、特征版本和 SID 映射；记录量化器拟合的数据范围。
2. 区分真实曝光负例与采样负例、缺失标签与负标签。
3. 检查请求时间之后的行为不可见；不同候选或阶段之间的可见性遵循论文。
4. 单独检查生成、排序、蒸馏损失的梯度流向，包括共享稀疏 embedding 的例外。
5. SID 指标与 item 级指标分开；明确冲突展开、去重、候选集合和历史物品过滤规则。
6. 验证集用于配置/模型选择，测试集用于最终报告；记录多 seed 结果及实现与论文的差异。
