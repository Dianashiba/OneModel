# OneTrans-V2 手写实现 API 表

这是接口设计与学习路线。当前 `layers.py` 和 `model.py` 的网络组装已实现。模型接收编码后的 token 张量；Inputs、原始数据处理、损失、解码搜索等仍是设计，不表示完整训练流程已实现。范围为 OneTrans-V2 的数据、网络、目标函数、解码和评估。

沿用当前 `layers.py` 中的 MixedLinear、GatedAttention、SparseMoE、MixedFFN、OnetransBlock，以及旧项目的 history/stage 计算边界。以下辅助函数可以先与调用者放在同一文件，不要求每个函数单独建文件。

参考：[旧 Block](/Users/gxy/workspace/onetrans/onetrans/blocks.py:93)、[旧模型](/Users/gxy/workspace/onetrans/onetrans/models.py:58)、[旧项目的复现边界](/Users/gxy/workspace/onetrans/docs/fidelity.md)、[当前工程约定](/Users/gxy/workspace/OneModel/docs/architecture.md)。

## 1. 统一符号与数据约定

| 符号/类型 | 约定 |
|---|---|
| `B` | 一个 batch 中的用户窗口数 |
| `E` | 这些窗口展开后的曝光数；通常与 B 不同 |
| `L` | padding 后的历史长度，包含一个始终有效的起始 token |
| `T` | 一次阶段调用中的 token 数；粗排为 1，精排为 F，召回随前缀长度变化 |
| `N` | 底层模块的通用批次维，可以是 B、E 或展开后的 beam 数 |
| `D`、`Hq`、`Hkv`、`dh` | 隐藏维度、Query 头数、KV 头数、每头维度；本版采用 D = Hq × dh，并要求 Hq % Hkv = 0 |
| `type_ids` | 一个 int，或长度为 T 的整数序列；同一次调用的所有样本使用相同的槽位布局 |
| token 类型 0 | 所有历史 token 共用的参数类型；其他 ID 按阶段内的语义槽位分配，与用户 ID、物品 ID 无关 |
| `allowed` | bool mask，True 表示允许注意；形状 `[N, Tq, Tk]` |
| `LayerKV` | `(k, v)`，各为 `[N, Hkv, sequence_length, dh]`；只含当前层需要读取的 K/V |
| `Context` | `cache`（每层历史 KV）、`valid`、`times`；选取曝光前第一维是 B，选取后是 E。召回上下文特征单独按曝光传入 |
| `anchors` | `[E]`，每次曝光最后可见的历史位置；与包含起始 token 的索引约定一致 |

采用旧实现的分开调用方式：每一行处理一个曝光在一个阶段的 token。跨曝光、跨阶段隔离通过分开的 batch 行/调用自然成立，阶段内部仍需因果 mask。

训练中 Context 的 KV 保留计算图，各阶段损失可以回传至共享历史。缓存复用与 stop-gradient 是两件事。

`layers.py` 中所有类及 `OneTransV2` 统一使用 `__init__(self, cfg: OneTransV2Config)`。运行时输入仍由 `forward` 传入。Gin 通过 `OneTransV2.cfg = @OneTransV2Config()` 构造配置，维度设置使用 `OneTransV2Config.dim` 等字段。

类型编号由 config 的属性集中计算：历史为 0，`retrieval_types` 为 CTX、DTp、DTd、各 SID 位置，随后是 `pre_type`、`fine_types`；`num_types` 为总参数组数。默认配置对应历史 0、召回 1–6、粗排 7、精排 8–10。精排数量 3 是复现配置，不代表论文固定数量。

`MixedLinear` 的输入宽度为 `cfg.dim`，输出宽度为 `cfg.projection_dim`（None 时使用 dim）。Attention 内用 `dataclasses.replace` 生成 K/V 的配置副本，不修改原配置。时间周期使用 `temporal_min_period` 和 `temporal_max_period`，默认 60 秒到 30 天，沿用旧复现的工程选择。

| 配置组 | 字段 |
|---|---|
| Block | dim、layers、heads、kv_heads；RMSNorm 沿用其默认 eps |
| MoE | ff_dim、experts（仅 routed experts 数）、top_k、routed_scale |
| token 布局 | fine_tokens、检索各位置的类型 ID、pre_type、fine_types、num_types |
| 时间编码 | temporal_min_period、temporal_max_period；时间单位为秒，head_dim 为偶数 |
| 任务与资产 | pre_tasks、fine_tasks、decision_sizes；artifact_dir 中保存 ID 映射、特征、SID、码本及版本信息 |

## 2. 基础计算模块

当前 RMSNorm 使用 `modules/normalize.py`；`layers.py` 沿用本文件中的 SwiGLU，其构造函数已改为 cfg。`modules/ffn.py` 中另一个通用 SwiGLU 未在本次修改中调整。下表中的独立 split_heads/head_norm/attend 为职责说明，当前计算保留在 GatedAttention 内。

| API | 输入 → 输出 | 职责 |
|---|---|---|
| `RMSNorm.__init__(dim, eps)` | 配置 → 模块 | 创建长度 dim、初始为 1 的可学习缩放参数 |
| `RMSNorm.forward(x)` | `[..., D] → [..., D]` | 最后一维的 RMS 归一化；低精度输入用 FP32 计算统计量 |
| `SwiGLU.__init__(cfg)` | 配置 → 模块 | 使用 dim、ff_dim 创建合并的 gate/up 投影和 down 投影 |
| `SwiGLU.forward(x)` | `[..., D] → [..., D]` | 两路投影、SiLU 门控、逐元素乘、输出投影 |
| `split_heads(x, heads)` | `[N,T,heads×dh] → [N,heads,T,dh]` | 拆出注意力头并调整轴顺序 |
| `head_norm(x, eps)` | `[N,heads,T,dh] → 同形状` | 沿 dh 做 Q/K 的 RMS 归一化；旧参考函数没有可学习缩放 |
| `temporal_rotate(x, times, periods)` | 头表示、`[N,T]` 时间戳 → 同形状 | 根据固定周期旋转 Q/K；周期由配置或模块 buffer 提供 |
| `attend(q, k, v, allowed, history_len=0, query_times=None, history_times=None, periods=None)` | Q/K/V 与可见性 → `[N,Tq,D]` | GQA 的 KV 分组、注意力打分、mask、softmax、加权求和及合头 |

`attend` 的 Q 为 `[N,Hq,Tq,dh]`，K/V 为 `[N,Hkv,Tk,dh]`。当启用时间编码时，只替换 stage→history 区域的打分；历史自注意力、阶段内部注意力保持各自原来的计算。

## 3. 混合参数与注意力/MoE 子模块

建议位置：`models/onetrans_v2/layers.py`。

| API | 输入 → 输出 | 职责 |
|---|---|---|
| `MixedLinear.__init__(cfg)` | 配置 → 模块 | 每个语义类型维护一组投影参数，沿用原注意力的 bias=True |
| `MixedLinear.forward(x, type_ids)` | `[N,T,in_dim] → [N,T,out_dim]` | 根据槽位选择参数；type_ids 为 0 时整段历史共用一组 |
| `GatedAttention.__init__(cfg)` | 配置 → 模块 | 创建 MixedLinear QKVG 投影、输出投影、时间周期 buffer |
| `GatedAttention.project(x_norm, token_type_ids=0)` | `[N,T,D] → (Q,K,V,G)` | 保留四路独立投影、分头、QKNorm；G 为 `[N,T,D]` |
| `GatedAttention.forward(x_norm, mask=None, token_type_ids=0, *, history_kv=None, query_times=None, history_times=None, return_kv=False)` | 归一化输入等 → delta；return_kv=True 时为 `(delta, current_kv)` | 读取历史与当前 token 的 K/V，计算注意力，再应用 sigmoid 门控和输出投影 |
| `SparseMoE.__init__(cfg)` | 配置 → 模块 | 一个始终激活的 shared expert、若干 routed experts、sigmoid router |
| `SparseMoE.forward(x_norm)` | `[N,T,D] → [N,T,D]` | Top-K 路由、选中分数归一化和缩放、专家计算与汇总 |
| `MixedFFN.__init__(cfg)` | 配置 → 模块 | 为各语义类型创建对应的 SparseMoE 参数组 |
| `MixedFFN.forward(x_norm, type_ids)` | `[N,T,D] → [N,T,D]` | 按槽位执行相应的 MoE |

QKVG 的宽度分别为 `D, Hkv×dh, Hkv×dh, D`，GQA 下通常不能四等分。`current_kv` 只返回当前输入产生的 KV，不把传入的历史重复装回缓存。

这里的两个“共享”具有不同范围：历史 token 共用一个 MoE；这个 MoE 内部的 shared expert 始终激活。不同 token 类型的 MoE 不因此自动共享参数。

Attention 接收已经归一化的输入，不再执行一遍 Block 的 pre-norm；QKNorm 仍在 Attention 内部。

## 4. OnetransBlock 与 mask

建议位置：`models/onetrans_v2/layers.py`。保留当前类名 `OnetransBlock`。

| API | 输入 → 输出 | 职责 |
|---|---|---|
| `build_history_mask(valid)` | `[B,L] → [B,L,L]` | 因果可见性与有效历史 key 的交集 |
| `build_stage_mask(visible, stage_len)` | `[E,L]`、T → `[E,T,L+T]` | 可见历史前缀 + 当前阶段内部的因果可见性 |
| `OnetransBlock.__init__(cfg)` | 配置 → 模块 | 两套独立 RMSNorm、GatedAttention、MixedFFN、残差系数 `1/sqrt(2×cfg.layers)` |
| `OnetransBlock.forward(x, attn_mask=None, token_type_ids=0, time_info=None, *, history_kv=None, return_kv=False)` | 输入及可见性 → y；return_kv=True 时为 `(y, current_kv)` | 两个 pre-norm 分支和两次缩放残差；time_info 包含 query_times 和 history_times |
| `OnetransBlock.history(x, valid)` | `[B,L,D]`、有效位 → `(y, history_kv)` | 构造历史 mask，使用类型 0 调用公共计算，保留本层历史 KV |
| `OnetransBlock.stage(x, type_ids, cache, visible, request_times, history_times)` | `[E,T,D]` 与本层历史 → `[E,T,D]` | 构造阶段 mask、展开请求时间、调用公共计算 |

`history` 和 `stage` 是两种输入路径，内部复用 `forward`。公共流程是：

1. `norm1(x)` → Attention → 输出乘残差系数后加回 x。
2. `norm2(x)` → MixedFFN → 输出乘残差系数后加回 x。

直接调用 forward 且不传 mask 时是全可见注意力；训练历史应调用 history 或显式传入因果 mask。stage 的请求时间和历史时间可同时省略以禁用时间旋转。全遮蔽行的注意力加权结果为零，输出投影仍可产生 bias，Block 的 FFN 和残差仍会执行；这不等于将 padding 的最终状态清零。

旧实现 `OneBlock.project` 对应现在的 `norm1 + GatedAttention.project`；旧 `finish` 中的注意力门控/输出投影归入 GatedAttention，两个残差和 FFN 归入 Block.forward。无需再保留一份相同的 project/finish 流程。

## 5. 输入编码与完整模型

Inputs 尚未实现，后续放在 `models/onetrans_v2/inputs.py`，对应旧项目的输入编码职责。当前 `model.py` 只组装已有层和预测头，显式接收编码后的 token，不依赖物品词表或特征分组方案。后续可由包含 Inputs 和 OneTransV2 的包装模块统一训练，确保输入编码参数也注册到优化器中。

| API | 输入 → 输出 | 职责 |
|---|---|---|
| `Inputs.__init__(cfg, artifacts)` | 配置、预处理资产 → 模块 | 创建 embedding、特征投影、决策/SID embedding；加载已有 SID 映射 |
| `Inputs.history(batch)` | 历史字段 → `[B,L,D]` | 编码历史 ID、特征和行为，处理起始 token |
| `Inputs.profile(users)` | `[B] → [B,D]` | 编码候选无关的用户信息；初版可沿用用户 ID embedding |
| `Inputs.pre_candidate(users, items, features)` | 曝光特征 → `[E,1,D]` | 构造粗排单 token；公开数据特征选择写入配置 |
| `Inputs.candidate(users, items, features)` | 曝光特征 → `[E,F,D]` | 构造精排各语义槽位的 token |
| `Inputs.retrieval(profile, decisions=None, sids=None)` | 用户上下文特征、已有前缀 → `[E,T,D]` | 组装 CTX、并行决策、依赖决策、SID 输入 |
| `OneTransV2.__init__(cfg)` | 模型配置 → 模型 | 创建 Block 堆叠、final_norm、决策/SID embedding 与预测头 |
| `OneTransV2.encode(batch)` | history_tokens、history_valid、history_times → Context | 历史逐层更新，一层保存一份 KV；输入须包含始终有效的起始位置 |
| `select_context(context, owners)` | Context、`[E]` owner 索引 → 曝光 Context | 让每条曝光选择所属用户的历史；index_select 保留梯度。beam 扩展也可传入重复 owner |
| `OneTransV2.stage(x, type_ids, context, anchors, times)` | 阶段输入与历史 → `[E,T,D]` | 根据 anchors 得到 visible，按层配对读取历史 KV，最后做 final_norm |
| `OneTransV2.retrieval(ctx, context, anchors, times, decisions=None, sids=None)` | ctx `[E,D]` 和已有前缀 → `[E,T,D]` | 组装 CTX、DTp、DTd、SID 输入并调用 stage；支持推理时逐步传入预测前缀 |
| `OneTransV2.retrieval_logits(hidden)` | 召回前缀隐藏状态 → logits 字典 | CTX 预测三个并行决策；DTp 预测依赖决策；DTd 预测 SID0；SID(i-1) 预测 SIDi |
| `OneTransV2.score(tokens, context, anchors, times, stage='fine')` | 编码好的候选 token → `[E,Cstage]` logits | 粗排读取单 token，精排拼接多个 token 后进入 MLP 头 |
| `OneTransV2.forward(batch)` | Batch → ModelOutput | 历史编码、曝光映射、teacher-forcing 召回、粗排和精排 |

`retrieval` 返回隐藏状态；`retrieval_logits` 将相应位置转成决策/SID logits；解码搜索放在 evaluation 中。`forward` 是联合 teacher forcing 训练入口，训练只输入目标 SID 的前 `sid_levels - 1` 个，最后一个 SID 仅作为预测目标。推理应使用 encode、select_context、retrieval/retrieval_logits、score，不调用读取标签的 forward。

当前 forward 的输入约定（张量应在同一设备，时间单位为秒）：

| 字段 | 形状与要求 |
|---|---|
| history_tokens | `[B,L,D]`，已含起始 token；起始位时间、valid 和 anchor 索引必须对齐 |
| history_valid、history_times | `[B,L]`，valid 为 bool，索引 0 始终有效；空历史仅保留起始位置 |
| owners、anchors、times | `[E]`，owners/anchors 为 long；anchors 是包含起始位置的最后可见历史索引 |
| retrieval_context | `[E,D]`，曝光对应请求的召回上下文特征，不应包含待预测物品或结果标签 |
| pre_tokens、fine_tokens | `[E,1,D]`、`[E,F,D]`，曝光时可用的候选特征 token |
| decisions、sid_targets | long `[E,4]`、`[E,sid_levels]`，用于 teacher forcing 的合法词表索引 |

以下第 6 节中的 ID/原始特征字段是未来数据与 Inputs 的设计，不是当前 model.forward 自动编码的输入。损失和蒸馏仍由 objectives.py 后续实现；本次组装没有实现 μP 初始化/优化器规则。

建议保留现有 ModelOutput 类型，使用如下 logits 键，保证每个字典值仍是 Tensor：

| 字段 | 形状 |
|---|---|
| `logits['sid_0']`、`sid_1`、… | 每层 `[E, codebook_size_i]` |
| `logits['decision_0']`、… | 每个决策维度 `[E, decision_size_i]` |
| `logits['pre']` | `[E, Cpre]` |
| `logits['fine']` | `[E, Cfine]` |
| `states['pre_tasks']` / `states['fine_tasks']` | 各列对应的任务名称，用于目标与蒸馏对齐 |
| `auxiliary_losses` | 初版可为空；只有明确实现额外训练目标后才填入 |

## 6. 数据、SID 和 batch

| 建议位置 | API | 输入 → 输出 / 职责 |
|---|---|---|
| `data/preprocess/` | `prepare(events, items, split_spec, output_dir)` | 原始日志 → 时间划分、稳定 ID 映射、物品资产、UserWindow 文件和 manifest |
| `models/tokenizers/sid.py` | `RQKMeans.fit(train_features)` | 只在允许的训练资产范围拟合并保存码本 |
| 同上 | `RQKMeans.encode(features)` | 特征 → 多层 SID；验证/测试使用既有码本 |
| 同上 | `RQKMeans.save(path)` / `RQKMeans.load(path)` | 码本、维度、版本与拟合范围的持久化 |
| 同上 | `SIDTrie.__init__(item_ids, codes)` | 构建有效 SID 前缀与 SID→物品列表映射 |
| 同上 | `SIDTrie.allowed_next(prefix)` | 返回该前缀下有效的下一层 code |
| 同上 | `SIDTrie.lookup(sid)` | 返回该 SID 对应的全部物品，保留碰撞语义 |
| `data/datasets/exposures.py` | `UserWindowDataset.__init__(root, split)`、`__len__()`、`__getitem__(index)` | 读取已预处理窗口；item 返回现有 UserWindow 对象 |
| `data/collators/onetrans_v2.py` | `OneTransV2Collator.__init__(artifact_dir)` | 加载只读 SID/特征映射与任务元数据，路径通过 Gin 注入 |
| 同上 | `OneTransV2Collator.__call__(samples)` | `list[UserWindow] → Batch`；padding、曝光展开、owner 映射、锚点与监督 mask |

Dataset 与 Collator 只读取已拟合资产。基础张量约定如下；业务特征可以继续增加命名字段。

| Batch 字段 | 形状 / 含义 |
|---|---|
| `users` | `[B]`，窗口所属用户 |
| `history`、`history_times`、`history_valid` | `[B,L]`，历史 ID、时间与有效位 |
| `history_labels` 或行为特征 | `[B,L,Cbehavior]`，历史行为信息 |
| `owners` | `[E]`，曝光对应的窗口索引 |
| `items`、`times`、`anchors` | `[E]`，候选 ID、请求时间、最后可见历史位置 |
| `request_ids`、`exposure_ids` | `[E]` 或等长元数据，用于配对、分组和评估 |
| `pre_features`、`fine_features` | 曝光时可用特征；字段与形状由 tokenizer 的输入规范确定 |
| `sid_targets` | `[E, sid_levels]`，来自只读物品 SID 映射 |
| `decisions`、`decision_mask` | `[E, decision_dims]`，决策监督及有效位 |
| `gen_mask` | `[E]`，哪些曝光参与召回监督 |
| `pre_labels`、`pre_label_mask` | `[E,Cpre]`，粗排监督与有效位 |
| `fine_labels`、`fine_label_mask` | `[E,Cfine]`，精排监督与有效位 |
| `kd_mask` | `[E,Cpre]`，具有同候选、同曝光且匹配精排目标的蒸馏配对 |

初版可沿用旧项目“同一曝光同时产生三阶段输出”的组织。如果以后接入真实阶段日志，应按曝光/候选关联配对；缺失监督用 mask 标识。

历史起始 token 位于 0，始终有效；它保证空历史也有可见 key。anchors 的计算与反馈可用时间、时间切分一起在数据准备中确定，模型只执行既定可见性。

## 7. 目标函数与训练

损失放在 `models/onetrans_v2/objectives.py`。现有训练器只调用 `objective(output, batch)`，因此目标 SID、标签和监督 mask 放入 batch，不在 objective 中依赖整个 model。

| API | 输入 → 输出 | 职责 |
|---|---|---|
| `masked_mean(values, mask)` | 元素损失及有效位 → 标量 | 有效样本平均，空 mask 返回可反传的零；分布式时处理全局有效数 |
| `retrieval_loss(logits, batch)` | SID/决策 logits 与目标 → 标量 | 决策和各 SID 层的交叉熵，使用相应监督 mask |
| `ranking_loss(logits, labels, label_mask, task_weights)` | 排序预测与目标 → 标量 | 各目标的带 mask BCEWithLogits |
| `distillation_loss(student, teacher, mask, temperature)` | 对齐的粗/精排 logits → 标量 | teacher detach、均值中心化、固定温度、软目标 BCE |
| `onetrans_v2_objective(output, batch)` | ModelOutput、Batch → 标量损失字典 | 返回 total、retrieval、pre、fine、kd；各权重通过 Gin 配置 |
| `OneTransV2Trainer.build_optimizer()` | 模型参数/配置 → optimizer | 按参数用途建立优化器组，并落实所选择的宽深缩放规则 |
| `OneTransV2Trainer.training_step(batch)` | Batch → 损失字典 | 优先复用当前基类：model(batch) → objective |
| `OneTransV2Trainer.fit(...)` | DataLoader、输出目录、evaluator → 训练结果/检查点 | 优先复用当前训练循环；工程策略按需要扩展 |

本表的基础 distillation_loss 使用有效配对样本的 batch 均值；多进程训练需聚合统计。旧代码的 EMA 中心化是一项另行选择的工程策略，若要保留，应增设持久化的统计状态并明确更新位置，不能直接把旧版依赖 model 的 loss 搬入现有二参数接口。

当前训练循环管理一个优化器。若进一步实现论文的 embedding/dense 双优化器方案，需同步扩展 engine 的 prepare、step 和保存路径，不能只改 build_optimizer 的返回值。

## 8. 解码与评估

| 建议位置 | API | 输入 → 输出 / 职责 |
|---|---|---|
| `evaluation/retrieval.py` | `retrieve(model, context, anchor, time, trie, beam_size, beta=0, offsets=None)` | 单请求上下文 → 生成 SID、物品与分数；枚举决策、施加决策偏置、约束 beam search、展开碰撞 |
| 同上 | `retrieval_metrics(pred_sids, pred_items, targets, ks)` | SID 与 item 结果 → 分别命名的召回指标，明确候选集合与过滤规则 |
| `evaluation/ranking.py` | `score_candidates(model, context, anchor, time, user, items, features, stage, chunk_size)` | 给定候选 → 目标概率；分块评分、显式选择粗排或精排 |
| 同上 | `ranking_metrics(labels, scores, users, label_mask)` | 全局有效预测 → 每目标 AUC/UAUC 等；不能简单平均各 batch 的 AUC |
| `evaluation/pipeline.py` | `evaluate_pipeline(model, dataloader, *, accelerator, config)` | 保留当前接口；组织召回/粗排/精排、no_grad、跨进程聚合和指标返回 |

真实决策标签只进入训练的 teacher forcing。retrieve 根据模型预测构造前缀，业务 offsets 只影响决策部分的分数。

## 9. 实现顺序与检查点

| 顺序 | 实现范围 | 应能独立检查的性质 |
|---|---|---|
| 1 | RMSNorm、SwiGLU | 形状、公式、有限梯度；归一化缩放参数正确初始化 |
| 2 | split_heads、head_norm、attend、MixedLinear | 轴顺序、GQA 头映射、槽位参数选择、因果可见性 |
| 3 | GatedAttention、时间旋转 | sigmoid gate 在输出投影前；时间编码只作用于指定注意力区域 |
| 4 | SparseMoE、MixedFFN | shared + routed 的汇总、Top-K 权重、token 参数隔离 |
| 5 | OnetransBlock.history/stage | 历史缓存复用等价性；未来历史、其他曝光不能影响当前结果 |
| 6 | Inputs、最小 Batch、OneTransV2.forward | 用户窗口和曝光维度对齐；预测位置与 SID 目标错位正确 |
| 7 | objectives 与现有 Trainer 接线 | 总损失可反传；KD 不更新 teacher 分支；共享历史接收各任务梯度 |
| 8 | 真实数据准备、SID 资产、Dataset/Collator | 时间边界、标签 mask、训练范围、ID 与 SID 映射一致 |
| 9 | retrieve、候选评分、评估 | 推理不读取真实决策；有效 SID 约束、碰撞展开、指标定义一致 |

第 6 步可先用手造的小 batch 运行，再接真实数据；模型阶段测试需要的有效 SID 也可先用小型固定映射。当前配置加载、CLI、通用 Trainer 和运行元数据工具已有骨架，可直接在其接口上接入。
