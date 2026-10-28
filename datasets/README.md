本地数据目录，除本说明外全部忽略 Git。

建议布局：`raw/<dataset>/`、`processed/<dataset>/<version>/`、`artifacts/<dataset>/`。
保存划分清单、ID 映射、特征说明、SID 双向映射及其版本；模型之间复用同一份预处理结果。
