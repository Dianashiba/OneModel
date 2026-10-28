训练输出目录，除本说明外全部忽略 Git。

训练入口为每次运行创建 `<experiment>/<UTC timestamp>/`，保存 `config.gin`、
`operative.gin`、`run.json` 和 `epoch_XXXX/` Accelerator 状态目录。
`run.json` 记录配置来源、命令行覆盖、随机种子与代码版本；尚无提交时 `git_sha` 为 null。
