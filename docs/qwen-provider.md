# 阶段 1：百炼模型配置依据

核对日期：2026-08-10。这里记录阶段 1 实际采用的公开配置依据；运行时值以 `config/default.toml` 为准，业务代码只引用 `advice` 模型角色。

## 当前选择

- 模型角色：`advice`；当前配置为 `qwen3.7-flash`。
- 地域：华北 2（北京），配置中的 OpenAI 兼容 Base URL 为 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
- 接口：`POST /chat/completions`，使用 `response_format={"type":"json_object"}`，并在提示中明确要求 JSON。
- 计费估算：输入不超过 32K Token 时，输入 ¥0.2/百万 Token、输出 ¥0.8/百万 Token。只按响应中的实际输入/输出 Token 估算，不把免费额度或活动折扣计入。
- 密钥：只从进程环境或仓库根目录被 Git 忽略的 `.env` 读取 `DASHSCOPE_API_KEY`，不写入配置对象、日志、异常或 UI。

官方资料：

- [模型大全与可用地域](https://help.aliyun.com/zh/model-studio/models)
- [模型调用价格](https://help.aliyun.com/zh/model-studio/model-pricing)
- [OpenAI 兼容 Base URL](https://help.aliyun.com/zh/model-studio/base-url)
- [千问结构化输出](https://help.aliyun.com/zh/model-studio/qwen-structured-output)

## 安全边界

悬浮窗默认使用本地模拟模型，只有显式添加 `--online` 且找到密钥时才联网。`api-smoke` 还要求显式添加 `--confirm-charge`，防止误调用。即使云端返回可解析 JSON，程序仍会校验字段集合、长度、置信度、状态序号以及引用是否来自本次本地检索；校验失败的内容禁止展示。
