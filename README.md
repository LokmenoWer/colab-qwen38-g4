# colab-qwen38-g4

在单张 **Google Colab G4 GPU（NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM）** 上一键部署 **`dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4`**（176B 总参数 / 6B 激活参数，支持 **262K 满上下文** 与 **MTP `NEXTN` 投机解码**），并通过 Cloudflare Tunnel 安全反代到本地 OpenAI 兼容客户端（如 `pi`、Cursor、Claude Code / OpenAI SDK）。

## 核心特性与实测指标

| 指标 | 开启 MTP 投机解码 (`NEXTN`, 默认) | 纯容量模式 (`No-MTP`) |
| --- | --- | --- |
| **硬件要求** | Google Colab `G4`（96 GB VRAM + 176 GB RAM + 236 GB NVMe） | 同左 |
| **单流生成速度** | **`120 ~ 176.6 tokens/s`**（Accept len `2.3 ~ 3.3`） | `~80.6 tokens/s` |
| **FP8 KV Cache 池** | **`472,384 tokens`**（262K 上下文的 **1.8×**） | **`946,624 tokens`**（262K 上下文的 **3.6×**） |
| **空闲显存余量** | `5.91 GB`（4096 chunked-prefill 峰值仅需 ~2.3 GB） | `5.02 GB` |
| **Host RAM 占用** | `74 GiB / 176 GiB`（47.7 GiB FP8 PLE N-gram 表锁页卸载） | 同左 |
| **SGLang 启动 + Warmup** | **`~480 s`（8 分 00 秒）** | `~450 s`（7 分 30 秒） |

### 关键工程优化

1. **预编译 Blackwell `SM120` 算子直灌（省去 15 分钟 `nvcc`/`cicc` 编译与 OOM）**：
   首次在 RTX PRO 6000 (`sm_120f`) 上运行 NVFP4 MoE 时，FlashInfer 默认会并发拉起 20+ 个 `cicc` 进程编译 96 个 CUTLASS 模板，耗时 15 分钟且极易撑爆 176 GB 内存触发 Linux OOM Killer。本仓库 Release 已内置预编译好的 `fused_moe_120.so` 与 Triton 缓存，启动时 2 秒拉取并零编译直通 CUDA Graph 捕获。
2. **3-Shard MTP 极速草稿目录（省去 200 秒重复扫描）**：
   `Qwen3.8-Flash-Next` 共有 206 个 `.safetensors` 分片，但 1 层 MTP 草稿头（`mtp.*`）仅存在于最后 3 个 BF16 分片中。脚本自动构建仅含这 3 个分片的符号链接目录传给 `--speculative-draft-model-path`，将 MTP 草稿头加载时间从 `202.0s` 压缩至 **`1.8s`**。
3. **支持交互式首次向导 & Google Drive 按需挂载/备份**：
   首次使用的用户无需预先准备 Google Drive 权重，可直接通过 HuggingFace Xet CDN（`~400 MB/s`）6.5 分钟拉取至实例本地 NVMe；也可在交互式向导中选择挂载 Google Drive 自动备份或复用。

---

## 前置准备

1. **Google Colab Pro / Pro+ 账号**（具备 G4 GPU 配额，约消耗 8~9 Compute Units/小时）。
2. **安装并登录 `colab` CLI**：
   ```bash
   uv tool install google-colab-cli
   # 首次使用通过 ADC 或 OAuth 登录：
   gcloud auth application-default login \
     --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
   colab sessions
   ```
3. **克隆本仓库**：
   ```bash
   git clone https://github.com/mizorewww/colab-qwen38-g4.git
   cd colab-qwen38-g4
   ```

---

## 快速开始

### 方式一：本地交互式向导（推荐首次使用 / 需要挂载 Google Drive 的用户）

直接在本地终端运行：

```bash
python3 run_qwen38_g4.py
```

向导会依次引导你选择：
1. **Colab 会话名称**（默认 `qwen38-g4`，若不存在则自动创建 G4 实例）。
2. **权重加载与 Google Drive 策略**：
   - `[1] Fast Direct NVMe`（推荐首次使用）：无需挂载 Google Drive，直接通过 `HF_XET_HIGH_PERFORMANCE=1` 以 ~400 MB/s 下载到 Colab 本地 NVMe（约 6.5 分钟），不占云盘空间。
   - `[2] Mount Google Drive`：自动调用 `colab drivemount` 在终端打印授权链接引导完成挂载；若云盘已有权重则直接同步到本地 NVMe，若首次使用无权重则支持在后台自动备份一份到 `/content/drive/MyDrive/models/...`（需云盘剩余空间 ≥ 136 GB）。
3. **推理加速模式**：默认开启 `[1] MTP Speculative Decoding (NEXTN)`（`120~176 tok/s`）。
4. **API Key 保护**：默认自动生成随机高强度 Bearer Token 保护公网 `trycloudflare.com` 隧道。
5. **自动写入 `pi` 配置**：部署完成后可一键写入 `~/.pi/agent/models.json`。

*(如需免交互全部使用默认配置，可运行 `python3 run_qwen38_g4.py -y`)*

---

### 方式二：仅用 `colab` CLI 单条命令直接跑（适合自动化 / 脚本直跑）

`run_qwen38_g4.py` 同时支持在远端 Colab VM 内直接执行（脚本头部已配置 `#!/usr/bin/env -S colab run` shebang）：

```bash
# 1. 从零分配 G4 实例并在远端一键拉起服务（--keep 保持实例存活）：
colab run --gpu G4 --keep -s qwen38-g4 --timeout 1800 run_qwen38_g4.py

# 或者直接作为可执行文件运行：
./run_qwen38_g4.py

# 2. 若实例已存在，在已有会话上重跑/重启服务：
colab exec -s qwen38-g4 --timeout 1800 -f run_qwen38_g4.py
```

可通过 `--env` 传入自定义环境变量：
- `--env COLAB_SGLANG_API_KEY=sk-your-custom-key`：自定义 API Key
- `--env ENABLE_MTP=0`：关闭 MTP 投机解码，切换至 946K KV 池容量模式
- `--env WEIGHT_SOURCE=drive`：优先从已挂载的 `/content/drive/MyDrive/models/...` 拷贝权重
- `--env BACKUP_TO_DRIVE=1`：下载完成后在后台自动备份权重到已挂载的 Google Drive

---

## 接入客户端测试

脚本执行完毕后会输出：

```text
========================================================================
DEPLOYMENT SUCCESSFUL (total: 480.4s | env: 0.2s | weights: 0.1s | sglang+warmup: 480.1s)
TUNNEL_BASE_URL=https://xxxx-xxxx-xxxx.trycloudflare.com/v1
API_KEY=sk-qwen38-g4-cyber-9f8a7b6c5d4e3f2a
MODEL_ID=dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4
========================================================================
```

### 1. 使用 `curl` 测试

```bash
curl -s "$TUNNEL_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4",
    "messages": [{"role": "user", "content": "你好，请介绍一下你的架构。"}],
    "chat_template_kwargs": {"reasoning_effort": "low"}
  }'
```

### 2. 接入 `pi` Coding Agent（`~/.pi/agent/models.json`）

如果使用方式一（本地向导），脚本会自动为你更新 `~/.pi/agent/models.json`。若手动配置，可将以下条目加入 `~/.pi/agent/models.json`（注意 `compat.supportsDeveloperRole: false` 与 `thinkingLevelMap` 必填，因为模型模板仅接受 `low | medium | xhigh`）：

```json
{
  "providers": {
    "colab-g4": {
      "name": "Colab G4 SGLang",
      "baseUrl": "https://xxxx-xxxx-xxxx.trycloudflare.com/v1",
      "api": "openai-completions",
      "apiKey": "sk-qwen38-g4-cyber-9f8a7b6c5d4e3f2a",
      "models": [
        {
          "id": "dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4",
          "name": "Qwen 3.8 Flash Next Cybersecurity NVFP4 (Colab G4)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 262144,
          "maxTokens": 65536,
          "compat": {
            "supportsStore": false,
            "supportsDeveloperRole": false
          },
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "xhigh",
            "xhigh": "xhigh",
            "max": "xhigh"
          },
          "samplingParams": {
            "temperature": 1.0,
            "top_p": 0.95
          },
          "samplingParamsByThinkingLevel": {
            "off": {
              "temperature": 0.7,
              "top_p": 0.8
            }
          }
        }
      ]
    }
  }
}
```

运行测试：

```bash
pi --model "colab-g4/dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4" -p "看一下本机配置"
```

---

## 用完释放实例（停止计费）

Colab G4 实例按时长消耗 Compute Units，使用完毕后请务必执行：

```bash
colab stop -s qwen38-g4
colab sessions  # 确认无残留实例
```

## License

MIT
