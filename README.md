# colab-qwen38-g4

在 Google Colab **G4 GPU（96 GB 显存）** 上一键跑起 **`dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4`**（262K 上下文 + MTP 投机解码 `130~176 tok/s`），自动通过 Cloudflare Tunnel 反代为 OpenAI 兼容接口。

## 1. 安装并登录 `colab` CLI

```bash
uv tool install google-colab-cli

gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

## 2. 下载脚本并运行

```bash
git clone https://github.com/mizorewww/colab-qwen38-g4.git
cd colab-qwen38-g4

# 交互式启动（按提示回车即可，支持可选挂载 Google Drive 与自动写入 pi 配置）：
python3 run_qwen38_g4.py
```

或者不走交互式向导，直接单条命令全自动跑起来：

```bash
colab run --gpu G4 --keep -s qwen38-g4 --timeout 1800 run_qwen38_g4.py
```

跑完后终端会直接输出反代地址与 Key：

```text
TUNNEL_BASE_URL=https://xxxx.trycloudflare.com/v1
API_KEY=sk-qwen38-...
MODEL_ID=dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4
```

## 3. 在 `pi` 中使用

如果用交互式向导（`python3 run_qwen38_g4.py`），脚本会自动帮你写好 `~/.pi/agent/models.json`。直接运行：

```bash
pi --model "colab-g4/dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4" -p "看一下本机配置"
```

如需手动配置 `~/.pi/agent/models.json`，复制粘贴以下内容（把 `baseUrl` 和 `apiKey` 换成终端输出的值）：

```json
{
  "providers": {
    "colab-g4": {
      "name": "Colab G4 SGLang",
      "baseUrl": "https://xxxx.trycloudflare.com/v1",
      "api": "openai-completions",
      "apiKey": "sk-qwen38-...",
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

## 4. 用完关闭实例（停止计费）

```bash
colab stop -s qwen38-g4
```
