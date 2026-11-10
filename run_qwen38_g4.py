#!/usr/bin/env -S colab run --gpu G4 --keep -s qwen38-g4 --timeout 1800
"""
One-click & Interactive Deployment Script for Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4
on Google Colab G4 GPU (NVIDIA RTX PRO 6000 Blackwell 96GB) with 262K Context & MTP (NEXTN).

Repository: https://github.com/mizorewww/colab-qwen38-g4

Two ways to run:
  Mode A — Interactive Local Wizard (recommended for first-time users & Google Drive setup):
    python3 run_qwen38_g4.py

  Mode B — Direct One-Command Remote Execution via colab CLI:
    colab run --gpu G4 --keep -s qwen38-g4 --timeout 1800 run_qwen38_g4.py
    # or re-run on an existing session:
    colab exec -s qwen38-g4 --timeout 1800 -f run_qwen38_g4.py
"""

import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "dealignai/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4")
LOCAL_MODEL_DIR = "/content/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4"
LOCAL_MTP_DRAFT_DIR = "/content/Qwen3.8-Flash-Next-MTP-Draft"
DRIVE_MODEL_DIR = "/content/drive/MyDrive/models/Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4"

GITHUB_RELEASE_BASE = os.environ.get(
    "KERNEL_CACHE_BASE_URL",
    "https://github.com/mizorewww/colab-qwen38-g4/releases/download/v1.0.0",
)
API_KEY = os.environ.get("COLAB_SGLANG_API_KEY", "sk-qwen38-g4-cyber-9f8a7b6c5d4e3f2a")
ENABLE_MTP = os.environ.get("ENABLE_MTP", "1") not in ("0", "false", "False", "no")
WEIGHT_SOURCE = os.environ.get("WEIGHT_SOURCE", "auto")  # auto | hf | drive
BACKUP_TO_DRIVE = os.environ.get("BACKUP_TO_DRIVE", "0") in ("1", "true", "True", "yes")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_running_on_colab_vm() -> bool:
    return os.path.exists("/var/colab/hostname") or (
        os.path.isdir("/content") and sys.platform.startswith("linux")
    )


# ============================================================================
# Remote Colab VM Execution Logic
# ============================================================================

def run_cmd(cmd: str, log_path: str = "/content/cmd.log", check: bool = True) -> int:
    with open(log_path, "a") as f:
        p = subprocess.run(
            cmd,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    if check and p.returncode != 0:
        tail = subprocess.getoutput(f"tail -n 30 {log_path}")
        raise RuntimeError(f"Command failed (rc={p.returncode}): {cmd}\n{tail}")
    return p.returncode


def step1_install_and_restore_cache() -> float:
    t0 = time.time()
    log("Step 1/4: Installing SGLang >= 0.5.23, ModelOpt, cloudflared & restoring precompiled SM120 kernels...")

    with open("/tmp/overrides.txt", "w") as f:
        f.write("outlines-core>=0.2.11\n")

    run_cmd(
        'export PATH="/tools/node/bin:/usr/local/bin:$PATH" && '
        'if ! command -v cloudflared >/dev/null 2>&1; then '
        '  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && '
        '  chmod +x /usr/local/bin/cloudflared; '
        'fi && '
        'if [ ! -f /usr/bin/ninja ]; then apt-get update -qq && apt-get install -y -qq ninja-build; fi',
        log_path="/content/setup_env.log",
    )
    run_cmd(
        'uv pip install --system --overrides /tmp/overrides.txt "sglang[all]>=0.5.23" nvidia-modelopt hf_transfer',
        log_path="/content/setup_env.log",
    )

    moe_so_dir = "/root/.cache/sglang/.cache/flashinfer/0.7.2/120f/cached_ops/fused_moe_120"
    moe_so_path = f"{moe_so_dir}/fused_moe_120.so"
    os.makedirs(moe_so_dir, exist_ok=True)

    if not os.path.exists(moe_so_path) or os.path.getsize(moe_so_path) < 1000000:
        run_cmd(
            f'curl -fsSL "{GITHUB_RELEASE_BASE}/fused_moe_120.so.xz" -o /tmp/fused_moe_120.so.xz',
            log_path="/content/setup_env.log",
            check=False,
        )
        if os.path.exists("/tmp/fused_moe_120.so.xz") and os.path.getsize("/tmp/fused_moe_120.so.xz") > 1000000:
            run_cmd(f"xz -dc /tmp/fused_moe_120.so.xz > {moe_so_path} && chmod +x {moe_so_path} && rm -f /tmp/fused_moe_120.so.xz")
            log("Restored precompiled fused_moe_120.so (60MB) from GitHub Release!")

    if not os.path.isdir("/root/.cache/sglang/triton"):
        run_cmd(
            f'curl -fsSL "{GITHUB_RELEASE_BASE}/other_kernels.tar.xz" -o /tmp/other_kernels.tar.xz && '
            f'tar -xJf /tmp/other_kernels.tar.xz -C /root && rm -f /tmp/other_kernels.tar.xz',
            log_path="/content/setup_env.log",
            check=False,
        )
        log("Restored precompiled Triton & FlashInfer auxiliary kernels from GitHub Release!")

    # Install ninja wrapper so FlashInfer immediately uses precompiled .so without rebuilding 96 .cuda.o files
    ninja_wrapper = """#!/usr/bin/env python3
import glob, os, sys
if "-C" in sys.argv:
    d = sys.argv[sys.argv.index("-C") + 1]
    if any(os.path.getsize(p) > 10240 for p in glob.glob(os.path.join(d, "*.so"))):
        sys.exit(0)
os.execv("/usr/bin/ninja", ["/usr/bin/ninja"] + sys.argv[1:])
"""
    if os.path.exists("/usr/local/bin/ninja"):
        os.unlink("/usr/local/bin/ninja")
    with open("/usr/local/bin/ninja", "w") as f:
        f.write(ninja_wrapper)
    os.chmod("/usr/local/bin/ninja", 0o755)
    dt = time.time() - t0
    log(f"Step 1/4 complete in {dt:.1f}s!")
    return dt


def prepare_mtp_draft_dir():
    """Create a 3-shard symlink directory containing only MTP draft weights so MTP draft loads in ~1.8s instead of ~202s!"""
    os.makedirs(LOCAL_MTP_DRAFT_DIR, exist_ok=True)
    idx_path = os.path.join(LOCAL_MODEL_DIR, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        return
    with open(idx_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    mtp_wm = {k: v for k, v in wm.items() if "mtp" in k}
    mtp_shards = sorted(set(mtp_wm.values()))

    for fname in os.listdir(LOCAL_MODEL_DIR):
        if fname.startswith(".") or fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        src = os.path.join(LOCAL_MODEL_DIR, fname)
        dst = os.path.join(LOCAL_MTP_DRAFT_DIR, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    for shard in mtp_shards:
        src = os.path.join(LOCAL_MODEL_DIR, shard)
        dst = os.path.join(LOCAL_MTP_DRAFT_DIR, shard)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    with open(os.path.join(LOCAL_MTP_DRAFT_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": idx.get("metadata", {}), "weight_map": mtp_wm}, f, indent=2)


def copy_from_drive_to_nvme():
    log(f"Copying weights from Google Drive ({DRIVE_MODEL_DIR}) to local NVMe ({LOCAL_MODEL_DIR}) with 16 threads...")
    all_files = [f for f in os.listdir(DRIVE_MODEL_DIR) if not f.startswith(".")]

    def copy_one(fname):
        src = os.path.join(DRIVE_MODEL_DIR, fname)
        dst = os.path.join(LOCAL_MODEL_DIR, fname)
        if os.path.isdir(src):
            return
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return
        with open(src, "rb") as fsrc, open(dst + ".tmp", "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
        os.replace(dst + ".tmp", dst)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(copy_one, all_files))


def download_from_hf_to_nvme():
    log(f"Downloading weights to local NVMe {LOCAL_MODEL_DIR} via HuggingFace Xet (~400 MB/s)...")
    with open("/content/hf_download.log", "w") as f:
        p = subprocess.Popen(
            f'export PATH="/tools/node/bin:/usr/local/bin:$PATH" && '
            f'HF_XET_HIGH_PERFORMANCE=1 hf download {MODEL_ID} --local-dir {LOCAL_MODEL_DIR}',
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    # Monitor hf_download.log; hf-xet can hang at C++ atexit after printing "Downloaded", so kill once all 206 shards are on disk
    last_prog = ""
    while p.poll() is None:
        dl_txt = subprocess.getoutput("cat /content/hf_download.log 2>/dev/null | tr '\r' '\n'")
        lines = [l for l in dl_txt.splitlines() if l.strip()]
        if lines and lines[-1] != last_prog:
            last_prog = lines[-1]
            if "Fetching" in last_prog or "Downloaded" in last_prog:
                log(f"  [hf] {last_prog[:100]}")
        if "Downloaded" in dl_txt and len(glob.glob(os.path.join(LOCAL_MODEL_DIR, "*.safetensors"))) >= 206:
            subprocess.run(f"kill -9 -{p.pid} 2>/dev/null || pkill -9 -f 'hf download' || true", shell=True)
            break
        time.sleep(5)


def step2_prepare_weights() -> float:
    t0 = time.time()
    log("Step 2/4: Preparing model weights on local NVMe (/content/...)...")
    if os.path.islink(LOCAL_MODEL_DIR):
        os.unlink(LOCAL_MODEL_DIR)
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    existing_shards = glob.glob(os.path.join(LOCAL_MODEL_DIR, "*.safetensors"))
    if len(existing_shards) >= 206 and os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        prepare_mtp_draft_dir()
        dt = time.time() - t0
        log(f"All {len(existing_shards)} safetensors shards + 3-shard MTP draft dir ready on local NVMe ({dt:.1f}s)!")
        return dt

    drive_shards = glob.glob(os.path.join(DRIVE_MODEL_DIR, "*.safetensors")) if os.path.isdir(DRIVE_MODEL_DIR) else []

    if WEIGHT_SOURCE == "drive" and len(drive_shards) >= 206:
        copy_from_drive_to_nvme()
    else:
        if os.path.isdir(DRIVE_MODEL_DIR):
            for fname in os.listdir(DRIVE_MODEL_DIR):
                if not fname.startswith(".") and not fname.endswith(".safetensors"):
                    src = os.path.join(DRIVE_MODEL_DIR, fname)
                    dst = os.path.join(LOCAL_MODEL_DIR, fname)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)
        download_from_hf_to_nvme()

    prepare_mtp_draft_dir()

    if BACKUP_TO_DRIVE and os.path.isdir("/content/drive/MyDrive") and len(drive_shards) < 206:
        log(f"Starting background backup of weights to Google Drive ({DRIVE_MODEL_DIR})...")
        os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)
        subprocess.Popen(
            ["nohup", "cp", "-rn", f"{LOCAL_MODEL_DIR}/.", f"{DRIVE_MODEL_DIR}/"],
            stdout=open("/content/drive_backup.log", "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    dt = time.time() - t0
    log(f"Step 2/4 complete: 206 shards + 3-shard MTP draft dir ready on local NVMe in {dt:.1f}s!")
    return dt


def step3_start_tunnel() -> str:
    log("Step 3/4: Starting Cloudflare Quick Tunnel (trycloudflare.com)...")
    if not subprocess.getoutput("pgrep -f '[c]loudflared'"):
        cf_sh = """#!/bin/bash
export PATH="/tools/node/bin:/usr/local/bin:$PATH"
exec cloudflared --no-autoupdate tunnel --url http://localhost:30000
"""
        with open("/content/start_cf.sh", "w") as f:
            f.write(cf_sh)
        subprocess.Popen(
            ["nohup", "bash", "/content/start_cf.sh"],
            stdout=open("/content/cf_tunnel.log", "w"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    tunnel_url = None
    for _ in range(30):
        log_txt = subprocess.getoutput("cat /content/cf_tunnel.log 2>/dev/null")
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", log_txt)
        if m:
            tunnel_url = m.group(0)
            break
        time.sleep(1)

    log(f"Step 3/4 complete: Cloudflare Tunnel active at {tunnel_url}")
    return tunnel_url


def step4_start_sglang_and_wait() -> float:
    mode_label = "MTP Speculative Decoding (NEXTN)" if ENABLE_MTP else "Standard High-Capacity (No-MTP)"
    log(f"Step 4/4: Launching SGLang Server [{mode_label} + 262K Context]...")
    subprocess.run("pkill -9 -f '[s]glang' || true", shell=True)

    spec_flags = ""
    if ENABLE_MTP:
        spec_flags = f"""  --speculative-draft-model-path {LOCAL_MTP_DRAFT_DIR} \\
  --speculative-algorithm NEXTN \\
  --speculative-num-steps 3 \\
  --speculative-eagle-topk 1 \\
  --speculative-num-draft-tokens 4 \\
  --speculative-draft-kv-cache-dtype fp8_e4m3 \\
  --enable-linear-replayssm-spec \\
  --max-running-requests 4 \\
  --max-mamba-cache-size 20 \\
  --chunked-prefill-size 4096 \\
  --mem-fraction-static 0.95 \\"""
    else:
        spec_flags = """  --max-running-requests 8 \\
  --max-mamba-cache-size 24 \\
  --chunked-prefill-size 8192 \\
  --mem-fraction-static 0.93 \\"""

    sglang_sh = f"""#!/bin/bash
set -ex
export PATH="/tools/node/bin:/usr/local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
export FLASHINFER_JIT_MAX_JOBS=4

exec sglang serve \\
  --model-path {LOCAL_MODEL_DIR} \\
  --served-model-name {MODEL_ID} \\
  --tp-size 1 \\
  --quantization modelopt_fp4 \\
  --attention-backend flashinfer \\
  --kv-cache-dtype fp8_e4m3 \\
  --mamba-ssm-dtype bfloat16 \\
  --ple-offload-embedding \\
  --mamba-radix-cache-strategy extra_buffer_lazy \\
{spec_flags}
  --disable-flashinfer-autotune \\
  --weight-loader-drop-cache-after-load \\
  --cuda-graph-max-bs-decode 4 \\
  --disable-prefill-cuda-graph \\
  --reasoning-parser qwen3 \\
  --tool-call-parser qwen3_coder \\
  --model-loader-extra-config '{{"enable_multithread_load": true}}' \\
  --api-key {API_KEY} \\
  --host 0.0.0.0 \\
  --port 30000
"""
    with open("/content/start_sglang.sh", "w") as f:
        f.write(sglang_sh)
    subprocess.Popen(
        ["nohup", "bash", "/content/start_sglang.sh"],
        stdout=open("/content/sglang.log", "w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )

    t0 = time.time()
    last_line = ""
    while time.time() - t0 < 1200:
        sgl_txt = subprocess.getoutput("cat /content/sglang.log 2>/dev/null | tr '\r' '\n'")
        lines = sgl_txt.splitlines()
        if "The server is fired up and ready to roll!" in sgl_txt:
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:30000/v1/models",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                )
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        dt = time.time() - t0
                        log(f"Step 4/4 complete: SGLang server is READY in {dt:.1f}s!")
                        return dt
            except Exception:
                pass

        if lines and lines[-1] != last_line:
            last_line = lines[-1]
            log(f"  [sglang] {last_line[:120]}")
        if any("Rank 0 scheduler died" in l or "SIGQUIT received" in l for l in lines[-15:]):
            raise RuntimeError("SGLang scheduler died during initialization:\n" + "\n".join(lines[-35:]))
        time.sleep(10)

    raise TimeoutError("Timed out waiting for SGLang server to become ready.")


def run_remote_vm_main():
    t_start = time.time()
    dt_env = step1_install_and_restore_cache()
    dt_weights = step2_prepare_weights()
    tunnel_url = step3_start_tunnel()
    dt_sglang = step4_start_sglang_and_wait()
    total_dt = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"DEPLOYMENT SUCCESSFUL (total: {total_dt:.1f}s | env: {dt_env:.1f}s | weights: {dt_weights:.1f}s | sglang+warmup: {dt_sglang:.1f}s)")
    print(f"TUNNEL_BASE_URL={tunnel_url}/v1")
    print(f"API_KEY={API_KEY}")
    print(f"MODEL_ID={MODEL_ID}")
    print("=" * 72, flush=True)


# ============================================================================
# Local Interactive Wizard Mode (when invoked via `python3 run_qwen38_g4.py`)
# ============================================================================

def prompt_choice(prompt_text: str, default: str) -> str:
    if not sys.stdin.isatty() or "--yes" in sys.argv or "-y" in sys.argv:
        return default
    try:
        val = input(f"{prompt_text} [{default}]: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)


def update_pi_models_json(base_url: str, api_key: str):
    pi_models_path = Path.home() / ".pi" / "agent" / "models.json"
    pi_models_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(pi_models_path.read_text()) if pi_models_path.exists() else {"providers": {}}
    data.setdefault("providers", {})["colab-g4"] = {
        "name": "Colab G4 SGLang",
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": api_key,
        "models": [
            {
                "id": MODEL_ID,
                "name": "Qwen 3.8 Flash Next Cybersecurity NVFP4 (Colab G4)",
                "reasoning": True,
                "input": ["text", "image"],
                "contextWindow": 262144,
                "maxTokens": 65536,
                "compat": {
                    "supportsStore": False,
                    "supportsDeveloperRole": False,
                },
                "thinkingLevelMap": {
                    "off": None,
                    "minimal": "low",
                    "low": "low",
                    "medium": "medium",
                    "high": "xhigh",
                    "xhigh": "xhigh",
                    "max": "xhigh",
                },
                "samplingParams": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                },
                "samplingParamsByThinkingLevel": {
                    "off": {
                        "temperature": 0.7,
                        "top_p": 0.8,
                    }
                },
            }
        ],
    }
    pi_models_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"✓ Updated {pi_models_path} (provider: colab-g4)")


def run_local_interactive_wizard():
    print("=" * 72)
    print(" Qwen3.8-Flash-Next-CYBERSECURITY-NVFP4 · Google Colab G4 (96GB) Wizard")
    print("=" * 72)

    if not shutil.which("colab"):
        print("Error: 'colab' CLI not found. Install it first via:")
        print("  uv tool install google-colab-cli")
        sys.exit(1)

    session_name = prompt_choice("1. Colab session name", "qwen38-g4")

    print("\n2. Weight loading & Google Drive strategy:")
    print("   [1] Fast Direct NVMe (Recommended for first-time users: ~400 MB/s via HF Xet, ~6.5 min, no Drive quota needed)")
    print("   [2] Mount Google Drive (Reuse cached weights from Drive, or download & back up to Drive if first time)")
    drive_choice = prompt_choice("   Select strategy (1 or 2)", "1")

    print("\n3. Inference acceleration mode:")
    print("   [1] Enable MTP Speculative Decoding (NEXTN: 120~176 tok/s, 472K FP8 KV cache pool, 262K context)")
    print("   [2] High-Capacity Mode (No-MTP: ~81 tok/s, 946K FP8 KV cache pool, higher concurrency)")
    mtp_choice = prompt_choice("   Select mode (1 or 2)", "1")
    enable_mtp = "0" if mtp_choice == "2" else "1"

    default_key = os.environ.get("COLAB_SGLANG_API_KEY", f"sk-qwen38-{secrets.token_hex(8)}")
    api_key = prompt_choice("\n4. SGLang API Key (for Cloudflare Tunnel protection)", default_key)

    sess_out = subprocess.getoutput("colab sessions")
    if f"[{session_name}]" not in sess_out:
        print(f"\n-> Allocating Google Colab G4 GPU instance '{session_name}'...")
        subprocess.run(["colab", "new", "-s", session_name, "--gpu", "G4"], check=True)
    else:
        print(f"\n-> Reusing existing Colab G4 session '{session_name}'.")

    weight_source = "hf"
    backup_to_drive = "0"
    if drive_choice == "2":
        print(f"\n-> Mounting Google Drive on session '{session_name}' (follow browser prompt if asked)...")
        subprocess.run(["colab", "drivemount", "-s", session_name], check=True)
        reuse_choice = prompt_choice(
            "   If Drive does not have the 126 GiB weights yet, back them up to Drive in background? (y/n)",
            "y",
        )
        weight_source = "auto"
        backup_to_drive = "1" if reuse_choice.lower().startswith("y") else "0"

    print(f"\n-> Executing deployment on remote Colab G4 VM '{session_name}'...")
    cmd = [
        "colab", "exec", "-s", session_name,
        "--timeout", "1800",
        "--env", f"COLAB_SGLANG_API_KEY={api_key}",
        "--env", f"ENABLE_MTP={enable_mtp}",
        "--env", f"WEIGHT_SOURCE={weight_source}",
        "--env", f"BACKUP_TO_DRIVE={backup_to_drive}",
        "-f", os.path.abspath(__file__),
    ]

    tunnel_base_url = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("TUNNEL_BASE_URL="):
            tunnel_base_url = line.split("=", 1)[1].strip()
    proc.wait()

    if proc.returncode != 0:
        sys.exit(proc.returncode)

    if tunnel_base_url:
        pi_choice = prompt_choice("\n5. Update ~/.pi/agent/models.json with this endpoint? (y/n)", "y")
        if pi_choice.lower().startswith("y"):
            update_pi_models_json(tunnel_base_url, api_key)
            print("\nTest with pi:")
            print(f'  pi --model "colab-g4/{MODEL_ID}" -p "看一下本机配置"')


if __name__ == "__main__":
    if is_running_on_colab_vm():
        run_remote_vm_main()
    else:
        run_local_interactive_wizard()
