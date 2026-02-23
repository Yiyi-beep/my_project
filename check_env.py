import os, json, time
import torch

os.makedirs("runs/logs", exist_ok=True)
os.makedirs("runs/outputs", exist_ok=True)

info = {
    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "torch": torch.__version__,
    "torch_cuda_compiled": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}

# 额外：跑一次 GPU 计算，证明真在跑
if torch.cuda.is_available():
    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    info["gpu_compute_mean"] = float(y.mean().item())

print(json.dumps(info, indent=2, ensure_ascii=False))

with open("runs/outputs/env_check.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
