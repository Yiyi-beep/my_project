import torch, os

ckpt_path = r"checkpoints_v2\stage3_full_ppo.pth"
obj = torch.load(ckpt_path, map_location="cpu")

print("Loaded type:", type(obj))

if isinstance(obj, dict):
    keys = list(obj.keys())
    print("Top-level keys (first 30):", keys[:30])

    for k in ["state_dict","model_state_dict","policy_state_dict","optimizer_state_dict","config","args","step","epoch"]:
        if k in obj:
            print(f"Contains key: {k}, type={type(obj[k])}")

print("File size (bytes):", os.path.getsize(ckpt_path))
