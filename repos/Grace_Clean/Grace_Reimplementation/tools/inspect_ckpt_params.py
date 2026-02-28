import torch

ckpt_path = r"checkpoints_v2\stage3_full_ppo.pth"
obj = torch.load(ckpt_path, map_location="cpu")

sd = None
if isinstance(obj, dict):
    for k in ["model_state_dict","policy_state_dict","state_dict"]:
        if k in obj and isinstance(obj[k], dict):
            sd = obj[k]
            break
    if sd is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
        sd = obj

if sd is None:
    print("Could not find a tensor state_dict-like object. Inspect B1 keys.")
else:
    n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print("State-dict entries:", len(sd))
    print("Total params:", n_params)

    shown = 0
    for name, t in sd.items():
        if hasattr(t, "shape"):
            print(name, tuple(t.shape))
            shown += 1
            if shown >= 10:
                break
