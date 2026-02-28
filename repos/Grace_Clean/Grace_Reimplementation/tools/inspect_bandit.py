import os, sys, pickle

# 强制把 repo root 加入 sys.path，避免找不到 grace_v2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("ROOT =", ROOT)
print("sys.path[0:3] =", sys.path[:3])

# 先验证一下 grace_v2 确实可导入（这一步很关键）
import grace_v2
import inspect
print("grace_v2 imported from:", inspect.getfile(grace_v2))

path = os.path.join(ROOT, "checkpoints_v2", "stage3_full_bandit.pkl")
with open(path, "rb") as f:
    obj = pickle.load(f)

print("Loaded type:", type(obj))

if isinstance(obj, dict):
    print("dict keys (first 50):", list(obj.keys())[:50])
    for k in list(obj.keys())[:10]:
        v = obj[k]
        if isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")
        else:
            print(f"  {k}: {type(v)}")
else:
    attrs = [a for a in dir(obj) if not a.startswith("_")]
    print("attrs (first 50):", attrs[:50])
