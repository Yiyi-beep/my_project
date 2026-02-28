import os, sys, pickle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

path = os.path.join(ROOT, "checkpoints_v2", "stage3_full_bandit.pkl")
with open(path, "rb") as f:
    bandit = pickle.load(f)

print("type:", type(bandit))
print("alpha:", getattr(bandit, "alpha", None))
print("n_features:", getattr(bandit, "n_features", None))

solvers = getattr(bandit, "solvers", None)
if solvers is not None:
    try:
        print("num solvers:", len(solvers))
        # 打印前 3 个 solver 的类型
        for i, s in enumerate(list(solvers)[:3]):
            print(f"  solver[{i}] type:", type(s))
    except Exception as e:
        print("solvers inspect failed:", e)
