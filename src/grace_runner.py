import os
import sys
import subprocess
from pathlib import Path

import pandas as pd

# === 你的项目根目录 ===
PROJECT_ROOT = Path(r"D:\SparkCup\my_project")

# === Grace 仓库根目录（注意有空格）===
GRACE_ROOT = Path(r"D:\SparkCup\my_project\repos\Grace_Reimplementation\Grace_Reimplementation2 2.3")

E0_EVAL = GRACE_ROOT / "grace_v2" / "eval_E0_static_burst.py"
E0_PLOT = GRACE_ROOT / "grace_v2" / "plot_E0_grouped.py"


def run_cmd(args, log_path: Path, cwd: Path):
    """运行命令并把stdout/stderr都写入日志，同时也返回最后一部分文本方便界面展示。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 重要：用当前 venv 的 python，避免跑到系统 python
    proc = subprocess.run(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    out = proc.stdout or ""
    log_path.write_text(out, encoding="utf-8", errors="replace")

    # 返回码非0时，把尾巴抛出去方便你定位
    if proc.returncode != 0:
        tail = "\n".join(out.splitlines()[-80:])
        raise RuntimeError(f"Command failed (code={proc.returncode}). Log tail:\n{tail}")

    return "\n".join(out.splitlines()[-40:])  # 返回日志最后40行


def summarize_e0(csv_path: Path):
    """读取 E0_runs.csv，输出一个更适合展示的汇总表。"""
    df = pd.read_csv(csv_path)

    # 尝试猜“策略列名”
    method_col = None
    for c in ["policy", "method", "algo", "variant", "name"]:
        if c in df.columns:
            method_col = c
            break

    # 常见指标列
    wanted = [c for c in df.columns if any(k in c.lower() for k in ["delivery", "goodput", "p95"])]
    if not wanted:
        # 兜底：只取数值列
        wanted = df.select_dtypes(include="number").columns.tolist()

    if method_col:
        # 按策略求均值（你也可以改成 median）
        out = df.groupby(method_col)[wanted].mean(numeric_only=True).reset_index()
    else:
        # 没有策略列，就直接展示前几列
        out = df[wanted].copy()

    # 只展示前20行，避免UI太长
    return out.head(20)


def run_e0(seed=0, offset_ms_range=0, burst_jitter_ms=0, arrival_process="poisson"):
    """跑一次 E0（seed=0）+ 画图 + 返回：汇总表、图路径、日志尾巴。"""
    outdir = PROJECT_ROOT / "runs" / "outputs" / f"E0_seed{seed}"
    plots_dir = outdir / "plots"
    logs_dir = PROJECT_ROOT / "runs" / "logs"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) eval
    eval_log = logs_dir / f"E0_seed{seed}_eval.txt"
    eval_args = [
        sys.executable, str(E0_EVAL),
        "--seeds", str(seed),
        "--outdir", str(outdir),
        "--offset-ms-range", str(offset_ms_range),
        "--burst-jitter-ms", str(burst_jitter_ms),
        "--arrival-process", str(arrival_process),
    ]
    eval_tail = run_cmd(eval_args, log_path=eval_log, cwd=GRACE_ROOT)

    csv_path = outdir / "E0_runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    # 2) plot
    plot_log = logs_dir / f"E0_seed{seed}_plot.txt"
    plot_args = [
        sys.executable, str(E0_PLOT),
        "--runs", str(csv_path),
        "--outdir", str(plots_dir),
    ]
    plot_tail = run_cmd(plot_args, log_path=plot_log, cwd=GRACE_ROOT)

    # 3) summarize
    table = summarize_e0(csv_path)

    # 4) images
    img_delivery = plots_dir / "E0_delivery.png"
    img_goodput = plots_dir / "E0_goodput.png"
    img_p95 = plots_dir / "E0_p95.png"

    return {
        "table": table,
        "img_delivery": str(img_delivery) if img_delivery.exists() else None,
        "img_goodput": str(img_goodput) if img_goodput.exists() else None,
        "img_p95": str(img_p95) if img_p95.exists() else None,
        "log_tail": "[EVAL tail]\n" + eval_tail + "\n\n[PLOT tail]\n" + plot_tail,
        "outdir": str(outdir),
    }