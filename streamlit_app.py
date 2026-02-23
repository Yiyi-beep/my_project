import os
import time
import sys
import zipfile
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(r"D:\SparkCup\my_project")
GRACE_ROOT = Path(r"D:\SparkCup\my_project\repos\Grace_Reimplementation\Grace_Reimplementation2 2.3")

E0_EVAL = GRACE_ROOT / "grace_v2" / "eval_E0_static_burst.py"
E0_PLOT = GRACE_ROOT / "grace_v2" / "plot_E0_grouped.py"

DEFAULT_OUTDIR = PROJECT_ROOT / "runs" / "outputs" / "E0_seed0"
DEFAULT_PLOTDIR = DEFAULT_OUTDIR / "plots"
LOG_DIR = PROJECT_ROOT / "runs" / "logs"


def tail_text(path: Path, n=60) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def summarize(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    metrics = [
        "delivery_ratio", "goodput_mbps", "p95_latency_ms",
        "delivery_burst", "goodput_burst_mbps", "p95_latency_burst"
    ]
    metrics = [m for m in metrics if m in df.columns]
    if "policy" in df.columns:
        out = df.groupby("policy")[metrics].mean(numeric_only=True).reset_index()
    else:
        out = df[metrics].copy()
    return out


def best_cards(summary: pd.DataFrame):
    # 选出“最优策略”用于卡片展示
    def pick_best(col, higher_is_better=True):
        if col not in summary.columns:
            return None, None
        s = summary[["policy", col]].dropna()
        if s.empty:
            return None, None
        idx = s[col].idxmax() if higher_is_better else s[col].idxmin()
        return str(s.loc[idx, "policy"]), float(s.loc[idx, col])

    b1 = pick_best("delivery_ratio", True)
    b2 = pick_best("goodput_mbps", True)
    b3 = pick_best("p95_latency_ms", False)

    return b1, b2, b3


def run_with_live_log(cmd_list, cwd: Path, log_path: Path, placeholder):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # 直接把 stdout/stderr 写进文件，前端循环读 tail
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(
            cmd_list,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    while p.poll() is None:
        placeholder.code(tail_text(log_path, 60) or "Running...")
        time.sleep(1)
    # 最后再刷新一次
    placeholder.code(tail_text(log_path, 80) or "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd_list}\n\nLog tail:\n{tail_text(log_path, 120)}")


def zip_dir(dir_path: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in dir_path.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(dir_path)))


st.set_page_config(page_title="Grace E0 Dashboard", layout="wide")
st.title("Grace 复现实验仪表盘（E0）")

with st.sidebar:
    st.header("控制面板")
    seed = st.number_input("seed", value=0, step=1)
    arrival = st.selectbox("arrival_process", ["poisson", "const"], index=0)
    offset = st.number_input("offset_ms_range（0=关闭）", value=0, step=1)
    jitter = st.number_input("burst_jitter_ms（0=关闭）", value=0, step=1)

    outdir = PROJECT_ROOT / "runs" / "outputs" / f"E0_seed{int(seed)}"
    plotdir = outdir / "plots"
    st.caption(f"输出目录：{outdir}")

    load_btn = st.button("加载已有结果（秒开）", use_container_width=True)
    run_btn = st.button("重新跑 E0（慢）", type="primary", use_container_width=True)

    st.divider()
    st.caption("下载本次输出目录（csv+图+日志）")
    dl_btn = st.button("打包下载（zip）", use_container_width=True)


# 主区域
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("结果表（按 policy 汇总）")
    table_slot = st.empty()

    st.subheader("关键指标卡片")
    c1, c2, c3 = st.columns(3)
    card1 = c1.empty()
    card2 = c2.empty()
    card3 = c3.empty()

with right:
    st.subheader("图表")
    img_slot1 = st.empty()
    img_slot2 = st.empty()
    img_slot3 = st.empty()

    st.subheader("运行日志（实时尾巴）")
    log_slot = st.empty()
    st.caption("提示：E0 是仿真评测，通常更吃 CPU，不一定显著用到 GPU。")


def render(outdir_path: Path):
    csv_path = outdir_path / "E0_runs.csv"
    if not csv_path.exists():
        table_slot.warning(f"找不到 {csv_path}，请先运行一次 E0 或点击“加载已有结果”。")
        return

    summary = summarize(csv_path)
    table_slot.dataframe(summary, use_container_width=True)

    b1, b2, b3 = best_cards(summary)
    if b1[0]:
        card1.metric("最高 delivery_ratio", f"{b1[1]:.4f}", help=f"policy = {b1[0]}")
    if b2[0]:
        card2.metric("最高 goodput_mbps", f"{b2[1]:.2f}", help=f"policy = {b2[0]}")
    if b3[0]:
        card3.metric("最低 p95_latency_ms", f"{b3[1]:.2f}", help=f"policy = {b3[0]}")

    # 图
    p1 = outdir_path / "plots" / "E0_delivery.png"
    p2 = outdir_path / "plots" / "E0_goodput.png"
    p3 = outdir_path / "plots" / "E0_p95.png"
    if p1.exists(): img_slot1.image(str(p1), caption="E0_delivery.png", use_container_width=True)
    if p2.exists(): img_slot2.image(str(p2), caption="E0_goodput.png", use_container_width=True)
    if p3.exists(): img_slot3.image(str(p3), caption="E0_p95.png", use_container_width=True)


# 1) 秒开：加载已有
if load_btn:
    render(outdir)

# 2) 慢跑：重新跑
if run_btn:
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    eval_log = LOG_DIR / f"E0_seed{int(seed)}_eval_live.txt"
    plot_log = LOG_DIR / f"E0_seed{int(seed)}_plot_live.txt"

    st.toast("开始运行 E0（会比较慢，日志会实时更新）", icon="⏳")

    # eval
    st.info("Step 1/2：评测（eval）")
    eval_cmd = [
        sys.executable, str(E0_EVAL),
        "--seeds", str(int(seed)),
        "--outdir", str(outdir),
        "--offset-ms-range", str(int(offset)),
        "--burst-jitter-ms", str(int(jitter)),
        "--arrival-process", str(arrival),
    ]
    run_with_live_log(eval_cmd, cwd=GRACE_ROOT, log_path=eval_log, placeholder=log_slot)

    # plot
    st.info("Step 2/2：出图（plot）")
    csv_path = outdir / "E0_runs.csv"
    plot_cmd = [
        sys.executable, str(E0_PLOT),
        "--runs", str(csv_path),
        "--outdir", str(plotdir),
    ]
    run_with_live_log(plot_cmd, cwd=GRACE_ROOT, log_path=plot_log, placeholder=log_slot)

    st.success("完成！已生成 csv + 三张图。")
    render(outdir)


# 3) 打包下载
if dl_btn:
    if not outdir.exists():
        st.warning("还没有输出目录可以打包。")
    else:
        zip_path = PROJECT_ROOT / "runs" / "outputs" / f"E0_seed{int(seed)}.zip"
        zip_dir(outdir, zip_path)
        with open(zip_path, "rb") as f:
            st.download_button(
                "点击下载 zip",
                data=f,
                file_name=zip_path.name,
                mime="application/zip",
                use_container_width=True
            )

# 页面初始默认展示：如果已有默认结果就展示（更像产品）
if DEFAULT_OUTDIR.exists() and (DEFAULT_OUTDIR / "E0_runs.csv").exists():
    st.caption("已检测到默认结果 E0_seed0，已自动展示（你也可以在左侧重新跑）。")
    render(DEFAULT_OUTDIR)
else:
    st.caption("未检测到默认结果。请先点击左侧“重新跑 E0（慢）”。")