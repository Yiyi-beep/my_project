import gradio as gr

from src.grace_runner import run_e0

def ui_run_e0(seed, offset_ms_range, burst_jitter_ms, arrival_process):
    res = run_e0(
        seed=int(seed),
        offset_ms_range=int(offset_ms_range),
        burst_jitter_ms=int(burst_jitter_ms),
        arrival_process=arrival_process,
    )
    # gradio Dataframe 支持 pandas dataframe
    return (
        res["table"],
        res["img_delivery"],
        res["img_goodput"],
        res["img_p95"],
        res["log_tail"],
        res["outdir"],
    )

with gr.Blocks(title="Grace Reproduction Demo") as demo:
    gr.Markdown("# Grace 复现 Demo（E0）\n点一下按钮：运行 E0 → 生成 CSV → 出三张图 → 展示汇总表")

    with gr.Row():
        seed = gr.Number(value=0, label="seed", precision=0)
        arrival = gr.Dropdown(["poisson", "const"], value="poisson", label="arrival_process")
    with gr.Row():
        offset = gr.Number(value=0, label="offset_ms_range（0=关闭）", precision=0)
        jitter = gr.Number(value=0, label="burst_jitter_ms（0=关闭）", precision=0)

    run_btn = gr.Button("运行 E0（复现实验+出图）", variant="primary")

    table = gr.Dataframe(label="E0 汇总（前20行）", wrap=True)
    with gr.Row():
        img1 = gr.Image(label="E0_delivery.png")
        img2 = gr.Image(label="E0_goodput.png")
        img3 = gr.Image(label="E0_p95.png")
    log = gr.Textbox(label="日志尾巴（方便排错）", lines=16)
    outdir = gr.Textbox(label="输出目录", lines=1)

    run_btn.click(
        fn=ui_run_e0,
        inputs=[seed, offset, jitter, arrival],
        outputs=[table, img1, img2, img3, log, outdir],
    )

if __name__ == "__main__":
    demo.launch()

