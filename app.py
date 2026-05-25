from __future__ import annotations

import gradio as gr


def status() -> str:
    return "Pose QA demo scaffold is ready. Add COCO assets or an uploaded image pipeline before deployment."


demo = gr.Interface(fn=status, inputs=None, outputs="text", title="Pose Estimation QA")


if __name__ == "__main__":
    demo.launch()
