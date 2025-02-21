"""Gradio app to demo a 2d input image > views (as b&w contours) output

Note this will require installing TRELLI (https://github.com/rayonapp/TRELLIS) and its dependencies which can be a bit of a pain
This also requires the DexiNED repo (https://github.com/xavysp/DexiNed)
"""

import logging
import os
from pathlib import Path

os.environ["ATTN_BACKEND"] = (
    "flash-attn"  # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
)
os.environ["SPCONV_ALGO"] = "native"

import gradio as gr
import numpy as np
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

logging.basicConfig(level=logging.INFO)


def get_three_views(image: np.ndarray | None) -> list[str]: # None is because live gradio mode is True so it will be None on image clearing 
    """Make 3d model with Trellis, then extract 2d views

    Args:
        image (_type_): _description_

    Returns:
        list[str]: the paths to the 2d views saved on disk
    """
    image_pil = Image.fromarray(image)
    out_paths = ["rembg_output.png"]  # this is the input to the 3d-conversion model
    outputs = pipeline.run(
        image_pil,
        seed=18676,
        formats=["gaussian"],
    )

    image: dict = render_utils.render_snapshot(
        outputs["gaussian"][0], bg_color=(0, 0, 0), offset=(0, 0)
    )

    views = {0: "behind", 1: "left_side", 2: "front", 3: "right_side", 4: "above"}
    logging.info("Saving 2d views...")
    for ind in range(len(image["color"])):
        img = Image.fromarray(image["color"][ind])
        img.save(f"2d_image_color_{views[ind]}.png")

    image: dict = render_utils.render_snapshot(
        outputs["gaussian"][0], bg_color=(0, 0, 0), offset=(0, np.pi / 2)
    )

    img = Image.fromarray(image["color"][0])
    img.save(f"2d_image_color_{views[4]}.png")

    logging.info("Running edge-detection CNN on the images...")
    images_to_copy = [
        f"2d_image_color_{views[ind]}.png" for ind in range(5)
    ]  # behind, left, front, right, above
    # copy to folder /workspace/DexiNed/data, but before, remove the previous images
    os.system("rm /workspace/DexiNed/data/*")
    for img_path in images_to_copy:
        os.system(f"cp {img_path} /workspace/DexiNed/data/{Path(img_path).name}")

    os.chdir("/workspace/DexiNed")
    os.system("python main.py")
    out_paths.extend(
        [
            filename
            for filename in Path(
                "/workspace/DexiNed/result/BIPED2CLASSIC/fused"
            ).iterdir()
            if filename.is_file() and filename.suffix == ".png"
        ]
    )
    os.chdir("/workspace/TRELLIS")

    return out_paths


if __name__ == "__main__":
    logging.info("Loading pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large"
    )
    logging.info("Moving pipeline to GPU...")
    pipeline.cuda()
    demo = gr.Interface(
        fn=get_three_views,
        live=True,
        inputs=[
            gr.Image(),
        ],
        outputs=[
            gr.Image(label="rembg (input to the 3D converter)"),
            gr.Image(label="behind"),
            gr.Image(label="left"),
            gr.Image(label="front"),
            gr.Image(label="right"),
            gr.Image(label="above"),
        ],
        title="One image > several views",
        description="Input any image of an object",
        examples=[
            [np.array(Image.open("dog.png"))],
        ],
    )

    demo.launch(share=True, allowed_paths=["/workspace/DexiNed"]) # allow path else gradio cannot access the images outside of the current directory
