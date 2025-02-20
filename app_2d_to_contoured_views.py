"""Gradio app to demo a 2d input image > views (as contours) output

Note this will require installing TRELLI and its dependencies which can be a bit of a pain"""

import logging
import os

os.environ["ATTN_BACKEND"] = (
    "flash-attn"  # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
)
os.environ["SPCONV_ALGO"] = "native"

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

logging.basicConfig(level=logging.INFO)


def get_three_views(image: np.ndarray) -> list[str]:
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

    views = {0: "behind", 1: "left_side", 2: "front", 3: "right_side"}
    logging.info("Saving 2d views...")
    for ind in range(len(image["color"])):
        img = Image.fromarray(
            image["color"][ind]
        )  # ind: 0 is behind, 1 is left side, 2 is front, 3 is right side
        img.save(f"2d_image_color_{views[ind]}.png")
        contour_image = img_to_contours(f"2d_image_color_{views[ind]}.png")
        contour_image.save(f"contour_2d_image_color_{views[ind]}.png")
        out_paths.append(f"contour_2d_image_color_{views[ind]}.png")

    image: dict = render_utils.render_snapshot(
        outputs["gaussian"][0], bg_color=(0, 0, 0), offset=(0, np.pi / 2)
    )

    views = {0: "above"}
    img = Image.fromarray(
        image["color"][0]
    )  # ind: 0 is behind, 1 is left side, 2 is front, 3 is right side
    img.save(f"2d_image_color_{views[0]}.png")
    contour_image = img_to_contours(f"2d_image_color_{views[0]}.png")
    contour_image.save(f"contour_2d_image_color_{views[0]}.png")
    out_paths.append(f"contour_2d_image_color_{views[0]}.png")

    return out_paths


def img_to_contours(img_path: str) -> Image.Image:
    # Load the image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Gaussian filter
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # find good canny thresholds
    # logging.info("Finding good Canny thresholds...")
    # v = np.median(image)
    # sigma = 0.75  # I don't want all the little details
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(image, 0, 255)  # lower, upper)
    return Image.fromarray(255-edges)
    # Find contours
    #logging.info("Finding contours...")
    #contours, *_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # this will locate the outside contour only (too minimalist for our use case)

    #contour_image = np.zeros_like(image)
    #cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

    #contour_image = 255 - contour_image

    #return Image.fromarray(contour_image)


if __name__ == "__main__":
    logging.info("Loading pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large"
    )
    logging.info("Moving pipeline to GPU...")
    pipeline.cuda()
    demo = gr.Interface(
        fn=get_three_views,
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

    demo.launch(share=True)
