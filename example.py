"""Example code to get the side, front and top view of a 3d representation
of a 2d input image


These images are colored like in the original image ie not in wireframe style"""

import os
import numpy as np

os.environ["ATTN_BACKEND"] = (
    "flash-attn"  # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
)
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio, logging

logging.basicConfig(level=logging.INFO)
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
logging.info("Loading pipeline...")
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Load an image
image = Image.open("rembg_output.png")  # assets/example_image/bed.jpg")

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    formats=["gaussian"],  # if the glb file is needed, add "mesh" to the list
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 3,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes


# print(len(outputs['gaussian'])) # 1
image: dict = render_utils.render_snapshot(
    outputs["gaussian"][0], bg_color=(255, 255, 255), offset=(0, 0)
)
# print(len(image['color'])) # 4
views = {0: "behind", 1: "left_side", 2: "front", 3: "right_side"}
for ind in range(len(image["color"])):
    img = Image.fromarray(
        image["color"][ind]
    )  # ind: 0 is behind, 1 is left side, 2 is front, 3 is right side
    img.save(f"2d_image_color_{views[ind]}.png")

image: dict = render_utils.render_snapshot(
    outputs["gaussian"][0], bg_color=(255, 255, 255), offset=(0, np.pi / 2)
)
print(len(image["color"]))  # 4
views = {0: "above"}
img = Image.fromarray(
    image["color"][0]
)  # ind: 0 is behind, 1 is left side, 2 is front, 3 is right side
img.save(f"2d_image_color_{views[0]}.png")


exit()
# GLB files can be extracted from the outputs
logging.info("Exporting GLB...")
glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    # Optional parameters
    simplify=0.95,  # Ratio of triangles to remove in the simplification process
    texture_size=1024,  # Size of the texture used for the GLB
)
glb.export("sample_.glb")

# Save Gaussians as PLY files
# outputs['gaussian'][0].save_ply("sample.ply")
