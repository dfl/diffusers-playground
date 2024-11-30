This is a Gradio app to explore different ways of using Stable Diffusion image generation with the Diffusers library.
The most developed tools are Gradio apps to use [Trajectory Consistent Distillation](https://mhh0318.github.io/tcd/)
I have created one for SDXL and for SD 2.1
They support img2img and hires-fix!
The image metadata is also stored in the image files, and can be recovered by dropping the image into the UI where it says "Drop Image Here" (make sure you see the red-dotted line appear, or it will just load the image in the browser)

To get started:

```bash
# make a new environment
python3 -m venv myenv
pip install -r requirements.txt

# run the app
python tcd_gradio_appXL.py # for SDXL
python tcd_gradio_app2.py # for SD 2.1
```
