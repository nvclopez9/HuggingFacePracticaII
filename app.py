import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import pipeline
from PIL import Image
import imageio
import numpy as np

DEVICE = "cpu"  # Forzar CPU

#########################################
# 1) TEXT → IMAGE (sd-turbo)
#########################################
try:
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo"
    ).to(DEVICE)
except Exception as e:
    sd_pipe = None
    print(f"ERROR cargando SD-Turbo: {e}")

#########################################
# 2) IMAGE → TEXT (Captioning)
#########################################
try:
    blip_pipe = pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning",
        device=0 if DEVICE == "cuda" else -1
    )
except Exception as e:
    blip_pipe = None
    print(f"ERROR cargando BLIP: {e}")

#########################################
# 3) IMAGE → 3D (Zero123-XL)
#########################################
try:
    zero123_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/zero123-xl"
    ).to(DEVICE)
except Exception as e:
    zero123_pipe = None
    print(f"ERROR cargando Zero123-XL: {e}")

#########################################
# FUNCIONES
#########################################

def text_to_image(prompt):
    if sd_pipe is None:
        return None, "[ERROR] No se pudo cargar SD-Turbo"
    try:
        img = sd_pipe(prompt).images[0]
        return img, "[OK] Imagen generada"
    except Exception as e:
        return None, f"[ERROR] {e}"

def image_to_image(img, prompt):
    if sd_pipe is None:
        return None, "[ERROR] No se pudo cargar SD-Turbo"
    try:
        img = sd_pipe(prompt if prompt else "enhance", image=img).images[0]
        return img, "[OK] Imagen refinada"
    except Exception as e:
        return None, f"[ERROR] {e}"

def image_to_3d_gif(img):
    if zero123_pipe is None:
        return None, "[ERROR] No se pudo cargar Zero123-XL"
    try:
        # Generamos 6 vistas
        outputs = zero123_pipe(
            img,
            num_inference_steps=20,
            num_frames=6
        ).images

        frames = [np.array(frame) for frame in outputs]
        gif_path = "3d_preview.gif"
        imageio.mimsave(gif_path, frames, fps=4)
        return gif_path, "[OK] GIF 3D generado"
    except Exception as e:
        return None, f"[ERROR] {e}"

def image_to_text(img):
    if blip_pipe is None:
        return None, "[ERROR] No se pudo cargar captioning"
    try:
        caption = blip_pipe(img)[0]["generated_text"]
        return caption, "[OK] Texto generado"
    except Exception as e:
        return None, f"[ERROR] {e}"

#########################################
# INTERFAZ
#########################################

with gr.Blocks(title="Proyecto Final con Zero123-XL (CPU)") as demo:
    gr.Markdown("# Proyecto con Modelos Ligeros + 3D REAL (Zero123-XL)")

    with gr.Tab("Text → Image (SD-Turbo)"):
        prompt1 = gr.Textbox(label="Prompt")
        out_img1 = gr.Image(label="Resultado")
        log1 = gr.Textbox(label="Log", interactive=False)
        btn1 = gr.Button("Generar")
        btn1.click(fn=text_to_image, inputs=prompt1, outputs=[out_img1, log1])

    with gr.Tab("Image → Image (Refinado con SD-Turbo)"):
        img2 = gr.Image(label="Imagen original")
        prompt2 = gr.Textbox(label="Prompt de edición", placeholder="Opcional")
        out_img2 = gr.Image(label="Resultado")
        log2 = gr.Textbox(label="Log", interactive=False)
        btn2 = gr.Button("Refinar")
        btn2.click(fn=image_to_image, inputs=[img2, prompt2], outputs=[out_img2, log2])

    with gr.Tab("Image → 3D (GIF Zero123-XL)"):
        img3 = gr.Image(label="Imagen original")
        out_gif = gr.Image(label="GIF Preview", type="filepath")
        log3 = gr.Textbox(label="Log", interactive=False)
        btn3 = gr.Button("Generar GIF 3D")
        btn3.click(fn=image_to_3d_gif, inputs=img3, outputs=[out_gif, log3])

    with gr.Tab("Image → Text (Captioning)"):
        img4 = gr.Image(label="Imagen")
        out_text = gr.Textbox(label="Descripción generada")
        log4 = gr.Textbox(label="Log", interactive=False)
        btn4 = gr.Button("Generar descripción")
        btn4.click(fn=image_to_text, inputs=img4, outputs=[out_text, log4])

demo.launch()
