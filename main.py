from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline as hf_pipeline
import numpy as np
import tempfile
import soundfile as sf

# CONFIGURACIÓN DE MODELOS

device = "cuda" if torch.cuda.is_available() else "cpu"

# Text-to-Image: Arcane-Diffusion
txt2img_pipe = StableDiffusionPipeline.from_pretrained("nitrosocke/Arcane-Diffusion").to(device)

# Image-to-Image: Stable Diffusion 1.5
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Image-to-Text: BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Text-to-Speech: Bark (Google model via Hugging Face)
tts_pipe = hf_pipeline("text-to-speech", model="suno/bark-small")

# FUNCIONES DE GENERACIÓN

def generate_initial(prompt):
    if not prompt.strip():
        return None
    image = txt2img_pipe(prompt, num_inference_steps=20, height=512, width=512).images[0]
    return image

def regenerate_image(prompt):
    if not prompt.strip():
        return None
    image = txt2img_pipe(prompt, num_inference_steps=25, height=512, width=512).images[0]
    return image

def generate_skate_image(init_image):
    if init_image is None:
        return None
    if not isinstance(init_image, Image.Image):
        init_image = Image.fromarray(init_image)
    init_image = init_image.convert("RGB").resize((512, 512))
    prompt = "Pon esta imagen en la parte inferior de un skate"
    result = img2img_pipe(prompt=prompt, image=init_image, num_inference_steps=30, strength=0.7)
    return result.images[0]

def describe_image(image):
    """
    Genera una descripción en texto de la imagen usando BLIP.
    """
    if image is None:
        return "No hay imagen para describir."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    output_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption

def text_to_speech(text):
    """
    Convierte el texto a audio usando Bark TTS (Google model).
    Retorna directamente el audio compatible con Gradio.
    """
    if not text.strip():
        return None

    result = tts_pipe(text)
    audio_array = result["audio"]
    sampling_rate = result["sampling_rate"]

    # Normaliza y convierte a float32
    audio_array = np.array(audio_array, dtype=np.float32)

    # Gradio acepta directamente (audio_array, sampling_rate)
    return (audio_array, sampling_rate)

# INTERFAZ GRADIO

with gr.Blocks() as demo:
    gr.Markdown("## Proyecto 'Skate'")

    prompt_input = gr.Textbox(label="Describe tu diseño:", placeholder="Ejemplo: un dragón de fuego estilo Arcane")

    with gr.Row():
        generate_btn = gr.Button("Generar imagen")
        regenerate_btn = gr.Button("Regenerar")

    first_image = gr.Image(label="Imagen generada")

    send_to_skate_btn = gr.Button("Enviar al modelo Image→Image")
    skate_image = gr.Image(label="Imagen aplicada al skate")

    with gr.Row():
        describe_btn = gr.Button("Describir imagen del skate")
        tts_btn = gr.Button("enerar voz de la descripción")

    description_output = gr.Textbox(label="Descripción generada", interactive=False)
    audio_output = gr.Audio(label="Voz generada (TTS)")

    # Conexiones
    generate_btn.click(fn=generate_initial, inputs=prompt_input, outputs=first_image)
    regenerate_btn.click(fn=regenerate_image, inputs=prompt_input, outputs=first_image)
    send_to_skate_btn.click(fn=generate_skate_image, inputs=first_image, outputs=skate_image)
    describe_btn.click(fn=describe_image, inputs=skate_image, outputs=description_output)
    tts_btn.click(fn=text_to_speech, inputs=description_output, outputs=audio_output)

demo.launch()
