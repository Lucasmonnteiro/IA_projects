# -*- coding: utf-8 -*-

from diffusers import StableDiffusionXLPipeline

# Carrega o modelo
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
)

# Texto que descreve a imagem
prompt = "A game girl in sideral space playing Minecraft."

# Tamanho da imagem
width = 1024
height = 1024

# Gera a imagem
image = pipeline(prompt, width=width, height=height).images[0]

# Salva a imagem
image.save("generated_image.png")

# Mostra a imagem
image.show()
