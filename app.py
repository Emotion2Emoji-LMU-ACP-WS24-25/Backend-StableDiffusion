from flask import Flask, request, jsonify
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import torch

app = Flask(__name__)

# Stable Diffusion Pipeline initialisieren

pipe = StableDiffusionXLPipeline.from_pretrained("/app/models/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, local_files_only=True).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
#pipe.enable_xformers_memory_efficient_attention()

pipe.load_lora_weights('/app/models/ios_emoji_xl_v2_lora.safetensors', lora_scale=0.6)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Bild generieren
    image = pipe(prompt, negative_prompt="blurry").images[0]
    image_path = "generated_image.png"
    image.save(image_path)

    # Antwort senden
    return jsonify({"image_path": image_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)