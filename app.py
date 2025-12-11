import gradio as gr
import torch
# WE USE THE STANDARD PIPELINE CLASS - IT IS BULLETPROOF
from diffusers import DiffusionPipeline
import os
import gc

# --- Configuration ---
# ⚠️ CRITICAL: Ensure this ID is correct. 
# If "Tongyi-MAI/Z-Image-Turbo" fails, try "black-forest-labs/FLUX.1-schnell" 
# or the specific ID provided by the Z-Image release.
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo" 

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print(f"⏳ Loading Model: {MODEL_ID}...")
        try:
            # 1. Use the Universal 'DiffusionPipeline' class
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="balanced" # Dual GPU
            )
            
            # 2. Optimizations
            # We check if the model supports these before calling them
            if hasattr(pipe, 'enable_vae_tiling'):
                pipe.enable_vae_tiling()
            
            print("✅ Model Loaded Successfully!")
            return "✅ Ready"
        except OSError:
            return "❌ Error: Model ID not found on HuggingFace. Please check the spelling."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    return "✅ Ready"

def generate(prompt, width, height, steps, seed, num_images):
    global pipe
    if pipe is None:
        status = load_model()
        if "Error" in status:
            return None, status
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()
    
    generator = torch.Generator("cpu").manual_seed(int(seed))
    
    print(f"🎨 Generating {num_images} image(s)...")
    
    try:
        # Standard Inference Call
        images = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0, 
            num_images_per_prompt=num_images,
            generator=generator
        ).images
        return images, seed
    except Exception as e:
        return None, f"Runtime Error: {e}"

# --- Gradio UI ---
custom_css = "body {background-color: #0b0f19;}"

with gr.Blocks(theme='Yntec/HaleyCH_Theme_Orange', css=custom_css, title="Z-Image-Turbo") as demo:
    gr.Markdown("# ⚡ Z-Image-Turbo")
    
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Prompt", placeholder="Describe image...", lines=3)
            with gr.Accordion("⚙️ Settings", open=True):
                width = gr.Slider(512, 1280, value=1024, step=32, label="Width")
                height = gr.Slider(512, 1280, value=1024, step=32, label="Height")
                steps = gr.Slider(1, 20, value=8, step=1, label="Steps")
                num_images = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
                seed = gr.Number(label="Seed", value=-1)

            btn_run = gr.Button("🚀 Generate", variant="primary", size="lg")
            status = gr.Textbox(label="System Status", value="Idle", interactive=False)

        with gr.Column(scale=6):
            gallery = gr.Gallery(label="Output", columns=2, height="auto")
            seed_out = gr.Number(label="Seed Used")

    btn_run.click(fn=generate, inputs=[prompt, width, height, steps, seed, num_images], outputs=[gallery, seed_out])

if __name__ == "__main__":
    demo.launch(share=True)
