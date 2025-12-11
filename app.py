import gradio as gr
import torch
from diffusers import ZImagePipeline
import os
import gc

# --- Configuration ---
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print(f"⏳ Loading Z-Image-Turbo from {MODEL_ID}...")
        try:
            # Z-Image requires bfloat16 for best performance
            pipe = ZImagePipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
            pipe.to("cuda")
            # Enable memory optimizations for 6B param model
            pipe.enable_model_cpu_offload() 
            pipe.enable_vae_tiling()
            print("✅ Model Loaded Successfully!")
            return "✅ Model Ready"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    return "✅ Model Already Loaded"

def generate(prompt, neg_prompt, width, height, steps, seed, num_images):
    global pipe
    if pipe is None:
        load_model()
    
    # Z-Image Turbo specific: Guidance usually 0.0 for distilled/turbo models
    # But we leave it adjustable just in case (standard is 0.0)
    guidance_scale = 0.0 
    
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(int(seed))
    
    print(f"🎨 Generating {num_images} image(s)...")
    
    images = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt, # Z-Image supports negative prompts
        width=width,
        height=height,
        num_inference_steps=steps, # Default is 8 for Turbo
        guidance_scale=guidance_scale, 
        num_images_per_prompt=num_images,
        generator=generator
    ).images
    
    return images, seed

# --- Gradio UI ---
custom_css = """
#run-btn {background-color: #FF5722; color: white;} 
"""

with gr.Blocks(theme='Yntec/HaleyCH_Theme_Orange', css=custom_css, title="Z-Image-Turbo WebUI") as demo:
    gr.Markdown("# ⚡ Z-Image-Turbo (6B SOTA)")
    gr.Markdown("The state-of-the-art 8-step generator by Tongyi-MAI.")
    
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Prompt", placeholder="Describe your image (English or Chinese supported)...", lines=3)
            neg_prompt = gr.Textbox(label="Negative Prompt", value="low quality, blurry, deformed", lines=1)
            
            with gr.Accordion("⚙️ Advanced Config", open=True):
                with gr.Row():
                    width = gr.Slider(512, 2048, value=1024, step=32, label="Width")
                    height = gr.Slider(512, 2048, value=1024, step=32, label="Height")
                with gr.Row():
                    steps = gr.Slider(1, 20, value=8, step=1, label="Steps (Default: 8)")
                    num_images = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
                seed = gr.Number(label="Seed (-1 = Random)", value=-1)

            btn_run = gr.Button("🚀 Generate Images", elem_id="run-btn", size="lg")
            status = gr.Textbox(label="System Status", value="System Idle", interactive=False)

        with gr.Column(scale=6):
            gallery = gr.Gallery(label="Generated Results", columns=2, height="auto")
            seed_out = gr.Number(label="Seed Used")

    # Event Triggers
    demo.load(fn=load_model, outputs=status)
    
    btn_run.click(
        fn=generate,
        inputs=[prompt, neg_prompt, width, height, steps, seed, num_images],
        outputs=[gallery, seed_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
