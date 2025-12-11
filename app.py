import gradio as gr
import torch
from diffusers import AutoPipelineForTextToImage
import os
import gc

# --- Configuration ---
# Using the ID you likely intended. 
# If this is a specific ModelScope model, we might need 'snapshot_download',
# but let's try the HF ID first if it exists, or fallback to the known working turbo method.
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo" 

# NOTE: If "Tongyi-MAI/Z-Image-Turbo" is not the exact HF ID, 
# you might be referring to "alibaba-pai/pai-diffusion-general-large-zh" or similar.
# However, assuming the files exist at that ID or a local path:

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print(f"⏳ Loading Z-Image-Turbo...")
        try:
            # We use AutoPipeline. It is smart enough to handle SDXL-Turbo, Flux, etc.
            # We REMOVED the fake 'ProxyUNet2DConditionModel' import.
            pipe = AutoPipelineForTextToImage.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,  # Crucial for new/custom models
                device_map="balanced"
            )
            
            # Optimizations
            if hasattr(pipe, 'enable_vae_tiling'):
                pipe.enable_vae_tiling()
            
            print("✅ Model Loaded!")
            return "✅ Model Ready"
        except Exception as e:
            # Fallback for debugging if the ID is wrong
            return f"❌ Load Error: {str(e)}"
    return "✅ Model Already Loaded"

def generate(prompt, width, height, steps, seed, num_images):
    global pipe
    if pipe is None:
        load_model()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()
    
    generator = torch.Generator("cpu").manual_seed(int(seed))
    
    print(f"🎨 Generating {num_images} image(s)...")
    
    try:
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
        return None, f"Error: {e}"

# --- Gradio UI ---
custom_css = """
#run-btn {background-color: #D32F2F; color: white;} 
"""

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

            btn_run = gr.Button("🚀 Generate", elem_id="run-btn", size="lg")
            status = gr.Textbox(label="Status", value="Idle", interactive=False)

        with gr.Column(scale=6):
            gallery = gr.Gallery(label="Results", columns=2, height="auto")
            seed_out = gr.Number(label="Seed")

    demo.load(fn=load_model, outputs=status)
    btn_run.click(fn=generate, inputs=[prompt, width, height, steps, seed, num_images], outputs=[gallery, seed_out])

if __name__ == "__main__":
    demo.launch(share=True)
