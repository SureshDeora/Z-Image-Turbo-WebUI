import gradio as gr
import torch
from diffusers import ZImagePipeline, ProxyUNet2DConditionModel
from diffusers.models import AutoencoderKL
from transformers import AutoModel
import os
import gc
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# --- Configuration ---
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
pipe = None

def load_model():
    global pipe
    if pipe is None:
        print(f"⏳ Loading Z-Image-Turbo across BOTH GPUs...")
        try:
            # 1. Load the Transformer (The Heavy Part) 
            # We use device_map="balanced" to split it across GPU 0 and GPU 1 automatically
            from diffusers import Transformer2DModel
            
            print("   ...Splitting Transformer across GPUs...")
            transformer = Transformer2DModel.from_pretrained(
                MODEL_ID,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                device_map="balanced" # <--- THE MAGIC KEY: Uses both GPUs
            )
            
            # 2. Load the rest of the pipeline
            # We pass the pre-loaded (split) transformer into the pipeline
            pipe = ZImagePipeline.from_pretrained(
                MODEL_ID,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
            
            # 3. Move the lighter parts (Text Encoder, VAE) to GPU 0
            # (The transformer is already spread out, so we don't move the whole pipe)
            pipe.vae.to("cuda:0")
            pipe.text_encoder.to("cuda:0")
            
            # 4. Enable tiling to save memory during the final decode step
            pipe.enable_vae_tiling()
            
            print("✅ Model Loaded! (Running on Dual GPUs)")
            return "✅ Dual-GPU Active"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    return "✅ Model Already Loaded"

def generate(prompt, neg_prompt, width, height, steps, seed, num_images):
    global pipe
    if pipe is None:
        load_model()
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()
    
    generator = torch.Generator("cpu").manual_seed(int(seed))
    
    print(f"🎨 Generating {num_images} image(s)...")
    
    try:
        images = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0, 
            num_images_per_prompt=num_images,
            generator=generator
        ).images
        return images, seed
    except RuntimeError as e:
        return None, f"Error: {e}"

# --- Gradio UI ---
custom_css = """
#run-btn {background-color: #6200EA; color: white;} 
"""

with gr.Blocks(theme='Yntec/HaleyCH_Theme_Orange', css=custom_css, title="Z-Image Dual-GPU") as demo:
    gr.Markdown("# ⚡ Z-Image-Turbo (Dual-GPU Power)")
    gr.Markdown("Using **2x T4 GPUs** for maximum speed and stability.")
    
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Prompt", placeholder="Describe your image...", lines=3)
            neg_prompt = gr.Textbox(label="Negative Prompt", value="low quality, blurry", lines=1)
            
            with gr.Accordion("⚙️ Config", open=True):
                width = gr.Slider(512, 1536, value=1024, step=32, label="Width")
                height = gr.Slider(512, 1536, value=1024, step=32, label="Height")
                steps = gr.Slider(1, 20, value=8, step=1, label="Steps")
                num_images = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
                seed = gr.Number(label="Seed (-1 = Random)", value=-1)

            btn_run = gr.Button("🚀 Generate on 2 GPUs", elem_id="run-btn", size="lg")
            status = gr.Textbox(label="Status", value="Idle", interactive=False)

        with gr.Column(scale=6):
            gallery = gr.Gallery(label="Generated Results", columns=2, height="auto")
            seed_out = gr.Number(label="Seed Used")

    demo.load(fn=load_model, outputs=status)
    btn_run.click(fn=generate, inputs=[prompt, neg_prompt, width, height, steps, seed, num_images], outputs=[gallery, seed_out])

if __name__ == "__main__":
    demo.launch(share=True)
