import os
import sys
import logging
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
from rembg import remove # Background removal

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NeuralSplat-Backend")

# --- CONFIGURATION ---
PORT = 8080
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Ensure Trellis is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "TRELLIS"))

# --- GLOBAL MODEL STORE ---
model_store = {
    "pipeline": None
}

# --- MODEL LIFECYCLE MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the heavy AI model on startup, release on shutdown.
    """
    logger.info("Initializing TRELLIS Pipeline...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA NOT DETECTED. The model will likely fail or run extremely slowly.")
    
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
        
        # Load Model (HuggingFace)
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
        
        model_store["pipeline"] = pipeline
        logger.info("✅ TRELLIS Pipeline loaded successfully into VRAM.")
    except ImportError as e:
        logger.critical(f"❌ Failed to import TRELLIS modules. Did you run setup_env.sh? Error: {e}")
    except Exception as e:
        logger.critical(f"❌ Failed to load Model Weights. Error: {e}")
        
    yield
    
    # Cleanup (Optional)
    logger.info("Shutting down...")
    if model_store["pipeline"]:
        del model_store["pipeline"]
        torch.cuda.empty_cache()

# --- APP INITIALIZATION ---
app = FastAPI(title="Neural Splat API", lifespan=lifespan)

# CORS (Allow Vercel frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_path: str) -> Image.Image:
    """
    Opens image and removes background using rembg.
    TRELLIS requires RGBA with transparent background for best results.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
        # Remove background
        img_no_bg = remove(img)
        return img_no_bg
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise ValueError("Image preprocessing failed")

@app.get("/")
def health_check():
    status = "ready" if model_store["pipeline"] is not None else "loading_or_failed"
    gpu_status = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "status": status, 
        "gpu": gpu_status,
        "vram_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "N/A"
    }

@app.post("/generate")
async def generate_splat(file: UploadFile = File(...)):
    """
    Endpoint: Image -> PLY File
    """
    if model_store["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Check server logs.")

    # 1. Validate File
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # 2. Save Upload
    file_id = f"{file.filename}_{os.urandom(4).hex()}"
    input_path = os.path.join(UPLOAD_DIR, file_id)
    
    try:
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())
        
        logger.info(f"Received image: {file.filename}")

        # 3. Preprocess (Remove Background)
        logger.info("Removing background...")
        processed_image = preprocess_image(input_path)

        # 4. Run Inference
        logger.info("Running TRELLIS Inference (this takes ~15-30s)...")
        pipeline = model_store["pipeline"]
        
        # Run generation
        outputs = pipeline.run(processed_image, seed=1)
        
        # 5. Extract and Save PLY
        output_filename = f"splat_{file_id}.ply"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Check available keys in output to find Gaussian data
        # Usually outputs['gaussian'][0] is the object
        if 'gaussian' in outputs and len(outputs['gaussian']) > 0:
            outputs['gaussian'][0].save_ply(output_path)
        else:
            raise ValueError("Model generated no Gaussian data.")

        logger.info(f"✅ Generation complete: {output_path}")
        
        # 6. Return File
        return FileResponse(
            output_path, 
            media_type="application/octet-stream", 
            filename="model.ply"
        )

    except Exception as e:
        logger.error(f"Generation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
    finally:
        # Cleanup input file to save space
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)
