import os
import tempfile
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from moviepy.editor import VideoFileClip, AudioFileClip
import cv2

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.pose_encoder import PoseEncoder
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.utils.util import save_videos_grid
from src.utils.dwpose_util import draw_pose_select_v2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "echomimic-video-gen-models")  # Get from env or use default
PRETRAINED_WEIGHTS_DIR = Path("./pretrained_weights")
CONFIG_PATH = "./configs/prompts/infer_acc.yaml"

# Model files to check and download if missing
REQUIRED_MODEL_FILES = {
    "sd-image-variations-diffusers": "directory",
    "sd-vae-ft-mse": "directory",
    "denoising_unet_acc.pth": "file",
    "reference_unet.pth": "file",
    "pose_encoder.pth": "file",
    "motion_module_acc.pth": "file",
    "audio_processor/tiny.pt": "file"
}


def download_from_s3_if_needed():
    """Download models from S3 only if they're missing locally"""
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        logger.warning("boto3 not installed. Skipping S3 download. Install with: pip install boto3")
        return
    
    models_status = check_models_available()
    missing_models = [k for k, v in models_status.items() if not v]
    
    if not missing_models:
        logger.info("All models are available locally. Skipping S3 download.")
        return
    
    logger.info(f"Missing models: {missing_models}")
    logger.info(f"Attempting to download from S3 bucket: {S3_BUCKET}")
    
    try:
        s3_client = boto3.client('s3')
        
        for model_name in missing_models:
            local_path = PRETRAINED_WEIGHTS_DIR / model_name
            model_type = REQUIRED_MODEL_FILES[model_name]
            
            if model_type == "directory":
                # Download entire directory
                logger.info(f"Downloading directory: {model_name}...")
                s3_prefix = f"pretrained_weights/{model_name}/"
                
                try:
                    paginator = s3_client.get_paginator('list_objects_v2')
                    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
                        if 'Contents' not in page:
                            logger.warning(f"No files found in S3 for {model_name}")
                            continue
                        
                        for obj in page['Contents']:
                            s3_key = obj['Key']
                            if s3_key.endswith('/'):
                                continue
                            
                            rel_path = s3_key.replace(s3_prefix, '', 1)
                            file_local_path = local_path / rel_path
                            
                            file_local_path.parent.mkdir(parents=True, exist_ok=True)
                            s3_client.download_file(S3_BUCKET, s3_key, str(file_local_path))
                            logger.info(f"  Downloaded: {rel_path}")
                    
                    logger.info(f"✓ Downloaded directory: {model_name}")
                except ClientError as e:
                    logger.error(f"Failed to download {model_name} from S3: {e}")
            
            else:
                # Download single file
                logger.info(f"Downloading file: {model_name}...")
                s3_key = f"pretrained_weights/{model_name}"
                
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    s3_client.download_file(S3_BUCKET, s3_key, str(local_path))
                    logger.info(f"✓ Downloaded: {model_name}")
                except ClientError as e:
                    logger.error(f"Failed to download {model_name} from S3: {e}")
        
        # Check again after download
        models_status = check_models_available()
        still_missing = [k for k, v in models_status.items() if not v]
        
        if still_missing:
            logger.warning(f"Some models are still missing after S3 download: {still_missing}")
        else:
            logger.info("✓ All models downloaded successfully from S3!")
    
    except NoCredentialsError:
        logger.error("AWS credentials not found. Cannot download from S3.")
        logger.info("Please configure AWS credentials or ensure all models are available locally.")
    except Exception as e:
        logger.error(f"Error during S3 download: {e}")


# Global pipeline
pipeline = None
audio_processor = None
config = None
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.float16 if device == "cuda" else torch.float32


def check_models_available() -> dict:
    """Check which models are available locally"""
    status = {}
    for model_name, model_type in REQUIRED_MODEL_FILES.items():
        model_path = PRETRAINED_WEIGHTS_DIR / model_name
        if model_type == "directory":
            status[model_name] = model_path.exists() and model_path.is_dir()
        else:
            status[model_name] = model_path.exists() and model_path.is_file()
    
    return status


def load_models():
    """Load all required models into memory"""
    global pipeline, audio_processor, config
    
    logger.info("Loading models into memory...")
    
    # Check if all models are available
    models_status = check_models_available()
    missing_models = [k for k, v in models_status.items() if not v]
    
    if missing_models:
        raise RuntimeError(f"Missing models: {missing_models}. Please ensure all models are in {PRETRAINED_WEIGHTS_DIR}")
    
    # Load config
    config = OmegaConf.load(CONFIG_PATH)
    infer_config = OmegaConf.load(config.inference_config)
    
    logger.info("Initializing VAE...")
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to(device, dtype=weight_dtype)
    
    logger.info("Initializing Reference UNet...")
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device, dtype=weight_dtype)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    
    logger.info("Initializing Denoising UNet...")
    if os.path.exists(config.motion_module_path):
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)
    else:
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
            }
        ).to(dtype=weight_dtype, device=device)
    
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False
    )
    
    logger.info("Initializing Pose Encoder...")
    pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device
    )
    pose_net.load_state_dict(torch.load(config.pose_encoder_path))
    
    logger.info("Loading Audio Processor...")
    audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)
    
    logger.info("Creating Pipeline...")
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)
    
    pipeline = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    )
    
    pipeline = pipeline.to(device, dtype=weight_dtype)
    
    # Apply quantization for memory efficiency
    try:
        from torchao.quantization import quantize_, int8_weight_only
        quantize_(denoising_unet, int8_weight_only())
        logger.info("Applied INT8 quantization to denoising_unet")
    except ImportError:
        logger.warning("torchao not available, skipping quantization")
    except Exception as e:
        logger.warning(f"Could not apply quantization: {e}")
    
    logger.info("Models loaded successfully!")
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: Download missing models from S3 (if configured), then load into memory
    logger.info("Starting up FastAPI server...")
    
    try:
        # Check for missing models and download from S3 if needed
        download_from_s3_if_needed()
        
        # Load models into memory
        load_models()
        logger.info("✓ Server ready to accept requests!")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    global pipeline
    if pipeline:
        del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Create FastAPI app
app = FastAPI(
    title="EchoMimicV2 API",
    description="Audio-driven human animation generation API",
    version="2.0",
    lifespan=lifespan
)


@app.post("/generate-video")
async def generate_video(
    image: UploadFile = File(..., description="Reference image (PNG/JPG)"),
    audio: UploadFile = File(..., description="Audio file (WAV/MP3)"),
    pose_dir_name: Optional[str] = "01",
    width: Optional[int] = 768,
    height: Optional[int] = 768,
    length: Optional[int] = None,
    steps: Optional[int] = 6,
    cfg: Optional[float] = 1.0,
    fps: Optional[int] = 24,
    seed: Optional[int] = 420,
):
    """
    Generate video from image and audio
    
    Args:
        image: Reference image file
        audio: Audio file
        length: Video length in frames (default: 240 = 10 seconds at 24fps)
        steps: Denoising steps (default: 4)
        cfg: Classifier-free guidance scale (default: 2.5)
        fps: Frames per second (default: 24)
    
    Returns:
        Video file with synchronized audio
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Create temp directory for processing
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Save uploaded files
        image_path = temp_dir / f"reference.{image.filename.split('.')[-1]}"
        audio_path = temp_dir / f"audio.{audio.filename.split('.')[-1]}"
        
        with open(image_path, "wb") as f:
            f.write(await image.read())
        
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        
        logger.info(f"Processing - Image: {image.filename}, Audio: {audio.filename}")
        
        # Setup paths
        pose_dir = Path(f"./assets/halfbody_demo/pose/{pose_dir_name}")
        if not pose_dir.exists():
            raise HTTPException(status_code=400, detail=f"Pose directory '{pose_dir_name}' not found")
        
        # Load audio and calculate length
        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = audio_clip.duration
        
        # Get pose count
        pose_count = len([f for f in os.listdir(pose_dir) if f.endswith('.npy')])
        
        # Calculate video length
        max_audio_frames = int(audio_duration * fps)
        if length is None:
            video_length = max_audio_frames
        else:
            video_length = min(length, max_audio_frames)
        
        logger.info(f"Audio: {audio_duration:.2f}s ({max_audio_frames} frames)")
        logger.info(f"Pose files: {pose_count}")
        logger.info(f"Generating: {video_length} frames ({video_length/fps:.2f}s)")
        logger.info(f"Pose loops: {video_length / pose_count:.2f}x")
        
        # Load reference image
        ref_img_pil = Image.open(image_path).convert("RGB")
        
        # Generate seed
        if seed is not None and seed > -1:
            generator = torch.manual_seed(seed)
        else:
            generator = torch.manual_seed(42)
        
        # Load poses with looping
        logger.info("Loading poses...")
        pose_list = []
        for index in range(video_length):
            # Use modulo to loop poses infinitely
            pose_index = index % pose_count
            
            tgt_musk = np.zeros((height, width, 3)).astype('uint8')
            tgt_musk_path = os.path.join(pose_dir, f"{pose_index}.npy")
            
            if not os.path.exists(tgt_musk_path):
                logger.warning(f"Pose file {pose_index}.npy not found, skipping")
                continue
            
            detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
            im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=min(width, height))
            im = np.transpose(np.array(im), (1, 2, 0))
            
            # Calculate target dimensions
            target_h = re - rb
            target_w = ce - cb
            
            # Resize if needed
            if im.shape[0] != target_h or im.shape[1] != target_w:
                if target_h > 0 and target_w > 0:
                    im = cv2.resize(im, (target_w, target_h))
            
            # Ensure coordinates are within bounds
            if rb >= 0 and cb >= 0 and re <= height and ce <= width and rb < re and cb < ce:
                tgt_musk[rb:re, cb:ce, :] = im
            else:
                logger.warning(f"Pose coordinates out of bounds for frame {index}")
                continue
            
            tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
            pose_list.append(
                torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device=device).permute(2, 0, 1) / 255.0
            )
        
        poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
        
        # Trim audio to video length
        audio_clip_trimmed = audio_clip.set_duration(video_length / fps)
        
        # Generate video
        logger.info("Generating video frames...")
        with torch.no_grad():
            video = pipeline(
                ref_img_pil,
                str(audio_path),
                poses_tensor[:, :, :video_length, ...],
                width,
                height,
                video_length,
                steps,
                cfg,
                generator=generator,
                audio_sample_rate=16000,
                context_frames=12,
                fps=fps,
                context_overlap=3,
                start_idx=0
            ).videos
        
        # Ensure final length
        final_length = min(video.shape[2], poses_tensor.shape[2], video_length)
        video_sig = video[:, :, :final_length, :, :]
        
        # Save video without audio
        output_path_no_audio = temp_dir / "output_no_audio.mp4"
        save_videos_grid(
            video_sig,
            str(output_path_no_audio),
            n_rows=1,
            fps=fps,
        )
        
        # Add audio
        logger.info("Adding audio to video...")
        output_path = temp_dir / "output.mp4"
        video_clip = VideoFileClip(str(output_path_no_audio))
        video_clip = video_clip.set_audio(audio_clip_trimmed)
        video_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            threads=2,
            logger=None
        )
        
        # Clean up
        video_clip.close()
        audio_clip.close()
        audio_clip_trimmed.close()
        
        logger.info(f"Video generated successfully: {output_path}")
        
        # Return video file
        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=f"generated_{Path(image.filename).stem}_{Path(audio.filename).stem}.mp4"
        )
    
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": pipeline is not None,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.get("/models-status")
async def models_status():
    """Check which models are available"""
    status = check_models_available()
    
    return {
        "models_directory": str(PRETRAINED_WEIGHTS_DIR),
        "models": status,
        "all_available": all(status.values())
    }


@app.get("/pose-directories")
async def list_pose_directories():
    """List available pose directories"""
    pose_base = Path("./assets/halfbody_demo/pose")
    if not pose_base.exists():
        return {"pose_directories": []}
    
    pose_dirs = [d.name for d in pose_base.iterdir() if d.is_dir()]
    pose_info = {}
    
    for pose_dir in pose_dirs:
        pose_path = pose_base / pose_dir
        pose_count = len([f for f in pose_path.iterdir() if f.suffix == '.npy'])
        pose_info[pose_dir] = {
            "file_count": pose_count,
            "duration_at_24fps": f"{pose_count / 24:.2f}s"
        }
    
    return {
        "pose_base_directory": str(pose_base),
        "pose_directories": pose_info
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)