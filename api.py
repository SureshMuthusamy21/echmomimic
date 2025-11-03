import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
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
import os
from dotenv import load_dotenv

load_dotenv()  # <-- this loads .env into the environment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "echomimic-video-gen-models")  # Get from env or use default
S3_OUTPUT_BUCKET = "demo-goml"  # Output bucket for generated videos
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

# Job status tracking
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"

# In-memory job storage (use Redis/database for production)
jobs: Dict[str, Dict] = {}


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


def upload_to_s3(file_path: Path, s3_key: str) -> str:
    """
    Upload file to S3 and return public URL
    
    Args:
        file_path: Local file path to upload
        s3_key: S3 key (filename in bucket)
    
    Returns:
        S3 URL of uploaded file
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3_client = boto3.client('s3')
        
        # Upload file
        logger.info(f"Uploading {file_path.name} to s3://{S3_OUTPUT_BUCKET}/{s3_key}")
        s3_client.upload_file(
            str(file_path),
            S3_OUTPUT_BUCKET,
            s3_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        
        # Generate public URL (or presigned URL for private buckets)
        s3_url = f"https://{S3_OUTPUT_BUCKET}.s3.amazonaws.com/{s3_key}"
        
        # Alternative: Generate presigned URL (expires after 7 days)
        # s3_url = s3_client.generate_presigned_url(
        #     'get_object',
        #     Params={'Bucket': S3_OUTPUT_BUCKET, 'Key': s3_key},
        #     ExpiresIn=604800  # 7 days
        # )
        
        logger.info(f"Upload successful: {s3_url}")
        return s3_url
        
    except ImportError:
        logger.error("boto3 not installed. Cannot upload to S3.")
        raise Exception("boto3 not available for S3 upload")
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        raise Exception(f"Failed to upload to S3: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during S3 upload: {e}")
        raise


def process_video_task(
    job_id: str,
    image_path: Path,
    audio_path: Path,
    pose_dir_name: str,
    width: int,
    height: int,
    length: Optional[int],
    steps: int,
    cfg: float,
    fps: int,
    seed: int,
    image_filename: str,
    audio_filename: str
):
    """
    Background task to process video generation and upload to S3
    """
    temp_dir = image_path.parent
    
    try:
        # Update job status
        jobs[job_id]["status"] = JobStatus.PROCESSING
        jobs[job_id]["progress"] = "Loading resources..."
        
        logger.info(f"[Job {job_id}] Starting video generation")
        
        # Setup paths
        pose_dir = Path(f"./assets/halfbody_demo/pose/{pose_dir_name}")
        if not pose_dir.exists():
            raise Exception(f"Pose directory '{pose_dir_name}' not found")
        
        # Load audio and calculate length
        jobs[job_id]["progress"] = "Processing audio..."
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
        
        logger.info(f"[Job {job_id}] Audio: {audio_duration:.2f}s ({max_audio_frames} frames)")
        logger.info(f"[Job {job_id}] Generating: {video_length} frames ({video_length/fps:.2f}s)")
        
        jobs[job_id]["video_duration"] = f"{video_length/fps:.2f}s"
        jobs[job_id]["total_frames"] = video_length
        
        # Load reference image
        jobs[job_id]["progress"] = "Loading reference image..."
        ref_img_pil = Image.open(image_path).convert("RGB")
        ref_img_pil = ref_img_pil.resize((width, height))
        
        # Validate reference image
        ref_array = np.array(ref_img_pil)
        if ref_array.size == 0:
            raise Exception("Invalid reference image: empty array")
        
        if np.isnan(ref_array).any() or np.isinf(ref_array).any():
            logger.warning(f"[Job {job_id}] Reference image contains invalid values, cleaning")
            ref_array = np.nan_to_num(ref_array, nan=0.0, posinf=255.0, neginf=0.0)
            ref_img_pil = Image.fromarray(ref_array.astype(np.uint8))
        
        # Generate seed
        if seed is not None and seed > -1:
            generator = torch.manual_seed(seed)
        else:
            generator = torch.manual_seed(42)
        
        # Load poses with looping
        jobs[job_id]["progress"] = f"Loading {video_length} pose frames..."
        logger.info(f"[Job {job_id}] Loading poses...")
        pose_list = []
        
        for index in range(video_length):
            pose_index = index % pose_count
            
            tgt_musk = np.zeros((height, width, 3)).astype('uint8')
            tgt_musk_path = os.path.join(pose_dir, f"{pose_index}.npy")
            
            if not os.path.exists(tgt_musk_path):
                logger.warning(f"[Job {job_id}] Pose file {pose_index}.npy not found, skipping")
                continue
            
            detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
            
            # Validate pose parameters
            if rb < 0 or cb < 0 or re > height or ce > width or rb >= re or cb >= ce:
                rb = max(0, min(rb, height))
                re = max(rb + 1, min(re, height))
                cb = max(0, min(cb, width))
                ce = max(cb + 1, min(ce, width))
            
            im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=min(width, height))
            im = np.transpose(np.array(im), (1, 2, 0))
            
            if im.size == 0:
                tgt_musk_pil = Image.fromarray(tgt_musk).convert('RGB')
                pose_list.append(
                    torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device=device).permute(2, 0, 1) / 255.0
                )
                continue
            
            if np.isnan(im).any() or np.isinf(im).any():
                im = np.nan_to_num(im, nan=0.0, posinf=255.0, neginf=0.0)
            
            target_h = re - rb
            target_w = ce - cb
            
            if im.shape[0] != target_h or im.shape[1] != target_w:
                if target_h > 0 and target_w > 0:
                    im = cv2.resize(im, (target_w, target_h))
                else:
                    continue
            
            if rb >= 0 and cb >= 0 and re <= height and ce <= width and rb < re and cb < ce:
                tgt_musk[rb:re, cb:ce, :] = im
            else:
                continue
            
            tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
            pose_tensor = torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device=device).permute(2, 0, 1) / 255.0
            
            if torch.isnan(pose_tensor).any() or torch.isinf(pose_tensor).any():
                pose_tensor = torch.nan_to_num(pose_tensor, nan=0.0, posinf=1.0, neginf=0.0)
                pose_tensor = torch.clamp(pose_tensor, 0.0, 1.0)
            
            pose_list.append(pose_tensor)
        
        poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
        audio_clip_trimmed = audio_clip.set_duration(video_length / fps)
        
        # Generate video
        jobs[job_id]["progress"] = f"Generating video ({video_length} frames)... This may take several minutes."
        logger.info(f"[Job {job_id}] Generating video frames...")
        
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
        
        # Clean invalid values
        if torch.isnan(video).any() or torch.isinf(video).any():
            logger.warning(f"[Job {job_id}] Video tensor contains invalid values! Cleaning...")
            video = torch.nan_to_num(video, nan=0.0, posinf=1.0, neginf=0.0)
            video = torch.clamp(video, 0.0, 1.0)
        
        final_length = min(video.shape[2], poses_tensor.shape[2], video_length)
        video_sig = video[:, :, :final_length, :, :]
        
        # Save video without audio
        jobs[job_id]["progress"] = "Encoding video..."
        logger.info(f"[Job {job_id}] Saving video frames...")
        
        output_path_no_audio = temp_dir / "output_no_audio.mp4"
        save_videos_grid(
            video_sig,
            str(output_path_no_audio),
            n_rows=1,
            fps=fps,
        )
        
        if not output_path_no_audio.exists():
            raise Exception("Failed to create video file")
        
        video_size = output_path_no_audio.stat().st_size
        if video_size < 1000:
            raise Exception("Generated video is corrupted")
        
        # Add audio
        jobs[job_id]["progress"] = "Adding audio..."
        logger.info(f"[Job {job_id}] Adding audio to video...")
        
        output_path = temp_dir / f"{job_id}.mp4"
        
        try:
            video_clip = VideoFileClip(str(output_path_no_audio))
            video_clip = video_clip.set_audio(audio_clip_trimmed)
            video_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                threads=2,
                logger=None,
                preset='medium',
                bitrate='5000k'
            )
            video_clip.close()
        except Exception as e:
            logger.error(f"[Job {job_id}] Error adding audio: {e}")
            output_path = output_path_no_audio
        
        audio_clip.close()
        audio_clip_trimmed.close()
        
        logger.info(f"[Job {job_id}] Video generated successfully: {output_path}")
        
        # Upload to S3
        jobs[job_id]["status"] = JobStatus.UPLOADING
        jobs[job_id]["progress"] = "Uploading to S3..."
        
        s3_key = f"generated/{job_id}.mp4"
        s3_url = upload_to_s3(output_path, s3_key)
        
        # Update job as completed
        jobs[job_id]["status"] = JobStatus.COMPLETED
        jobs[job_id]["progress"] = "Completed"
        jobs[job_id]["video_url"] = s3_url
        jobs[job_id]["s3_key"] = s3_key
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        jobs[job_id]["file_size_bytes"] = output_path.stat().st_size
        
        logger.info(f"[Job {job_id}] ✓ Completed! Video available at: {s3_url}")
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Error: {e}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["failed_at"] = datetime.utcnow().isoformat()
    
    finally:
        # Cleanup temp files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"[Job {job_id}] Cleaned up temp directory")
        except:
            pass


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
    # DISABLED: Quantization can cause NaN values in output
    # Uncomment below if you have memory issues and verify output is valid
    # try:
    #     from torchao.quantization import quantize_, int8_weight_only
    #     quantize_(denoising_unet, int8_weight_only())
    #     logger.info("Applied INT8 quantization to denoising_unet")
    # except ImportError:
    #     logger.warning("torchao not available, skipping quantization")
    # except Exception as e:
    #     logger.warning(f"Could not apply quantization: {e}")
    
    logger.info("Pipeline ready (quantization disabled for stability)")
    
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
        
        # Try to load models into memory
        try:
            load_models()
            logger.info("✓ Server ready to accept requests!")
        except RuntimeError as e:
            # Models are missing - server will start but endpoints won't work
            logger.error(f"Failed to load models: {e}")
            logger.error("=" * 80)
            logger.error("SERVER STARTED IN DEGRADED MODE - Models are not loaded!")
            logger.error("=" * 80)
            logger.error("")
            logger.error("To fix this issue, you have two options:")
            logger.error("")
            logger.error("Option 1: Configure AWS credentials to download from S3")
            logger.error("  - Configure AWS CLI: aws configure")
            logger.error("  - Or set environment variables:")
            logger.error("    export AWS_ACCESS_KEY_ID=your_access_key")
            logger.error("    export AWS_SECRET_ACCESS_KEY=your_secret_key")
            logger.error("    export AWS_DEFAULT_REGION=your_region")
            logger.error("")
            logger.error("Option 2: Download models manually")
            logger.error("  - Download the required models and place them in:")
            logger.error(f"    {PRETRAINED_WEIGHTS_DIR.absolute()}")
            logger.error("")
            logger.error("Required models:")
            for model_name, model_type in REQUIRED_MODEL_FILES.items():
                logger.error(f"  - {model_name} ({model_type})")
            logger.error("")
            logger.error("=" * 80)
            logger.error("The API will respond with 503 errors until models are available.")
            logger.error("After adding models, restart the server.")
            logger.error("=" * 80)
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}")
        # Don't raise - allow server to start
    
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
    background_tasks: BackgroundTasks,
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
    Generate video from image and audio (async background processing)
    
    Returns job ID immediately. Video is processed in background and uploaded to S3.
    Use /job-status/{job_id} to check progress.
    
    Args:
        image: Reference image file
        audio: Audio file
        pose_dir_name: Pose directory name (default: "01")
        width: Video width (default: 768)
        height: Video height (default: 768)
        length: Video length in frames (default: auto from audio)
        steps: Denoising steps (default: 6)
        cfg: Classifier-free guidance scale (default: 1.0)
        fps: Frames per second (default: 24)
        seed: Random seed (default: 420)
    
    Returns:
        JSON with job_id and status_url
    """
    if pipeline is None:
        models_status = check_models_available()
        missing_models = [k for k, v in models_status.items() if not v]
        
        error_detail = {
            "error": "Models not loaded",
            "reason": "Required model files are missing",
            "missing_models": missing_models,
            "instructions": {
                "option_1": "Configure AWS credentials and restart the server to auto-download from S3",
                "option_2": f"Manually download models to {PRETRAINED_WEIGHTS_DIR.absolute()} and restart"
            }
        }
        raise HTTPException(status_code=503, detail=error_detail)
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create temp directory for this job
    temp_dir = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))
    
    try:
        # Save uploaded files
        image_path = temp_dir / f"reference.{image.filename.split('.')[-1]}"
        audio_path = temp_dir / f"audio.{audio.filename.split('.')[-1]}"
        
        with open(image_path, "wb") as f:
            f.write(await image.read())
        
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        
        # Initialize job tracking
        jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "progress": "Queued for processing",
            "created_at": datetime.utcnow().isoformat(),
            "params": {
                "image_filename": image.filename,
                "audio_filename": audio.filename,
                "pose_dir_name": pose_dir_name,
                "width": width,
                "height": height,
                "length": length,
                "steps": steps,
                "cfg": cfg,
                "fps": fps,
                "seed": seed
            }
        }
        
        # Add background task
        background_tasks.add_task(
            process_video_task,
            job_id=job_id,
            image_path=image_path,
            audio_path=audio_path,
            pose_dir_name=pose_dir_name,
            width=width,
            height=height,
            length=length,
            steps=steps,
            cfg=cfg,
            fps=fps,
            seed=seed,
            image_filename=image.filename,
            audio_filename=audio.filename
        )
        
        logger.info(f"[Job {job_id}] Created - Image: {image.filename}, Audio: {audio.filename}")
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": JobStatus.QUEUED,
            "message": "Video generation started in background",
            "status_url": f"/job-status/{job_id}",
            "estimated_time": "Processing time depends on video length (typically 15-20 sec/sec of video)"
        })
        
    except Exception as e:
        logger.error(f"[Job {job_id}] Error creating job: {e}")
        # Clean up on error
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a video generation job
    
    Args:
        job_id: The job ID returned from /generate-video
    
    Returns:
        JSON with job status, progress, and video URL (when completed)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_info = jobs[job_id].copy()
    
    # Add helpful messages based on status
    if job_info["status"] == JobStatus.COMPLETED:
        job_info["message"] = "Video generation completed successfully!"
    elif job_info["status"] == JobStatus.FAILED:
        job_info["message"] = "Video generation failed. Check 'error' field for details."
    elif job_info["status"] == JobStatus.PROCESSING:
        job_info["message"] = "Video is being generated. This may take several minutes."
    elif job_info["status"] == JobStatus.UPLOADING:
        job_info["message"] = "Video generated, uploading to S3..."
    else:
        job_info["message"] = "Job is queued for processing."
    
    return job_info


@app.get("/list-jobs")
async def list_jobs(limit: Optional[int] = 50):
    """
    List all jobs (recent first)
    
    Args:
        limit: Maximum number of jobs to return (default: 50)
    
    Returns:
        List of all jobs with their status
    """
    job_list = list(jobs.values())
    # Sort by created_at (most recent first)
    job_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return {
        "total_jobs": len(job_list),
        "jobs": job_list[:limit]
    }


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
    missing_models = [k for k, v in status.items() if not v]
    
    response = {
        "models_directory": str(PRETRAINED_WEIGHTS_DIR.absolute()),
        "models": status,
        "all_available": all(status.values()),
        "pipeline_loaded": pipeline is not None
    }
    
    if missing_models:
        response["missing_models"] = missing_models
        response["setup_instructions"] = {
            "aws_credentials": {
                "description": "Configure AWS credentials to download from S3",
                "commands": [
                    "aws configure",
                    "# OR set environment variables:",
                    "export AWS_ACCESS_KEY_ID=your_access_key",
                    "export AWS_SECRET_ACCESS_KEY=your_secret_key",
                    "export AWS_DEFAULT_REGION=your_region"
                ]
            },
            "manual_download": {
                "description": "Download models manually",
                "s3_bucket": S3_BUCKET,
                "models_needed": missing_models,
                "target_directory": str(PRETRAINED_WEIGHTS_DIR.absolute()),
                "note": "After downloading, restart the server to load models"
            }
        }
    
    return response


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


@app.post("/download-models")
async def download_models():
    """
    Manually trigger model download from S3
    Use this after configuring AWS credentials
    """
    try:
        # Check current status
        models_status = check_models_available()
        missing_before = [k for k, v in models_status.items() if not v]
        
        if not missing_before:
            return {
                "status": "success",
                "message": "All models are already available",
                "models": models_status
            }
        
        # Try to download
        logger.info("Manual download triggered via API")
        download_from_s3_if_needed()
        
        # Check status after download
        models_status_after = check_models_available()
        missing_after = [k for k, v in models_status_after.items() if not v]
        
        if not missing_after:
            return {
                "status": "success",
                "message": "All models downloaded successfully. Please restart the server to load them.",
                "models": models_status_after,
                "downloaded": missing_before
            }
        else:
            return {
                "status": "partial",
                "message": "Some models are still missing. Check AWS credentials and S3 bucket access.",
                "models": models_status_after,
                "still_missing": missing_after,
                "downloaded": [m for m in missing_before if m not in missing_after]
            }
    
    except Exception as e:
        logger.error(f"Error during manual download: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)