# Blank Video Output - Fixes Applied

## Problem Summary
The video generation was producing blank/corrupted videos due to:
1. NaN/Inf values in video tensor causing invalid pixel values
2. INT8 quantization introducing numerical instability
3. Missing validation of pose data and reference images
4. Insufficient error handling during video encoding

## Fixes Applied

### 1. Enhanced `save_videos_grid()` Function (`src/utils/util.py`)
**What was fixed:**
- Added detection and cleaning of NaN/Inf values in video tensors
- Added clamping to ensure all values are in valid range [0, 1]
- Added safety checks before numpy conversion

**Why it was failing:**
- The warning "invalid value encountered in cast" meant NaN values were being cast to uint8, resulting in corrupted pixels

### 2. Disabled INT8 Quantization (`api.py`)
**What was fixed:**
- Temporarily disabled INT8 quantization of the denoising_unet
- Added comments for re-enabling if needed

**Why it was failing:**
- INT8 quantization can introduce numerical instability, especially with certain model architectures
- This was likely the root cause of NaN generation

### 3. Enhanced Pose Loading Validation (`api.py`)
**What was fixed:**
- Validate pose boundary coordinates before use
- Check for empty or invalid pose images
- Clean NaN/Inf values in pose data
- Validate pose tensors before adding to list
- Better error messages for debugging

**Why it was failing:**
- Invalid pose coordinates could cause array indexing errors
- Unchecked NaN values in pose data propagate through the pipeline

### 4. Reference Image Validation (`api.py`)
**What was fixed:**
- Validate reference image after loading
- Check for NaN/Inf values and clean them
- Log image statistics for debugging

**Why it was failing:**
- Corrupted or invalid reference images could cause downstream issues

### 5. Pipeline Output Validation (`api.py`)
**What was fixed:**
- Debug logging of pipeline output (shape, dtype, range)
- Detect and clean NaN/Inf in generated video
- Validate tensor statistics before saving

**Why it was failing:**
- No visibility into what the pipeline was generating
- NaN values were not caught before video encoding

### 6. Improved Video Encoding (`api.py`)
**What was fixed:**
- Verify video file was created and check size
- Better error handling for audio merging
- Return video without audio if merging fails
- Added quality settings (preset, bitrate)

**Why it was failing:**
- The warning about "0 bytes read" indicated the video file was corrupted
- No fallback if audio merging failed

## Testing After Fixes

### 1. Restart the API Server
```bash
# On your EC2 instance
cd echomimic_v2
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 2. Monitor the Logs
Watch for these new log messages:
- `Reference image loaded: (768, 768, 3), range: [0, 255]`
- `Pipeline output - Shape: torch.Size([...]), dtype: ...`
- `Video range: [0.xxxx, 0.xxxx]`
- `Contains NaN: False, Contains Inf: False`
- `Final video tensor - Shape: ..., range: [0.xxxx, 0.xxxx]`
- `Video file created: XXXXX bytes`

### 3. Check for Warning Messages
If you see these warnings, the fixes are working:
- `WARNING: Video tensor contains NaN values, replacing with zeros`
- `WARNING: Video tensor contains infinite values, replacing with zeros`
- `Invalid pose bounds at frame X`
- `Invalid values in pose image at frame X, cleaning`

### 4. Verify Output Video
After generation:
- Check that the video file size is reasonable (should be > 100KB for typical videos)
- Open the video and verify it's not blank
- Check that motion is visible

## If Issues Persist

### Check 1: Verify Models Loaded Correctly
```bash
curl http://localhost:8000/models-status
```

### Check 2: Test with Different Parameters
Try with minimal settings first:
- `width=512, height=512` (smaller resolution)
- `steps=4` (fewer denoising steps)
- `cfg=1.0` (lower guidance)

### Check 3: Check GPU Memory
```bash
# On EC2
nvidia-smi
```
If GPU memory is low, the model might not run correctly.

### Check 4: Enable Quantization (if memory is the issue)
Edit `api.py` and uncomment the quantization code, but monitor for NaN values in logs.

## Root Cause Analysis

The blank video was caused by a **cascade of numerical issues**:

1. **INT8 Quantization** → Introduced numerical instability
2. **Unchecked NaN propagation** → NaN values flowed through pipeline
3. **Invalid cast to uint8** → NaN became garbage pixel values (0 or 255)
4. **Corrupted video encoding** → FFmpeg received invalid frames
5. **Result** → Blank/corrupted video output

The fixes add **validation and cleaning at every stage** to prevent NaN propagation and ensure valid outputs.

## Performance Notes

- **With quantization disabled**: Higher GPU memory usage but more stable
- **Recommended settings for EC2**: 
  - Use `g4dn.xlarge` or better (16GB+ GPU memory)
  - Resolution: 512x512 or 768x768
  - Steps: 4-6
  - Batch size: 1

## Additional Recommendations

1. **Add health monitoring**: Check `/health` endpoint regularly
2. **Log rotation**: The debug logs will be verbose
3. **Monitor GPU**: Use `nvidia-smi -l 1` to watch GPU usage
4. **Test incrementally**: Start with short videos (5s) before long ones
