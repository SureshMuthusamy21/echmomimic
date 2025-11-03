#!/usr/bin/env python3
"""
Video validation script - checks if generated videos are valid
"""

import cv2
import numpy as np
import sys
from pathlib import Path

def analyze_video(video_path):
    """Analyze video file for common issues"""
    print(f"\nAnalyzing: {video_path}")
    print("=" * 60)
    
    if not Path(video_path).exists():
        print("✗ Error: Video file does not exist")
        return False
    
    file_size = Path(video_path).stat().st_size
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    if file_size < 1000:
        print("✗ Error: File too small, likely corrupted")
        return False
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("✗ Error: Cannot open video file")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {duration:.2f}s")
    
    if frame_count == 0:
        print("✗ Error: No frames in video")
        cap.release()
        return False
    
    # Sample frames for analysis
    sample_indices = [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1]
    sample_indices = [i for i in sample_indices if i < frame_count]
    
    print(f"\nAnalyzing {len(sample_indices)} sample frames...")
    
    issues = []
    blank_frames = 0
    valid_frames = 0
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            issues.append(f"Frame {idx}: Failed to read")
            continue
        
        # Check if frame is blank (all black, all white, or very uniform)
        mean_val = np.mean(frame)
        std_val = np.std(frame)
        min_val = np.min(frame)
        max_val = np.max(frame)
        
        # Debug info for first frame
        if idx == 0:
            print(f"\nFrame {idx} stats:")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Std: {std_val:.2f}")
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
        
        # Check for blank frame
        if std_val < 1.0:  # Very low variance = uniform color
            blank_frames += 1
            issues.append(f"Frame {idx}: Appears blank (std={std_val:.2f})")
        elif mean_val < 5 or mean_val > 250:  # All black or all white
            blank_frames += 1
            issues.append(f"Frame {idx}: Too dark or too bright (mean={mean_val:.2f})")
        else:
            valid_frames += 1
    
    cap.release()
    
    # Report results
    print(f"\nResults:")
    print(f"  Valid frames: {valid_frames}/{len(sample_indices)}")
    print(f"  Blank frames: {blank_frames}/{len(sample_indices)}")
    
    if issues:
        print(f"\n⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Overall assessment
    print("\n" + "=" * 60)
    if blank_frames == len(sample_indices):
        print("✗ FAIL: All sampled frames are blank!")
        print("\nLikely causes:")
        print("  1. NaN values in video tensor (check logs for warnings)")
        print("  2. Model quantization issues")
        print("  3. Invalid pose or reference image data")
        return False
    elif blank_frames > len(sample_indices) // 2:
        print("⚠ WARNING: More than half of sampled frames are blank")
        return False
    elif blank_frames > 0:
        print("⚠ WARNING: Some frames appear blank, but video may still be usable")
        return True
    else:
        print("✓ PASS: Video appears valid!")
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_video.py <video_path>")
        print("\nExample:")
        print("  python validate_video.py output/20251103/video.mp4")
        return 1
    
    video_path = sys.argv[1]
    success = analyze_video(video_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
