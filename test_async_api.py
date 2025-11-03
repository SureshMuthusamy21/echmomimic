"""
Test script for async video generation API
This demonstrates the new background task flow with S3 upload
"""

import requests
import time
from pathlib import Path

# API configuration
API_URL = "http://localhost:8000"

def test_async_video_generation():
    """Test the new async video generation endpoint"""
    
    # Prepare test files
    image_path = Path("./assets/halfbody_demo/refimag/natural_bk_openhand/0212.png")
    audio_path = Path("./assets/halfbody_demo/audio/chinese/AI配音_轻松搞定数字人视频，让内容更生动.wav")
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    if not audio_path.exists():
        print(f"❌ Audio not found: {audio_path}")
        return
    
    print("=" * 80)
    print("Testing Async Video Generation API")
    print("=" * 80)
    
    # Step 1: Submit job
    print("\n📤 Step 1: Submitting video generation job...")
    
    files = {
        'image': ('reference.png', open(image_path, 'rb'), 'image/png'),
        'audio': ('audio.wav', open(audio_path, 'rb'), 'audio/wav'),
    }
    
    data = {
        'pose_dir_name': '01',
        'width': 768,
        'height': 768,
        'steps': 6,
        'cfg': 1.0,
        'fps': 24,
        'seed': 420,
    }
    
    response = requests.post(f"{API_URL}/generate-video", files=files, data=data)
    
    if response.status_code != 200:
        print(f"❌ Failed to submit job: {response.status_code}")
        print(response.json())
        return
    
    result = response.json()
    job_id = result['job_id']
    
    print(f"✅ Job created successfully!")
    print(f"   Job ID: {job_id}")
    print(f"   Status: {result['status']}")
    print(f"   Message: {result['message']}")
    print(f"   Status URL: {API_URL}{result['status_url']}")
    
    # Step 2: Poll for status
    print(f"\n⏳ Step 2: Polling job status...")
    print("   (This may take several minutes depending on video length)")
    
    max_polls = 300  # 5 minutes max (300 * 1 second)
    poll_count = 0
    
    while poll_count < max_polls:
        time.sleep(1)  # Poll every second
        poll_count += 1
        
        status_response = requests.get(f"{API_URL}/job-status/{job_id}")
        
        if status_response.status_code != 200:
            print(f"❌ Failed to get status: {status_response.status_code}")
            break
        
        job_status = status_response.json()
        status = job_status['status']
        progress = job_status.get('progress', 'No progress info')
        
        # Print progress every 5 seconds
        if poll_count % 5 == 0:
            print(f"   [{poll_count}s] Status: {status} - {progress}")
        
        # Check if completed or failed
        if status == 'completed':
            print("\n✅ Job completed successfully!")
            print(f"   Video URL: {job_status['video_url']}")
            print(f"   S3 Key: {job_status['s3_key']}")
            print(f"   Duration: {job_status.get('video_duration', 'N/A')}")
            print(f"   Total Frames: {job_status.get('total_frames', 'N/A')}")
            print(f"   File Size: {job_status.get('file_size_bytes', 0) / 1024 / 1024:.2f} MB")
            print(f"   Completed At: {job_status['completed_at']}")
            break
        
        elif status == 'failed':
            print(f"\n❌ Job failed!")
            print(f"   Error: {job_status.get('error', 'Unknown error')}")
            print(f"   Failed At: {job_status['failed_at']}")
            break
    
    if poll_count >= max_polls:
        print(f"\n⚠️ Timeout: Job still processing after {max_polls} seconds")
    
    # Step 3: List all jobs
    print("\n📋 Step 3: Listing all jobs...")
    
    jobs_response = requests.get(f"{API_URL}/list-jobs?limit=5")
    if jobs_response.status_code == 200:
        jobs_data = jobs_response.json()
        print(f"   Total jobs: {jobs_data['total_jobs']}")
        print(f"   Recent jobs:")
        for job in jobs_data['jobs'][:3]:
            print(f"     - {job['job_id'][:8]}... | {job['status']} | {job['created_at']}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_async_video_generation()
