#!/usr/bin/env python3
"""
Quick test script to verify the fixes are working
Run this after starting the API server
"""

import requests
import sys

API_URL = "http://localhost:8000"

def test_health():
    """Test if server is running"""
    print("1. Testing server health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Server is healthy")
            print(f"   ✓ Models loaded: {data['models_loaded']}")
            print(f"   ✓ Device: {data['device']}")
            print(f"   ✓ CUDA available: {data['cuda_available']}")
            if data['gpu_name']:
                print(f"   ✓ GPU: {data['gpu_name']}")
            return True
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_models_status():
    """Test if models are available"""
    print("\n2. Testing models status...")
    try:
        response = requests.get(f"{API_URL}/models-status")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ All models available: {data['all_available']}")
            print(f"   ✓ Pipeline loaded: {data['pipeline_loaded']}")
            
            if not data['all_available']:
                print("   ⚠ Missing models:")
                for model in data.get('missing_models', []):
                    print(f"     - {model}")
            return data['all_available']
        else:
            print(f"   ✗ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_pose_directories():
    """Test if pose directories are available"""
    print("\n3. Testing pose directories...")
    try:
        response = requests.get(f"{API_URL}/pose-directories")
        if response.status_code == 200:
            data = response.json()
            pose_dirs = data['pose_directories']
            print(f"   ✓ Found {len(pose_dirs)} pose directories:")
            for name, info in pose_dirs.items():
                print(f"     - {name}: {info['file_count']} files ({info['duration_at_24fps']})")
            return len(pose_dirs) > 0
        else:
            print(f"   ✗ Pose check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("EchoMimic V2 API - Verification Script")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Models Status", test_models_status()))
    results.append(("Pose Directories", test_pose_directories()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Server is ready for video generation.")
        print("\nNext steps:")
        print("1. Open http://localhost:8000/docs to test the API")
        print("2. Try generating a video with small parameters first:")
        print("   - width=512, height=512")
        print("   - steps=4")
        print("   - length=120 (5 seconds)")
        return 0
    else:
        print("\n✗ Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure the server is running: uvicorn api:app --host 0.0.0.0 --port 8000")
        print("2. Check that all model files are downloaded")
        print("3. Verify GPU is available: nvidia-smi")
        return 1

if __name__ == "__main__":
    sys.exit(main())
