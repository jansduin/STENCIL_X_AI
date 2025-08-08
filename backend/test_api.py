"""
Test Script for Stencil AI API
Basic testing script to validate API endpoints

Author: Stencil AI Team
Date: 2024
"""

import requests
import json
import time

def test_api_endpoints():
    """Test basic API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Stencil AI API...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint failed: {e}")
    
    # Test API status endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/status", timeout=5)
        print(f"✅ API status endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API status endpoint failed: {e}")
    
    # Test stencil styles endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/stencils/styles", timeout=5)
        print(f"✅ Stencil styles endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Stencil styles endpoint failed: {e}")
    
    # Test model info endpoint
    try:
        response = requests.get(f"{base_url}/api/v1/stencils/model/info", timeout=5)
        print(f"✅ Model info endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info endpoint failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting API tests...")
    test_api_endpoints()
    print("✅ Testing completed!") 