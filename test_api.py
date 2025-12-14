"""
Simple API test - Run this AFTER starting the API
"""
import requests
import json

# Sample breast cancer data (30 features)
sample = {
    "features": [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
        184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
}

print("🧪 Testing API...")
print("=" * 60)

# Test 1: Health check
print("\n1️⃣ Testing /health endpoint...")
try:
    response = requests.get("http://localhost:8000/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Prediction
print("\n2️⃣ Testing /predict endpoint...")
try:
    response = requests.post(
        "http://localhost:8000/predict",
        json=sample
    )
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Prediction: {result['prediction']}")
    print(f"   Probability: {result['probability']:.4f}")
    print(f"   Diagnosis: {result['diagnosis']}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ Testing complete!")