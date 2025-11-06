#!/usr/bin/env python3
"""Quick diagnostic script to test Gemini API setup."""

import os
import sys

print("=" * 60)
print("GEMINI API SETUP DIAGNOSTIC")
print("=" * 60)

# Check environment variables
print("\n1. Checking Environment Variables:")
print("-" * 60)

gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
gemini_key = os.environ.get("GEMINI_API_KEY")
gcp_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if gcp_project:
    print(f"✅ GOOGLE_CLOUD_PROJECT: {gcp_project}")
    print(f"   Location: {gcp_location}")
    backend = "Vertex AI"
elif gemini_key:
    print(f"✅ GEMINI_API_KEY: {gemini_key[:10]}..." if len(gemini_key) > 10 else "✅ GEMINI_API_KEY: (set)")
    backend = "Free Tier API"
else:
    print("❌ Neither GOOGLE_CLOUD_PROJECT nor GEMINI_API_KEY is set!")
    print("\n   To fix:")
    print("   Option 1 (Free): export GEMINI_API_KEY='your-key'")
    print("   Option 2 (Vertex AI): export GOOGLE_CLOUD_PROJECT='your-project-id'")
    sys.exit(1)

print(f"\n   Will use: {backend}")

# Test import
print("\n2. Testing Imports:")
print("-" * 60)

try:
    from google import genai
    print("✅ google.genai imported successfully")
except ImportError as e:
    print(f"❌ Failed to import google.genai: {e}")
    sys.exit(1)

# Initialize client
print("\n3. Initializing Gemini Client:")
print("-" * 60)

try:
    if gcp_project:
        client = genai.Client(
            vertexai=True,
            project=gcp_project,
            location=gcp_location
        )
        print(f"✅ Vertex AI client initialized for project: {gcp_project}")
    else:
        client = genai.Client(api_key=gemini_key)
        print("✅ Free tier client initialized")
except Exception as e:
    print(f"❌ Failed to initialize client: {e}")
    sys.exit(1)

# Test API call
print("\n4. Testing API Call:")
print("-" * 60)

try:
    print("   Sending test request to Gemini...")
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents="Say 'API is working!' and nothing else.",
    )
    
    result = response.text
    print(f"✅ API call successful!")
    print(f"   Response: {result}")
    
except Exception as e:
    print(f"❌ API call failed: {e}")
    print("\n   Common fixes:")
    if gcp_project:
        print("   - Run: gcloud auth application-default login")
        print("   - Enable Vertex AI API in console")
        print("   - Check project ID is correct")
    else:
        print("   - Verify API key is valid at https://aistudio.google.com/apikey")
        print("   - Check you haven't hit rate limits")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED! Your Gemini setup is working correctly.")
print("=" * 60)
print("\nYou can now run your ARC challenges:")
print("  python -m src.run")
print()


