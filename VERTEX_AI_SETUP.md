# Vertex AI Setup Guide

The codebase now uses the modern Google Generative AI SDK which supports **both** Vertex AI and free tier:

- **Free Gemini API**: 2 requests/minute, 50 requests/day (good for testing)
- **Vertex AI**: 300-1000+ requests/minute, unlimited requests (pay per token, production-ready)

The code automatically chooses the best option based on your environment variables!

## Setup Steps

### 1. Install gcloud CLI (if not already installed)

```bash
# macOS
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### 2. Set Your Default Project First

```bash
# IMPORTANT: Set your project ID first (replace with your actual project ID)
gcloud config set project YOUR-PROJECT-ID
```

### 3. Authenticate with Google Cloud

```bash
# Login to your Google Cloud account
gcloud auth application-default login

# Set quota project to avoid warnings
gcloud auth application-default set-quota-project YOUR-PROJECT-ID
```

### 4. Set Environment Variables

Add these to your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
# Required: Your GCP project ID
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

# Optional: Change region if needed (default is us-central1)
export GOOGLE_CLOUD_LOCATION="us-central1"

# Optional: Keep this if you want fallback to free API
export GEMINI_API_KEY="your-api-key"  # Optional fallback

# Existing settings you should keep
export VIZ=0
export LOCAL_LOGS_ONLY=1
```

Then reload your shell:
```bash
source ~/.zshrc  # or source ~/.bashrc
```

### 5. Enable Vertex AI API in GCP

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Select your project
3. Navigate to "APIs & Services" > "Library"
4. Search for "Vertex AI API"
5. Click "Enable"

### 6. Verify Setup

Run a quick test to ensure it's working:

```bash
# Check your credentials
gcloud auth application-default print-access-token

# Verify project is set
echo $GOOGLE_CLOUD_PROJECT
```

### 7. Run Your Code

Now you can run your ARC challenges with Vertex AI:

```bash
python -m src.run
```

## Pricing

Vertex AI Gemini pricing (as of 2024):
- **Gemini 2.5 Pro**: ~$3.50 per 1M input tokens, ~$10.50 per 1M output tokens
- **Gemini 2.0 Flash**: ~$0.075 per 1M input tokens, ~$0.30 per 1M output tokens

For the ARC challenge workload (10 challenges with your config), expect:
- ~35-50 API calls per challenge
- ~500 total API calls for 10 challenges
- Cost estimate: $1-5 depending on prompt/response sizes

## Troubleshooting

### "Project not set" error
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### "Permission denied" errors
```bash
gcloud auth application-default login
```

### Rate limit errors (unlikely with Vertex AI)
- Increase `max_concurrent_tasks` in your config if hitting limits
- Current setting: 120 concurrent tasks (way more than Gemini free tier's 2!)

### Want to use free tier as fallback?
Set `GEMINI_API_KEY` environment variable - the code will fall back to it if Vertex AI fails.

## Benefits Over Free Gemini API

1. **Much Higher Rate Limits**: 300+ requests/minute vs 2
2. **No Daily Limits**: Pay per use vs 50 requests/day
3. **Better Reliability**: Enterprise-grade SLA
4. **Same Models**: Uses the exact same Gemini models
5. **Better Concurrency**: Can run 120 concurrent tasks vs basically 1

Enjoy the improved performance! 🚀

