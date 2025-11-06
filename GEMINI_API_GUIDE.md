# Gemini API Configuration Guide

The codebase now uses the **modern Google Generative AI SDK** which supports both free tier and Vertex AI in a unified way.

## How It Works

The code automatically detects which backend to use:

```python
# If GOOGLE_CLOUD_PROJECT is set → Uses Vertex AI (better limits)
# If only GEMINI_API_KEY is set → Uses free tier (limited)
```

## Option 1: Free Tier (Quick Start)

Perfect for testing and development:

### Setup:
```bash
# 1. Get a free API key from https://aistudio.google.com/apikey

# 2. Set the environment variable
export GEMINI_API_KEY="your-api-key-here"

# Add to your ~/.zshrc for persistence:
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Limits:
- 2 requests per minute
- 50 requests per day
- Free forever

### Best For:
- Testing your code
- Running 1-2 challenges
- Development

---

## Option 2: Vertex AI (Recommended for Production)

Much better for actual workloads:

### Setup:
```bash
# 1. Authenticate with GCP
gcloud auth application-default login

# 2. Set your project
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
gcloud config set project $GOOGLE_CLOUD_PROJECT

# 3. Set quota project (avoid warnings)
gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT

# 4. Enable Vertex AI API
# Go to: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com
# Click "Enable"
```

### Limits:
- 300-1000+ requests per minute (model dependent)
- Unlimited daily requests
- Pay per token (~$3.50 per 1M input tokens for Gemini 2.5 Pro)

### Best For:
- Running 10+ challenges
- Production workloads
- High concurrency (120 concurrent tasks)

---

## Automatic Switching

The code intelligently switches:

| Environment | Backend Used | Rate Limit |
|-------------|-------------|------------|
| `GEMINI_API_KEY` only | Free Tier | 2/min |
| `GOOGLE_CLOUD_PROJECT` set | Vertex AI | 300+/min |
| Both set | **Vertex AI** (prioritized) | 300+/min |
| Neither set | ❌ Error | N/A |

---

## Recommended Workflow

1. **Start with Free Tier** for testing:
   ```bash
   export GEMINI_API_KEY="your-key"
   python -m src.run  # Test with limit=1
   ```

2. **Upgrade to Vertex AI** when ready:
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project"
   python -m src.run  # Run with limit=10
   ```

---

## Cost Estimation (Vertex AI)

For your ARC challenge workload:

- **Config**: `gemini_config_prod` (5+5+20+5 calls per challenge)
- **Per Challenge**: ~35 API calls
- **10 Challenges**: ~350 API calls
- **Estimated Cost**: $2-5 (depends on prompt/response sizes)

**Much cheaper than hitting free tier limits and waiting!**

---

## Troubleshooting

### "API key not valid" (Free Tier)
- Verify: `echo $GEMINI_API_KEY`
- Get new key: https://aistudio.google.com/apikey

### "Project not set" (Vertex AI)
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### "Permission denied" (Vertex AI)
```bash
gcloud auth application-default login
```

### Want to verify which backend is being used?
Check the logs - they'll show either:
- `"Free tier API"` messages
- `"Vertex AI"` messages

---

## Migration Path

Already using free tier? Easy upgrade:

```bash
# Just add this - code automatically switches to Vertex AI
export GOOGLE_CLOUD_PROJECT="your-project-id"
gcloud auth application-default login

# Keep your old key as fallback if you want
# export GEMINI_API_KEY="your-key"
```

Done! Your next run will use Vertex AI automatically. 🚀

