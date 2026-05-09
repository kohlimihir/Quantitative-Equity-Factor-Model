# Cloud Deployment Guide

Deploy your model with automated monthly updates for **$0/month**.

## Architecture

```
GitHub (code) → GitHub Actions (runs model monthly)
                       ↓
                Azure Blob Storage (stores results)
                       ↓
                Azure App Service (API - free tier)
                       ↓
                Streamlit Cloud (dashboard - free)
```

## Prerequisites

- GitHub account
- Azure account (free tier)
- Git installed

## Step 1: Push to GitHub (5 min)

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/equity-factor-model.git
git push -u origin main
```

## Step 2: Setup Azure Storage (5 min)

```bash
# Login
az login

# Create storage
az group create --name equity-factor-rg --location eastus

az storage account create \
  --name equityfactorstorage123 \
  --resource-group equity-factor-rg \
  --sku Standard_LRS

# Get connection string (save this!)
az storage account show-connection-string \
  --name equityfactorstorage123 \
  --resource-group equity-factor-rg
```

## Step 3: Configure GitHub Actions (2 min)

1. Go to your GitHub repo
2. Settings → Secrets → Actions
3. Add secret: `AZURE_STORAGE_CONNECTION_STRING`
4. Paste your connection string from Step 2

GitHub Actions will now run automatically on:
- Every push to main
- 1st of each month at midnight
- Manual trigger anytime

## Step 4: Deploy API (10 min)

```bash
# Create app
az webapp up \
  --name equity-factor-api \
  --resource-group equity-factor-rg \
  --runtime "PYTHON:3.10" \
  --sku F1

# Get storage key
STORAGE_KEY=$(az storage account keys list \
  --account-name equityfactorstorage123 \
  --resource-group equity-factor-rg \
  --query '[0].value' -o tsv)

# Configure app
az webapp config appsettings set \
  --name equity-factor-api \
  --resource-group equity-factor-rg \
  --settings \
    AZURE_STORAGE_ACCOUNT="equityfactorstorage123" \
    AZURE_STORAGE_KEY="$STORAGE_KEY"
```

API URL: https://equity-factor-api.azurewebsites.net

Test: `curl https://equity-factor-api.azurewebsites.net/health`

## Step 5: Deploy Dashboard (5 min)

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: `app_cloud.py`
6. Advanced settings → Secrets:
   ```
   API_URL = "https://equity-factor-api.azurewebsites.net"
   ```
7. Click "Deploy"

Dashboard URL: https://YOUR_USERNAME-equity-factor-model.streamlit.app

## Verification

### Test API
```bash
curl https://equity-factor-api.azurewebsites.net/health
curl https://equity-factor-api.azurewebsites.net/top-stocks?n=10
```

### Test Dashboard
Open your Streamlit URL and verify:
- Data loads from API
- All tabs work
- Charts display correctly

### Test Automation
1. Go to GitHub repo → Actions tab
2. Click "Run workflow" to trigger manually
3. Wait ~15 minutes
4. Check Azure Blob Storage for new files
5. Refresh dashboard to see updated data

## Monthly Updates

On the 1st of each month:
1. GitHub Actions triggers automatically
2. Runs `python run_all.py --config optimized_low_turnover`
3. Uploads results to Azure Blob Storage
4. API reads new data automatically
5. Dashboard shows updated predictions

No manual intervention needed!

## Cost Breakdown

| Service | Free Tier | Your Usage | Cost |
|---------|-----------|------------|------|
| GitHub Actions | 2000 min/month | ~30 min | $0 |
| Azure Blob Storage | 5 GB | ~500 MB | $0 |
| Azure App Service F1 | 60 CPU min/day | ~10 min/day | $0 |
| Streamlit Cloud | 1 GB RAM | ~500 MB | $0 |
| **Total** | | | **$0/month** |

## Troubleshooting

### GitHub Actions Fails
- Check secrets are set correctly
- View logs in Actions tab
- Verify data files aren't too large

### API Returns 404
- Check if files uploaded to Azure Blob
- Run GitHub Actions manually
- Verify storage credentials

### Dashboard Can't Connect
- Check API URL in Streamlit secrets
- Test API health endpoint
- Verify CORS is enabled

### App Service Sleeping
- Normal for F1 tier (free)
- First request takes 30 seconds (wakes up)
- Subsequent requests are fast

## API Endpoints

Base URL: `https://equity-factor-api.azurewebsites.net`

- `GET /` - API info
- `GET /health` - Health check
- `GET /predictions` - Get predictions
- `GET /predictions/{ticker}` - Ticker predictions
- `GET /top-stocks?n=20` - Top N stocks
- `GET /performance` - Performance metrics
- `GET /features` - Feature importance
- `GET /docs` - Interactive API documentation

## Files Used for Deployment

- `.github/workflows/run-model.yml` - GitHub Actions workflow
- `upload_to_azure.py` - Uploads results to Azure
- `api_azure.py` - FastAPI service for Azure
- `app_cloud.py` - Streamlit dashboard for cloud
- `requirements.txt` - Python dependencies

## Interview Talking Points

**DevOps**: "I set up CI/CD with GitHub Actions for automated monthly model runs."

**Cloud Architecture**: "Deployed on Azure free tier with Blob Storage, App Service, and Streamlit Cloud for $0/month."

**API Design**: "Built RESTful API with FastAPI that serves predictions from cloud storage."

**Automation**: "Entire workflow is automated - from training to deployment to updates."

## Upgrade Options

If you need more than free tier:

**Azure App Service**:
- F1 (free) → B1 ($13/month) for 24/7 uptime
- Add custom domain and SSL

**Monitoring**:
- Add Azure Application Insights
- Set up email alerts
- Track usage metrics

**Features**:
- Add authentication
- Enable caching
- Add rate limiting

## Summary

You now have:
- ✅ Automated monthly model runs
- ✅ Cloud storage for results
- ✅ REST API for predictions
- ✅ Interactive dashboard
- ✅ **Total cost: $0/month**

Perfect for interviews and portfolio projects!
