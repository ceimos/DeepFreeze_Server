# FastAPI Food Identification API - Google Cloud Run Deployment

A FastAPI application that identifies food items and expiry dates from images using Google Cloud Vision API, with Firebase Authentication and Firestore storage.

## Features

- 🍎 **Food Recognition**: Identify food items from images using Google Cloud Vision API
- 📱 **Barcode Scanning**: Extract product information from barcodes using OpenFoodFacts API
- 🔐 **Firebase Authentication**: Secure user authentication with JWT tokens
- 📦 **Cloud Storage**: Store images in Firebase Storage
- 📊 **Inventory Management**: Track user inventory with expiry dates
- 🚀 **Cloud Run Ready**: Optimized for Google Cloud Run deployment

## Prerequisites

Before deploying to Google Cloud Run, ensure you have:

1. **Google Cloud Project** with billing enabled
2. **Google Cloud CLI** installed and configured
3. **Docker** installed on your local machine
4. **Firebase Project** set up with:
   - Authentication enabled
   - Firestore database created
   - Storage bucket created

## Quick Start

### Option 1: GitHub Actions Deployment (Recommended)

This is the easiest way to deploy your application with automatic CI/CD.

#### 1. Fork/Clone Repository
```bash
# Clone the repository
git clone <your-repo-url>
cd DeepFreeze_Server

# Push to your GitHub repository
git remote set-url origin https://github.com/your-username/your-repo-name.git
git push -u origin main
```

#### 2. Configure GitHub Secrets
Follow the detailed guide in [`GITHUB_SECRETS_SETUP.md`](./GITHUB_SECRETS_SETUP.md) to set up:

- `GOOGLE_CLOUD_PROJECT` - Your Google Cloud Project ID
- `GOOGLE_CLOUD_SA_KEY` - Service account JSON key
- `FIREBASE_STORAGE_BUCKET` - Firebase Storage bucket name

#### 3. Deploy Automatically
```bash
# Any push to main branch will trigger deployment
git add .
git commit -m "Deploy to Cloud Run"
git push origin main
```

The GitHub Action will:
- ✅ Run tests
- 🐳 Build Docker image
- 🚀 Deploy to Google Cloud Run
- 🔍 Run health checks
- 📋 Provide deployment summary

### Option 2: Manual Local Deployment

#### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd DeepFreeze_Server

# Copy environment configuration
copy .env.example .env
# Edit .env with your project configuration
```

#### 2. Configure Environment Variables
Edit `.env` file with your project settings:

```bash
# Required: Your Google Cloud Project ID
GOOGLE_CLOUD_PROJECT=your-project-id

# Required: Firebase Storage Bucket
FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com

# Optional: Service Name for Cloud Run
SERVICE_NAME=deepfreeze-api

# Optional: Deployment Region
REGION=us-central1
```

#### 3. Deploy Manually
##### Using PowerShell (Windows):
```powershell
# Make script executable and deploy
.\deploy.ps1 -ProjectId "your-project-id"
```

##### Using Bash (Linux/Mac):
```bash
# Make script executable
chmod +x deploy.sh

# Deploy
GOOGLE_CLOUD_PROJECT=your-project-id ./deploy.sh
```

### 4. Manual Deployment (Alternative)

If you prefer manual deployment without scripts:

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com vision.googleapis.com firestore.googleapis.com storage.googleapis.com

# Build and deploy
gcloud run deploy deepfreeze-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Deployment Status

After deployment (via GitHub Actions or manual), you can monitor:

- **GitHub Actions**: Check the "Actions" tab in your repository
- **Service URL**: Automatically provided in deployment summary
- **Health Check**: `https://your-service-url/health`
- **API Docs**: `https://your-service-url/docs`

## Continuous Deployment

With GitHub Actions configured:

- 🔄 **Automatic**: Push to `main` branch triggers deployment
- 🧪 **Testing**: Runs tests before deployment  
- 📊 **Monitoring**: Health checks and deployment summaries
- 🔔 **Notifications**: Success/failure notifications in GitHub


## API Endpoints

Once deployed, your API will be available at `https://YOUR_SERVICE_URL`:

### Core Endpoints

- **POST `/route/`** - Upload and analyze food images
  - Requires: `image` (file upload)
  - Headers: `Authorization: Bearer <firebase-jwt-token>`
  - Returns: Food identification and expiry date

- **GET `/health`** - Health check endpoint

- **GET `/docs`** - Interactive API documentation (Swagger UI)

### User Management

- **GET `/users/me/inventory`** - Get user's food inventory
- **DELETE `/users/me/inventory/{item_id}`** - Delete inventory item

### Image Management

- **GET `/users/me/images`** - List user's uploaded images
- **GET `/users/me/images/{image_id}`** - Get specific image metadata
- **DELETE `/users/me/images/{image_id}`** - Delete uploaded image

## Authentication

The API uses Firebase Authentication. Include the Firebase JWT token in requests:

```bash
curl -X POST "https://your-service-url/route/" \
  -H "Authorization: Bearer YOUR_FIREBASE_JWT_TOKEN" \
  -F "image=@your-food-image.jpg"
```

## GitHub Actions Workflow

The repository includes a comprehensive CI/CD pipeline (`.github/workflows/deploy.yml`) that:

### 🧪 **Testing Phase**
- Sets up Python 3.11 environment
- Installs dependencies with caching
- Runs import validation tests
- Validates FastAPI and Google Cloud imports

### 🚀 **Deployment Phase** (on main branch)
- Authenticates with Google Cloud using service account
- Builds optimized Docker image
- Pushes to Google Container Registry
- Deploys to Cloud Run with production configuration
- Runs health checks to verify deployment
- Provides comprehensive deployment summary

### 📊 **Monitoring & Notifications**
- Creates deployment summaries with service URLs
- Provides failure notifications with troubleshooting steps
- Generates workflow status reports

### Workflow Triggers
- **Push to main/master**: Full deployment
- **Pull Request**: Testing only
- **Manual**: Via GitHub Actions tab

## Configuration

### Environment Variables

The application supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | Google Cloud Project ID | Required |
| `FIREBASE_STORAGE_BUCKET` | Firebase Storage bucket name | Required |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | Service account JSON (for local dev) | Optional |
| `FIREBASE_CREDENTIALS_JSON` | Firebase credentials JSON | Optional |
| `ENVIRONMENT` | Environment name | `production` |
| `PORT` | Application port | `8080` |

### Google Cloud Services

The application uses these Google Cloud services:

- **Cloud Vision API** - For image analysis and OCR
- **Firestore** - For storing inventory and metadata
- **Firebase Storage** - For storing uploaded images
- **Firebase Authentication** - For user authentication

## Local Development

For local development:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up service account:
```bash
# Download service account key from Google Cloud Console
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

3. Run the application:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Docker

Build and run with Docker:

```bash
# Build image
docker build -t deepfreeze-api .

# Run container
docker run -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e FIREBASE_STORAGE_BUCKET=your-bucket \
  deepfreeze-api
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Ensure Firebase project is properly configured
   - Check that service account has required permissions
   - Verify JWT tokens are valid and not expired

2. **Vision API Errors**
   - Ensure Vision API is enabled in Google Cloud Console
   - Check that service account has Vision API permissions

3. **Storage Errors**
   - Verify Firebase Storage bucket exists
   - Ensure storage rules allow write access
   - Check that service account has Storage Admin permissions

### Debugging

Check application logs:

```bash
# View Cloud Run logs
gcloud logs read --service=deepfreeze-api --region=us-central1

# Follow logs in real-time
gcloud logs tail --service=deepfreeze-api --region=us-central1
```

## Security Considerations

1. **Service Account Permissions**: Use least-privilege principle
2. **Firebase Rules**: Configure proper Firestore and Storage security rules
3. **CORS**: Configure appropriate CORS origins for production
4. **Authentication**: Always validate JWT tokens server-side

## Cost Optimization

- **Cloud Run**: Scales to zero when not in use
- **Vision API**: Consider image size and compression
- **Storage**: Regular cleanup of old images
- **Firestore**: Optimize queries and indexes

## Support

For issues and questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Review Google Cloud and Firebase documentation
3. Check application logs for error details

## License

[Add your license information here]