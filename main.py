import torch
from torchvision import models, transforms
import torch.nn as nn
import os
import re
import requests
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Body
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from google.cloud import firestore
from pyzbar import pyzbar
from datetime import datetime, timedelta
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials as firebase_credentials, storage as firebase_storage

# Chatbot imports
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel
from typing import List, Dict, Any

# Environment variables
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# --- Device API Key Utility ---
import secrets
import string

# =============================================================================
# CHATBOT DATA MODELS
# =============================================================================

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    user_id: str
    timestamp: datetime

class InventoryItem(BaseModel):
    id: str
    name: str
    quantity: int
    unit: str
    expiry_date: str = None
    category: str = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_food_model(model_path, num_classes=36, device='cuda'):
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

food_model = load_food_model(os.path.join(os.path.dirname(__file__), "Models_01", "img_classifier_weight2.pth"), 36, device)

# Load Fruits and Vegetables class labels
def load_food_labels():
    labels_path = os.path.join(os.path.dirname(__file__), "labels.txt")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    # Fallback to classes.txt and convert
    classes_path = os.path.join(os.path.dirname(__file__), "classes.txt")
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    return [name.replace("_", " ").title() for name in class_names]

FOOD_LABEL_NAMES = load_food_labels()

def predict_food(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # EfficientNet V2 S typically uses 384x384 input size
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    food_model.eval()
    with torch.no_grad():
        outputs = food_model(image)
        _, predicted = torch.max(outputs, 1)
        return FOOD_LABEL_NAMES[predicted.item()]


def generate_pi_api_key() -> str:
    """Generate a cryptographically secure API key for Pi devices"""
    alphabet = string.ascii_letters + string.digits
    key = ''.join(secrets.choice(alphabet) for _ in range(32))
    return f"pk_live_{key}"

# --- User Auth: get_current_user_uid ---
def get_current_user_uid(Authorization: str | None = Header(default=None)) -> str:
    """Verify ID token and return the user's UID (used as Firestore document id)."""
    if not Authorization or not Authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = Authorization.split(" ", 1)[1]
    
    try:
        decoded = firebase_auth.verify_id_token(token)
        uid = decoded.get("uid")
        if not uid:
            raise ValueError("No UID in token")
        return uid
    except ValueError as ve:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(ve)}")
    except Exception as e:
        error_msg = str(e)
        if "expired" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Token expired. Please get a new token from your client app.")
        elif "project" in error_msg.lower() or "project_id" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Token project mismatch. Token may be from a different Firebase project.")
        elif "signature" in error_msg.lower() or "invalid" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid token signature. Please ensure you're using a Firebase ID token from your client app.")
        else:
            raise HTTPException(status_code=401, detail=f"Invalid token: {error_msg}")

# --- Device Authentication ---
def get_user_from_auth(Authorization: str | None = Header(default=None)) -> str:
    """Authenticate using either Bearer token (user) or Pi-Key (device)"""
    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if Authorization.lower().startswith("bearer "):
        return get_current_user_uid(Authorization)
    elif Authorization.lower().startswith("pi-key "):
        api_key = Authorization.split(" ", 1)[1]
        # Query Firestore for device with this API key
        if db is None:
            raise HTTPException(status_code=401, detail="Database not available")
        device_query = db.collection('pi_devices').where('api_key', '==', api_key).where('status', '==', 'active').stream()
        device = next(device_query, None)
        if not device:
            raise HTTPException(status_code=401, detail="Invalid or inactive Pi API key")
        device_data = device.to_dict()
        # Optionally update last_used timestamp
        try:
            device.reference.update({'last_used': datetime.now().isoformat()})
        except Exception:
            pass
        return device_data['user_id']
    else:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")


app = FastAPI(
    title="Food Identification API",
    description="API for identifying food items and expiry dates from images",
    version="1.0.0"
)


# --- Pi Device Registration Endpoint ---

@app.post("/pi/register")
async def register_pi_device(
    device_name: str = Body(...),
    device_location: str = Body(...),
    device_id: str = Body(...),
    user_key: str = Depends(get_current_user_uid)
):
    """
    Register a Pi device for the authenticated user and return a unique API key.
    """
    api_key = generate_pi_api_key()
    device_doc = {
        'api_key': api_key,
        'user_id': user_key,
        'device_name': device_name,
        'device_location': device_location,
        'device_id': device_id,
        'registered_at': datetime.now().isoformat(),
        'last_used': None,
        'status': 'active'
    }
    db.collection('pi_devices').add(device_doc)
    return {'success': True, 'api_key': api_key, 'message': 'Device registered successfully'}

# Add security scheme to OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add security requirement to all endpoints
    openapi_schema["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Initialize Firestore client
db = None
try:
    # Try multiple methods to get credentials
    google_creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    google_creds_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    creds = None
    creds_info = None
    
    # Method 1: Check for JSON string in environment variable
    if google_creds_json:
        import json
        creds_info = json.loads(google_creds_json)
        creds = service_account.Credentials.from_service_account_info(creds_info)
    # Method 2: Check for file path in environment variable
    elif google_creds_file and os.path.exists(google_creds_file):
        import json
        with open(google_creds_file, 'r', encoding='utf-8') as f:
            creds_info = json.load(f)
        creds = service_account.Credentials.from_service_account_file(google_creds_file)
    # Method 3: Check for default credentials file in project directory
    else:
        default_creds_file = os.path.join(os.path.dirname(__file__), "smiling-gasket-468408-u8-868e68126259.json")
        if os.path.exists(default_creds_file):
            import json
            with open(default_creds_file, 'r', encoding='utf-8') as f:
                creds_info = json.load(f)
            creds = service_account.Credentials.from_service_account_file(default_creds_file)
    
    if creds:
        db = firestore.Client(credentials=creds, project=creds_info.get('project_id'))
    else:
        # Try to use Application Default Credentials (recommended for Cloud Run)
        try:
            db = firestore.Client()
        except Exception:
            pass  # Client will be None if initialization fails
        
except Exception:
    pass  # Client will be None if initialization fails

# Initialize Firebase Admin for verifying ID tokens
try:
    if not firebase_admin._apps:
        firebase_storage_bucket = 'smiling-gasket-468408-u8.firebasestorage.app'
        firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS_JSON')
        firebase_creds_file = os.environ.get('FIREBASE_CREDENTIALS')
        
        cred = None
        
        # Method 1: Check for JSON string in environment variable
        if firebase_creds_json:
            import json
            creds_info = json.loads(firebase_creds_json)
            cred = firebase_credentials.Certificate(creds_info)
        # Method 2: Check for file path in environment variable
        elif firebase_creds_file and os.path.exists(firebase_creds_file):
            cred = firebase_credentials.Certificate(firebase_creds_file)
        # Method 3: Use the same Google Cloud credentials file (they're often the same)
        else:
            default_creds_file = os.path.join(os.path.dirname(__file__), "smiling-gasket-468408-u8-868e68126259.json")
            if os.path.exists(default_creds_file):
                cred = firebase_credentials.Certificate(default_creds_file)
        
        if cred:
            firebase_admin.initialize_app(cred, {
                'storageBucket': firebase_storage_bucket
            })
        else:
            # Try to use Application Default Credentials
            try:
                firebase_admin.initialize_app(options={
                    'storageBucket': firebase_storage_bucket
                })
            except Exception:
                pass
except Exception:
    pass

def save_inventory_item(item: dict, user_key: str | None = None) -> None:
    """Persist an inventory item to Firestore in collection 'inventory'."""
    if db is None:
        return
    try:
        if user_key:
            db.collection('users').document(user_key).collection('inventory').add(item)
        else:
            db.collection('inventory').add(item)
    except Exception as e:
        pass  # Silently fail to avoid breaking API response

def save_image_to_firestore(image_data: bytes, user_key: str, item_id: str = None) -> str:
    """Save image to Firebase Storage and return the image URL."""
    try:
        # Get image format and size
        try:
            img = Image.open(io.BytesIO(image_data))
            image_format = img.format or 'JPEG'
            image_size = len(image_data)
            width, height = img.size
        except Exception:
            image_format = 'UNKNOWN'
            image_size = len(image_data)
            width, height = 0, 0
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"image_{timestamp}_{user_key[:8]}.{image_format.lower()}"
        
        # Upload to Firebase Storage
        bucket_name = 'smiling-gasket-468408-u8.firebasestorage.app'
        bucket = firebase_storage.bucket(bucket_name)
        blob = bucket.blob(f"images/{user_key}/{filename}")
        blob.upload_from_string(image_data, content_type=f"image/{image_format.lower()}")
        blob.make_public()
        image_url = blob.public_url
        
        # Store metadata in Firestore
        image_doc = {
            'image_url': image_url,
            'storage_path': f"images/{user_key}/{filename}",
            'format': image_format,
            'size_bytes': image_size,
            'width': width,
            'height': height,
            'created_at': datetime.now().isoformat(),
            'user_key': user_key,
            'filename': filename
        }
        
        if item_id:
            image_doc['item_id'] = item_id
        
        doc_ref = db.collection('images').add(image_doc)
        image_id = doc_ref[1].id
        db.collection('images').document(image_id).update({'id': image_id})
        
        return image_id
        
    except Exception as e:
        return None

# Food expiry days mapping
FOOD_EXPIRY_DAYS = {
    # Fruits
    "apple": 30, "banana": 7, "grapes": 7, "guava": 5, "lemon": 21, "mango": 7,
    "orange": 20, "papaya": 5, "pineapple": 5, "pomegranate": 30, "watermelon": 7,
    # Vegetables
    "potato": 30, "tomato": 7, "onion": 30, "carrot": 15, "lettuce": 7, "cabbage": 10,
    "spinach": 5, "cauliflower": 10, "capsicum": 10, "garlic": 30, "ginger": 21,
    "peas": 7, "corn": 7, "radish": 10, "cucumber": 7, "eggplant": 5, "soy beans": 7,
    "sweet potato": 30, "turnip": 14, "pea": 7, "beans": 7,
    # Peppers and others
    "bell pepper": 10, "bitter gourd": 5, "chilli pepper": 14, "chili pepper": 14,
    "pepper": 10,
}



@app.post("/route/")
async def route_image(image: UploadFile = File(...), user_key: str = Depends(get_user_from_auth)):
    """
    Identify food items or barcode. If the image is invalid, return invalid.
    Returns: { "food_name": string, "expiry_date": YYYY-MM-DD } or { "message": "invalid" }
    """
    try:
        contents = await image.read()
        if not contents:
            return JSONResponse(status_code=400, content={"message": "No image data received. Please upload a valid image file."})
        mime_type = image.content_type or "image/jpeg"

        image_id = save_image_to_firestore(contents, user_key)

        # Check for barcode using pyzbar (free, open-source library)
        # Try multiple preprocessing techniques for robust detection
        barcode_detected = False
        barcode_value = None
        
        try:
            # Open image with PIL
            pil_image = Image.open(io.BytesIO(contents))
            
            # Try multiple preprocessing techniques to improve barcode detection
            image_variants = []
            
            # 1. Original RGB image
            if pil_image.mode != 'RGB':
                rgb_image = pil_image.convert('RGB')
            else:
                rgb_image = pil_image.copy()
            image_variants.append(('original', rgb_image))
            
            # 2. Grayscale (often better for barcode detection)
            gray_image = rgb_image.convert('L')
            image_variants.append(('grayscale', gray_image))
            
            # 3. Enhanced contrast (multiple levels)
            from PIL import ImageEnhance
            for contrast_level in [1.5, 2.0, 2.5]:
                enhancer = ImageEnhance.Contrast(rgb_image)
                high_contrast = enhancer.enhance(contrast_level)
                image_variants.append((f'contrast_{contrast_level}', high_contrast))
            
            # 4. Sharpened image (multiple levels)
            for sharpness_level in [1.5, 2.0, 2.5]:
                enhancer = ImageEnhance.Sharpness(rgb_image)
                sharpened = enhancer.enhance(sharpness_level)
                image_variants.append((f'sharp_{sharpness_level}', sharpened))
            
            # 5. Brightness adjusted (barcode might be too dark or too bright)
            for brightness_level in [0.7, 0.8, 1.2, 1.3]:
                enhancer = ImageEnhance.Brightness(rgb_image)
                brightened = enhancer.enhance(brightness_level)
                image_variants.append((f'brightness_{brightness_level}', brightened))
            
            # 6. Resized larger (if image is small, barcodes might be hard to detect)
            if rgb_image.width < 800 or rgb_image.height < 800:
                larger = rgb_image.resize((rgb_image.width * 2, rgb_image.height * 2), Image.Resampling.LANCZOS)
                image_variants.append(('resized_larger', larger))
                # Also try grayscale of larger image
                larger_gray = larger.convert('L')
                image_variants.append(('resized_larger_gray', larger_gray))
            
            # 7. Rotated versions (barcode might be at an angle)
            for angle in [90, 180, 270]:
                rotated = rgb_image.rotate(angle, expand=True)
                image_variants.append((f'rotated_{angle}', rotated))
                # Also try grayscale rotated
                rotated_gray = rotated.convert('L')
                image_variants.append((f'rotated_{angle}_gray', rotated_gray))
            
            # 8. Try with different resampling methods for better quality
            if rgb_image.width < 1200 or rgb_image.height < 1200:
                # Upscale with high quality
                upscaled = rgb_image.resize((rgb_image.width * 3, rgb_image.height * 3), Image.Resampling.LANCZOS)
                image_variants.append(('upscaled_3x', upscaled))
                upscaled_gray = upscaled.convert('L')
                image_variants.append(('upscaled_3x_gray', upscaled_gray))
            
            # Try detecting barcode in all image variants
            for variant_name, variant_image in image_variants:
                try:
                    # Detect barcodes using pyzbar
                    barcodes = pyzbar.decode(variant_image)
                    
                    if barcodes and len(barcodes) > 0:
                        # Get the first barcode value
                        try:
                            barcode_value = barcodes[0].data.decode('utf-8')
                        except UnicodeDecodeError:
                            # Try alternative decoding
                            barcode_value = barcodes[0].data.decode('latin-1')
                        
                        if barcode_value and len(barcode_value) >= 8:  # Valid barcodes are usually at least 8 digits
                            barcode_detected = True
                            break  # Found barcode, no need to try other variants
                except Exception:
                    continue  # Try next variant
            
            # If barcode detected, look up product information
            if barcode_detected and barcode_value:
                result = await process_barcode_from_value(barcode_value, contents, mime_type, user_key, image_id)
                
                # If lookup succeeded, return result
                if result and result.get("food_name"):
                    return result
                
                # If barcode detected but lookup failed, return invalid
                return JSONResponse(status_code=400, content={"message": "invalid"})
            
            # If barcode detection was attempted but failed, return invalid
            # This prevents misclassification (e.g., barcode image being classified as "kiwi")
            # Instead of falling back to food model, return invalid
            return JSONResponse(status_code=400, content={"message": "invalid"})
        except Exception:
            # If barcode detection fails due to error, return invalid
            return JSONResponse(status_code=400, content={"message": "invalid"})
    except HTTPException as he:
        return JSONResponse(status_code=he.status_code, content={"message": f"HTTP error: {he.detail}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing image: {str(e)}"})

@app.get("/users/me/inventory")
async def get_my_inventory(user_key: str = Depends(get_current_user_uid)):
    try:
        items = []
        docs = db.collection('users').document(user_key).collection('inventory').stream()
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            
            # If item has an image_id, fetch basic image info
            if 'image_id' in data and data['image_id']:
                try:
                    image_doc = db.collection('images').document(data['image_id']).get()
                    if image_doc.exists:
                        image_data = image_doc.to_dict()
                        data['image_info'] = {
                            'id': data['image_id'],
                            'format': image_data.get('format'),
                            'size_bytes': image_data.get('size_bytes'),
                            'width': image_data.get('width'),
                            'height': image_data.get('height'),
                            'filename': image_data.get('filename'),
                            'created_at': image_data.get('created_at'),
                            'image_url': image_data.get('image_url'),
                            'view_url': f"/users/me/images/{data['image_id']}/view",
                            'download_url': f"/users/me/images/{data['image_id']}/download"
                        }
                except Exception:
                    data['image_info'] = None
            
            items.append(data)
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/users/me/inventory/{item_id}")
async def delete_my_item(item_id: str, user_key: str = Depends(get_current_user_uid)):
    try:
        db.collection('users').document(user_key).collection('inventory').document(item_id).delete()
        return {"message": f"Item {item_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def parse_expiry_date(date_str: str) -> str | None:
    """Parse expiry date from various formats to YYYY-MM-DD"""
    if not date_str:
        return None
    try:
        from dateutil import parser as dateparser
        parsed = dateparser.parse(date_str, dayfirst=False, yearfirst=True)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        # If dateutil not available or parse fails, accept as-is if looks like YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str
        return None

def lookup_openfoodfacts(barcode: str) -> dict | None:
    """Look up product on OpenFoodFacts"""
    try:
        api_url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        r = requests.get(api_url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if data.get("status") != 1:
            return None
        
        product = data.get("product", {})
        product_name = product.get("product_name") or product.get("generic_name") or None
        
        # Clean product name
        if product_name:
            product_name = product_name.strip()
        
        expiry = product.get("expiration_date") or product.get("best_before_date") or product.get("minimum_durability_date")
        
        if product_name:
            return {
                "name": product_name,
                "expiry": expiry,
                "source": "openfoodfacts"
            }
        return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None

def lookup_upcitemdb(barcode: str) -> dict | None:
    """Look up product on UPCitemdb"""
    try:
        api_url = f"https://api.upcitemdb.com/prod/trial/lookup"
        r = requests.get(api_url, params={"upc": barcode}, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if data.get("code") != "OK" or not data.get("items"):
            return None
        
        item = data["items"][0]
        product_name = item.get("title") or item.get("description") or None
        
        # Clean product name
        if product_name:
            product_name = product_name.strip()
        
        if product_name:
            return {
                "name": product_name,
                "expiry": None,  # UPCitemdb doesn't provide expiry dates
                "source": "upcitemdb"
            }
        return None
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None


async def process_barcode_from_value(barcode_value: str, contents: bytes, mime_type: str = "image/jpeg", user_key: str | None = None, image_id: str | None = None):
    """
    Process barcode value by querying multiple free food APIs (OpenFoodFacts, UPCitemdb).
    OpenFoodFacts has excellent coverage of Indian food products.
    Returns { "food_name": product_name, "expiry_date": YYYY-MM-DD } or None if not found.
    """
    # Try multiple APIs
    results = []
    
    # API 1: OpenFoodFacts (best for food products, excellent Indian product coverage)
    result1 = lookup_openfoodfacts(barcode_value)
    if result1:
        results.append(result1)
    
    # API 2: UPCitemdb (free backup, general products)
    result2 = lookup_upcitemdb(barcode_value)
    if result2:
        results.append(result2)
    
    # If no results from any API, return None (caller will handle this)
    if not results:
        return None
    
    # Prioritize results: prefer OpenFoodFacts (has expiry dates and Indian products)
    # Sort by: has expiry date first, then by source priority
    results.sort(key=lambda x: (
        x["expiry"] is None,  # False (has expiry) comes before True (no expiry)
        x["source"] != "openfoodfacts"  # OpenFoodFacts first (best for food/Indian products)
    ))
    
    # Get best result
    best_result = results[0]
    product_name = best_result["name"]
    expiry_str = best_result.get("expiry")
    
    # Parse expiry date
    parsed_expiry = parse_expiry_date(expiry_str) if expiry_str else None
    
    # Clean product name
    if not product_name or product_name.lower() in ["unknown", "unknown product", ""]:
        product_name = "Unknown Product"
    
    # Store result
    try:
        save_inventory_item({
            "food_name": product_name,
            "expiry_date": parsed_expiry,
            "source": "barcode",
            "barcode": barcode_value,
            "image_id": image_id,
            "quantity": 1,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }, user_key)
    except Exception:
        pass
    
    return {
        "food_name": product_name,
        "expiry_date": parsed_expiry
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Food identification service is running"}

@app.get("/debug/barcode")
async def debug_barcode():
    """Debug endpoint to check barcode detection library"""
    try:
        from pyzbar import pyzbar
        return {
            "barcode_library": "pyzbar",
            "library_available": True,
            "db_available": db is not None
        }
    except ImportError:
        return {
            "barcode_library": "pyzbar",
            "library_available": False,
            "error": "pyzbar not installed. Install with: pip install pyzbar",
            "db_available": db is not None
        }


@app.get("/users/me/images")
async def get_my_images(user_key: str = Depends(get_current_user_uid)):
    """Get all images uploaded by the current user"""
    try:
        images = []
        docs = db.collection('images').where('user_key', '==', user_key).stream()
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            # Now we return the actual image URL instead of base64 data
            images.append(data)
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/users/me/images/{image_id}")
async def get_my_image(image_id: str, user_key: str = Depends(get_current_user_uid)):
    """Get a specific image by ID for the current user"""
    try:
        doc = db.collection('images').document(image_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Image not found")
        
        data = doc.to_dict()
        if data.get('user_key') != user_key:
            raise HTTPException(status_code=403, detail="Access denied")
        
        data['id'] = doc.id
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/users/me/images/{image_id}")
async def delete_my_image(image_id: str, user_key: str = Depends(get_current_user_uid)):
    """Delete a specific image by ID for the current user"""
    try:
        # Check if user owns the image
        doc = db.collection('images').document(image_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Image not found")
        
        data = doc.to_dict()
        if data.get('user_key') != user_key:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete the image
        db.collection('images').document(image_id).delete()
        return {"message": f"Image {image_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/users/me/images/{image_id}/download")
async def download_my_image(image_id: str, user_key: str = Depends(get_current_user_uid)):
    """Redirect to the Firebase Storage URL for the image"""
    try:
        doc = db.collection('images').document(image_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Image not found")
        
        data = doc.to_dict()
        if data.get('user_key') != user_key:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Redirect to the Firebase Storage URL
        image_url = data.get('image_url')
        if not image_url:
            raise HTTPException(status_code=404, detail="Image URL not found")
        
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=image_url)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/users/me/images/{image_id}/preview")
async def preview_my_image(image_id: str, user_key: str = Depends(get_current_user_uid)):
    """Get image metadata and preview information"""
    try:
        doc = db.collection('images').document(image_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Image not found")
        
        data = doc.to_dict()
        if data.get('user_key') != user_key:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Return image metadata and URLs
        preview_data = {
            'id': doc.id,
            'format': data.get('format'),
            'size_bytes': data.get('size_bytes'),
            'width': data.get('width'),
            'height': data.get('height'),
            'filename': data.get('filename'),
            'created_at': data.get('created_at'),
            'image_url': data.get('image_url'),
            'storage_path': data.get('storage_path'),
            'view_url': f"/users/me/images/{image_id}/view"
        }
        
        return preview_data
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/users/me/images/{image_id}/view")
async def view_image_html(image_id: str, user_key: str = Depends(get_current_user_uid)):
    """Return HTML page to view the image from Firebase Storage"""
    try:
        doc = db.collection('images').document(image_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Image not found")
        
        data = doc.to_dict()
        if data.get('user_key') != user_key:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create HTML page with embedded image from Firebase Storage
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Viewer - {data.get('filename', 'Unknown')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .image-info {{ margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
                .image-info p {{ margin: 5px 0; }}
                .image-container {{ text-align: center; }}
                img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }}
                .download-link {{ display: inline-block; margin-top: 15px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
                .download-link:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Image Viewer</h1>
                <div class="image-info">
                    <h3>Image Details</h3>
                    <p><strong>Filename:</strong> {data.get('filename', 'Unknown')}</p>
                    <p><strong>Format:</strong> {data.get('format', 'Unknown')}</p>
                    <p><strong>Size:</strong> {data.get('size_bytes', 0)} bytes</p>
                    <p><strong>Dimensions:</strong> {data.get('width', 0)} x {data.get('height', 0)} pixels</p>
                    <p><strong>Uploaded:</strong> {data.get('created_at', 'Unknown')}</p>
                </div>
                <div class="image-container">
                    <img src="{data.get('image_url', '')}" alt="{data.get('filename', 'Image')}" />
                    <br>
                    <a href="{data.get('image_url', '')}" class="download-link" download="{data.get('filename', 'image')}">Download Image</a>
                </div>
            </div>
        </body>
        </html>
        """
        
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- List Pi Devices Endpoint ---
@app.get("/pi/list")
async def list_pi_devices(user_key: str = Depends(get_current_user_uid)):
    """
    List all Pi devices registered for the authenticated user.
    """
    try:
        devices = db.collection('pi_devices').where('user_id', '==', user_key).stream()
        device_list = []
        for device in devices:
            data = device.to_dict()
            data['id'] = device.id
            device_list.append(data)
        return {"devices": device_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
# --- Pi Device Confirm Registration Endpoint ---

@app.post("/pi/confirm-registration")
async def confirm_pi_registration(api_key: str = Body(...)):
    """
    Pi device confirms registration by sending its API key. Server verifies and marks device as confirmed.
    """
    try:
        device_query = db.collection('pi_devices').where('api_key', '==', api_key).stream()
        device = next(device_query, None)
        if not device:
            raise HTTPException(status_code=404, detail="API key not found")
        device.reference.update({'confirmed': True, 'confirmation_time': datetime.now().isoformat()})
        return {"success": True, "message": "Registration complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
@app.post("/pi/status")
async def pi_registration_status(device_id: str = Body(...), user_key: str = Depends(get_current_user_uid)):
    """
    Check if a Pi device is registered/confirmed for the authenticated user.
    """
    try:
        device_query = db.collection('pi_devices').where('user_id', '==', user_key).where('device_id', '==', device_id).stream()
        device = next(device_query, None)
        if not device:
            return {"registered": False}
        data = device.to_dict()
        registered = bool(data.get('confirmed'))
        return {"registered": registered, "device": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# --- Pi Device Delete Endpoint ---
@app.post("/pi/delete")
async def delete_pi_device(device_id: str = Body(...), user_key: str = Depends(get_current_user_uid)):
    """
    Delete a Pi device for the authenticated user by device_id.
    """
    try:
        device_query = db.collection('pi_devices').where('user_id', '==', user_key).where('device_id', '==', device_id).stream()
        device = next(device_query, None)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        device.reference.delete()
        return {"success": True, "message": "Device deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# =============================================================================
# CHATBOT CLASS
# =============================================================================

class SimpleChatbot:
    def __init__(self):
        # Get API key from environment
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
            self.llm = None
        else:
            models_to_try = [
                "llama-3.1-8b-instant",
                "llama-3.2-3b-preview", 
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768"
            ]
            
            self.llm = None
            for model in models_to_try:
                try:
                    self.llm = ChatGroq(
                        groq_api_key=GROQ_API_KEY,
                        model_name=model,
                        temperature=0.7
                    )
                    break
                except Exception:
                    continue
    
    def get_user_inventory(self, user_id: str) -> List[Dict[str, Any]]:
        """Get available ingredients from user's fridge with quantity information"""
        try:
            inventory_ref = db.collection("users").document(user_id).collection("inventory")
            docs = inventory_ref.stream()
            
            ingredients = []
            for doc in docs:
                data = doc.to_dict()
                item_name = data.get('food_name', data.get('name', 'Unknown'))
                quantity = data.get('quantity', 0)
                unit = data.get('unit', '')
                status = data.get('status', 'active')
                
                if status == 'active' and quantity > 0:
                    ingredients.append({
                        'name': item_name.lower(),
                        'quantity': quantity,
                        'unit': unit,
                        'display_name': item_name
                    })
            
            return ingredients
        except Exception:
            return []
    
    def get_recipe_recommendations(self, ingredients: List[Dict[str, Any]], message: str) -> str:
        """Get recipe recommendations using AI"""
        try:
            if not self.llm:
                return self._fallback_recipe_response(ingredients, message)
                
            # Format ingredients with quantities
            if ingredients:
                ingredients_text = ", ".join([
                    f"{ing['display_name']} ({ing['quantity']} {ing['unit']})" 
                    for ing in ingredients
                ])
            else:
                ingredients_text = "No ingredients available"
            
            prompt = f"""You are a helpful cooking assistant. The user has these ingredients: {ingredients_text}
            
            User request: {message}
            
            Suggest 2-3 recipes they can make. If ingredients are missing, suggest substitutes.
            Be helpful and encouraging! Include the quantities when mentioning ingredients.
            
            Format your response nicely with emojis and clear instructions."""
            
            messages = [
                SystemMessage(content="You are a helpful cooking assistant specialized in recipe recommendations."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception:
            return self._fallback_recipe_response(ingredients, message)
    
    def _fallback_recipe_response(self, ingredients: List[Dict[str, Any]], message: str) -> str:
        """Fallback response when AI model is not available"""
        if not ingredients:
            return "🥺 Your fridge is empty! Time to go shopping for some ingredients."
        
        ingredients_list = "\n".join([
            f"• {ing['display_name']} ({ing['quantity']} {ing['unit']})" 
            for ing in ingredients
        ])
        return f"""🍳 Here's what you can make with your ingredients:

{ingredients_list}

💡 **Simple Recipe Ideas:**
• **Stir-fry**: Use any vegetables you have with some oil and seasonings
• **Pasta**: Cook pasta and add your ingredients for a quick meal
• **Salad**: Mix fresh ingredients together with dressing
• **Soup**: Combine ingredients in a pot with broth or water

Get creative and experiment! 🎨"""
    
    def get_inventory_info(self, ingredients: List[Dict[str, Any]], message: str) -> str:
        """Get inventory information using AI for smart item detection"""
        if not ingredients:
            return "Your fridge is empty! 🥺 Time to go shopping!"
        
        try:
            if not self.llm:
                return self._fallback_inventory_response(ingredients, message)
                
            # Format ingredients with quantities for AI
            ingredients_text = ", ".join([
                f"{ing['display_name']} ({ing['quantity']} {ing['unit']})" 
                for ing in ingredients
            ])
                
            # Use AI to understand what they're asking about
            prompt = f"""The user is asking about their fridge inventory. 
            
            Available ingredients: {ingredients_text}
            User question: {message}
            
            If they're asking about a specific item, tell them if they have it or not and include the quantity.
            If they want a general list, show them what they have with quantities.
            Be helpful and use emojis!
            
            Examples:
            - "Is there milk?" → "Yes! ✅ You have milk (2 cups) in your fridge!"
            - "Do I have chicken?" → "No chicken in your fridge right now. ❌"
            - "What's in my fridge?" → List all ingredients with quantities and emojis"""
            
            messages = [
                SystemMessage(content="You are a helpful fridge assistant. Answer questions about what's in the fridge."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception:
            return self._fallback_inventory_response(ingredients, message)
    
    def _fallback_inventory_response(self, ingredients: List[Dict[str, Any]], message: str) -> str:
        """Fallback response for inventory questions when AI model is not available"""
        message_lower = message.lower()
        
        # Check if asking about specific item
        for ingredient in ingredients:
            if ingredient['name'].lower() in message_lower or ingredient['display_name'].lower() in message_lower:
                return f"✅ Yes! You have {ingredient['display_name']} ({ingredient['quantity']} {ingredient['unit']}) in your fridge!"
        
        # Check for common "what's in my fridge" phrases
        if any(phrase in message_lower for phrase in ["what's in", "what do i have", "show me", "list"]):
            ingredients_list = "\n".join([
                f"• {ing['display_name']} ({ing['quantity']} {ing['unit']})" 
                for ing in ingredients
            ])
            return f"Here's what's in your fridge: 🥘\n\n{ingredients_list}\n\nTotal items: {len(ingredients)}"
        
        # Default response
        ingredients_list = "\n".join([
            f"• {ing['display_name']} ({ing['quantity']} {ing['unit']})" 
            for ing in ingredients
        ])
        return f"Here's what's in your fridge: 🥘\n\n{ingredients_list}\n\nTotal items: {len(ingredients)}"
    
    def chat(self, user_id: str, message: str) -> str:
        """Main chat function"""
        # Get user's ingredients
        ingredients = self.get_user_inventory(user_id)
        
        # Check if asking for recipes
        recipe_keywords = ["recipe", "cook", "make", "prepare", "dish", "meal", "what can i make"]
        is_recipe_request = any(keyword in message.lower() for keyword in recipe_keywords)
        
        # Check if asking about inventory
        inventory_keywords = ["inventory", "fridge", "ingredients", "what's in", "what do i have", 
                            "is there", "show me my", "list my", "do i have", "what about"]
        is_inventory_request = any(keyword in message.lower() for keyword in inventory_keywords)
        
        if is_recipe_request and ingredients:
            return self.get_recipe_recommendations(ingredients, message)
        elif is_inventory_request:
            return self.get_inventory_info(ingredients, message)
        else:
            # General chat
            try:
                if not self.llm:
                    return "I'm having trouble connecting to the AI service right now. Please try again later."
                    
                # Format ingredients with quantities for general chat
                if ingredients:
                    ingredients_text = ", ".join([
                        f"{ing['display_name']} ({ing['quantity']} {ing['unit']})" 
                        for ing in ingredients
                    ])
                else:
                    ingredients_text = "None"
                    
                prompt = f"User message: {message}\n\nAvailable ingredients: {ingredients_text}\n\nRespond helpfully as a cooking assistant."
                messages = [
                    SystemMessage(content="You are a helpful cooking assistant."),
                    HumanMessage(content=prompt)
                ]
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                return f"I'm having trouble responding right now. Error: {str(e)}"

# Initialize chatbot
chatbot = SimpleChatbot()

# =============================================================================
# CHATBOT API ENDPOINTS
# =============================================================================

@app.post("/chatbot/chat", response_model=ChatResponse)
async def chatbot_chat(request: ChatRequest, user_key: str = Depends(get_current_user_uid)):
    """Chat with the AI assistant"""
    try:
        # Use the authenticated user's ID instead of the request user_id for security
        response = chatbot.chat(user_key, request.message)
        return ChatResponse(
            response=response,
            user_id=user_key,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chatbot/inventory")
async def chatbot_get_inventory(user_key: str = Depends(get_current_user_uid)):
    """Get user's fridge inventory for chatbot"""
    try:
        inventory_ref = db.collection("users").document(user_key).collection("inventory")
        docs = inventory_ref.stream()
        
        items = []
        for doc in docs:
            data = doc.to_dict()
            # Map the correct field names from React Native app structure
            item_name = data.get('food_name', data.get('name', ''))
            status = data.get('status', 'active')
            
            # Only include active items
            if status == 'active':
                items.append(InventoryItem(
                    id=doc.id,
                    name=item_name,
                    quantity=data.get("quantity", 0),
                    unit=data.get("unit", ""),
                    expiry_date=data.get("expiry_date"),
                    category=data.get("source", data.get("category", ""))
                ))
        
        return {"user_id": user_key, "items": items, "total_items": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# RUN THE SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
