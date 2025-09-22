import os
import re
import requests
import numpy as np  # unused if OCR-only path is used; kept for compatibility
import base64
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from google.cloud import vision
from google.cloud import firestore
from datetime import datetime, timedelta
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials as firebase_credentials, storage as firebase_storage

# --- Device API Key Utility ---
import secrets
import string

def generate_pi_api_key() -> str:
    """Generate a cryptographically secure API key for Pi devices"""
    alphabet = string.ascii_letters + string.digits
    key = ''.join(secrets.choice(alphabet) for _ in range(32))
    return f"pk_live_{key}"

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


# --- User Auth: get_current_user_uid ---
def get_current_user_uid(Authorization: str | None = Header(default=None)) -> str:
    """Verify ID token and return the user's UID (used as Firestore document id)."""
    print(f"Debug: Authorization header received: {Authorization}")
    if not Authorization or not Authorization.lower().startswith("bearer "):
        print(f"Debug: Invalid Authorization header format: {Authorization}")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = Authorization.split(" ", 1)[1]
    print(f"Debug: Extracted token (first 50 chars): {token[:50]}...")
    try:
        print("Debug: Attempting to verify ID token...")
        decoded = firebase_auth.verify_id_token(token)
        print(f"Debug: Token decoded successfully: {decoded}")
        uid = decoded.get("uid")
        print(f"Debug: UID extracted: {uid}")
        if not uid:
            raise ValueError("No UID in token")
        return uid
    except Exception as e:
        print(f"Debug: Token verification failed with error: {str(e)}")
        print(f"Debug: Error type: {type(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

# --- Pi Device Registration Endpoint ---
from fastapi import Body

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

# Initialize Google Cloud clients
try:
    # In Cloud Run, use Application Default Credentials (ADC)
    # For local development, set GOOGLE_APPLICATION_CREDENTIALS environment variable
    google_creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    
    if google_creds_json:
        # Parse JSON credentials from environment variable
        import json
        creds_info = json.loads(google_creds_json)
        creds = service_account.Credentials.from_service_account_info(creds_info)
        client = vision.ImageAnnotatorClient(credentials=creds)
        db = firestore.Client(credentials=creds, project=creds_info.get('project_id'))
        print("Using credentials from environment variable")
    else:
        # Use Application Default Credentials (recommended for Cloud Run)
        client = vision.ImageAnnotatorClient()
        db = firestore.Client()
        print("Using Application Default Credentials")
        
except Exception as e:
    print(f"Failed to create Google clients: {str(e)}")
    raise RuntimeError("Failed to create Google clients: " + str(e))

# Initialize Firebase Admin for verifying ID tokens
try:
    if not firebase_admin._apps:
        firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS_JSON')
        firebase_storage_bucket = os.environ.get('FIREBASE_STORAGE_BUCKET')
        
        if firebase_creds_json:
            # Parse JSON credentials from environment variable
            import json
            creds_info = json.loads(firebase_creds_json)
            cred = firebase_credentials.Certificate(creds_info)
            
            # Initialize with storage bucket if provided
            if firebase_storage_bucket:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': firebase_storage_bucket
                })
                print(f"Initialized Firebase Admin with storage bucket: {firebase_storage_bucket}")
            else:
                firebase_admin.initialize_app(cred)
                print("Initialized Firebase Admin without storage bucket")
        else:
            # Use Application Default Credentials for Firebase as well
            firebase_admin.initialize_app()
            print("Initialized Firebase Admin with default credentials")
except Exception as e:
    print(f"Firebase Admin init failed: {str(e)}")

def get_current_user_uid(Authorization: str | None = Header(default=None)) -> str:
    """Verify ID token and return the user's UID (used as Firestore document id)."""
    print(f"Debug: Authorization header received: {Authorization}")
    
    if not Authorization or not Authorization.lower().startswith("bearer "):
        print(f"Debug: Invalid Authorization header format: {Authorization}")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = Authorization.split(" ", 1)[1]
    print(f"Debug: Extracted token (first 50 chars): {token[:50]}...")
    
    try:
        print("Debug: Attempting to verify ID token...")
        decoded = firebase_auth.verify_id_token(token)
        print(f"Debug: Token decoded successfully: {decoded}")
        
        uid = decoded.get("uid")
        print(f"Debug: UID extracted: {uid}")
        
        if not uid:
            raise ValueError("No UID in token")
        return uid
    except Exception as e:
        print(f"Debug: Token verification failed with error: {str(e)}")
        print(f"Debug: Error type: {type(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

def save_inventory_item(item: dict, user_key: str | None = None) -> None:
    """Persist an inventory item to Firestore in collection 'inventory'."""
    try:
        if user_key:
            print(f"Debug: Saving inventory item for user {user_key}: {item}")
            doc_ref = db.collection('users').document(user_key).collection('inventory').add(item)
            print(f"Debug: Item saved with ID: {doc_ref[1].id}")
        else:
            print(f"Debug: Saving inventory item to global collection: {item}")
            db.collection('inventory').add(item)
    except Exception as e:
        # Log and ignore to avoid breaking API response
        print(f"Firestore save failed: {str(e)}")

def save_image_to_firestore(image_data: bytes, user_key: str, item_id: str = None) -> str:
    """Save image to Firebase Storage and return the image URL."""
    try:
        print(f"Debug: Starting image upload for user: {user_key}")
        print(f"Debug: Image data size: {len(image_data)} bytes")
        
        # Get image format and size
        try:
            img = Image.open(io.BytesIO(image_data))
            image_format = img.format or 'JPEG'
            image_size = len(image_data)
            # Get image dimensions
            width, height = img.size
            print(f"Debug: Image format: {image_format}, dimensions: {width}x{height}")
        except Exception as e:
            print(f"Debug: Failed to analyze image: {str(e)}")
            image_format = 'UNKNOWN'
            image_size = len(image_data)
            width, height = 0, 0
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"image_{timestamp}_{user_key[:8]}.{image_format.lower()}"
        print(f"Debug: Generated filename: {filename}")
        
        # Upload to Firebase Storage
        try:
            print("Debug: Attempting to get Firebase Storage bucket...")
            # Get bucket name from environment or use default
            bucket_name = os.environ.get('FIREBASE_STORAGE_BUCKET')
            if bucket_name:
                bucket = firebase_storage.bucket(bucket_name)
            else:
                bucket = firebase_storage.bucket()  # Uses default bucket
            print(f"Debug: Got bucket: {bucket}")
            
            blob = bucket.blob(f"images/{user_key}/{filename}")
            print(f"Debug: Created blob: {blob}")
            
            print("Debug: Uploading image to Firebase Storage...")
            blob.upload_from_string(image_data, content_type=f"image/{image_format.lower()}")
            print("Debug: Upload successful")
            
            # Make the image publicly accessible
            print("Debug: Making image public...")
            blob.make_public()
            image_url = blob.public_url
            print(f"Debug: Image URL: {image_url}")
            
        except Exception as e:
            print(f"Debug: Firebase Storage error: {str(e)}")
            print(f"Debug: Error type: {type(e)}")
            raise
        
        # Store metadata in Firestore (without the large image data)
        try:
            print("Debug: Storing metadata in Firestore...")
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
            
            # Save metadata to Firestore
            doc_ref = db.collection('images').add(image_doc)
            image_id = doc_ref[1].id
            print(f"Debug: Stored in Firestore with ID: {image_id}")
            
            # Update the document with the ID for reference
            db.collection('images').document(image_id).update({'id': image_id})
            
            return image_id
            
        except Exception as e:
            print(f"Debug: Firestore error: {str(e)}")
            raise
        
    except Exception as e:
        print(f"Failed to save image to Firebase Storage: {str(e)}")
        print(f"Full error details: {type(e).__name__}: {str(e)}")
        return None

# Food data
FOOD_KEYWORDS = [
    # Fruits
    "apple", "banana", "orange", "grape", "mango", "pineapple", "papaya", "pear",
    "strawberry", "blueberry", "raspberry", "watermelon", "melon", "lemon", "lime",
    "peach", "plum", "cherry", "kiwi", "guava", "pomegranate", "coconut",
    # Vegetables
    "potato", "tomato", "onion", "carrot", "lettuce", "cabbage", "spinach", "broccoli",
    "cauliflower", "capsicum", "chili", "garlic", "ginger", "peas", "corn", "okra",
    "radish", "beetroot", "brinjal", "cucumber", "zucchini", "mushroom", "beans",
]

FOOD_EXPIRY_DAYS = {
    # Fruits
    "apple": 30, "banana": 7, "orange": 20, "grape": 7, "mango": 7, "pineapple": 5,
    "papaya": 5, "pear": 14, "strawberry": 5, "blueberry": 7, "raspberry": 4,
    "watermelon": 7, "melon": 7, "lemon": 21, "lime": 21, "peach": 7, "plum": 7,
    "cherry": 5, "kiwi": 10, "guava": 5, "pomegranate": 30, "coconut": 30,
    # Vegetables
    "potato": 30, "tomato": 7, "onion": 30, "carrot": 15, "lettuce": 7, "cabbage": 10,
    "spinach": 5, "broccoli": 7, "cauliflower": 10, "capsicum": 10, "chili": 14,
    "garlic": 30, "ginger": 21, "peas": 7, "corn": 7, "okra": 5, "radish": 10,
    "beetroot": 14, "brinjal": 5, "cucumber": 7, "zucchini": 7, "mushroom": 3, "beans": 7,
}



@app.post("/route/")
async def route_image(image: UploadFile = File(...), user_key: str = Depends(get_user_from_auth)):
    """
    Identify food items only. If the image is invalid, return invalid.
    Returns: { "food_name": string, "expiry_date": YYYY-MM-DD } or { "message": "invalid" }
    """
    try:
        contents = await image.read()
        mime_type = image.content_type or "image/jpeg"
        
        # Save image to Firestore first
        image_id = save_image_to_firestore(contents, user_key)
        
        # First try direct food prediction
        try:
            return await process_food_prediction(contents, user_key, image_id)
        except HTTPException as he:
            if he.status_code != 400:
                raise
            # If not a food by labels, try to interpret as barcode -> food
            try:
                return await process_barcode_to_food(contents, mime_type, user_key, image_id)
            except HTTPException as he2:
                if he2.status_code == 400:
                    # Store unknown as requested
                    try:
                        save_inventory_item({
                            "food_name": "unknown",
                            "expiry_date": None,
                            "source": "unknown",
                            "image_id": image_id,
                            "quantity": 1,
                            "created_at": datetime.now().isoformat(),
                            "status": "active"
                        }, user_key)
                    except Exception:
                        pass
                    return JSONResponse(status_code=400, content={"message": "invalid"})
                raise
    except HTTPException as he:
        if he.status_code == 400:
            return JSONResponse(status_code=400, content={"message": "invalid"})
        raise
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return JSONResponse(status_code=400, content={"message": "invalid"})

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
                except Exception as e:
                    print(f"Failed to fetch image info for item {doc.id}: {str(e)}")
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

async def process_barcode_to_food(contents: bytes, mime_type: str = "image/jpeg", user_key: str | None = None, image_id: str | None = None):
    """
    Extract a barcode using Google Vision OCR (simple and robust), look up the product on OpenFoodFacts,
    and return the product name (unmodified) with an estimated expiry date.
    Returns { "food_name": product_name, "expiry_date": YYYY-MM-DD } or raises 400 if invalid.
    """
    # Simple OCR-based extraction using Vision Text Detection with checksum validation
    try:
        vision_image = vision.Image(content=contents)
        resp = client.text_detection(image=vision_image)
        texts = resp.text_annotations or []
        if not texts:
            raise HTTPException(status_code=400, detail="invalid")
        full_text = texts[0].description.replace("\n", " ")

        # Extract digit runs 8-14 length
        raw_candidates = re.findall(r"\b\d{8,14}\b", full_text)
        if not raw_candidates:
            raise HTTPException(status_code=400, detail="invalid")

        def gtin_valid(code: str) -> bool:
            # Supports EAN-8 (8), UPC-A (12), EAN-13 (13), GTIN-14 (14)
            if len(code) not in (8, 12, 13, 14) or not code.isdigit():
                return False
            digits = [int(c) for c in code]
            check = digits[-1]
            body = digits[:-1]
            # Right to left weighting 3/1 starting from position 1
            total = 0
            for i, d in enumerate(reversed(body), start=1):
                total += d * (3 if i % 2 == 1 else 1)
            calc = (10 - (total % 10)) % 10
            return calc == check

        # Prioritize valid codes by common lengths
        ordered_lengths = [13, 12, 14, 8]
        barcode = None
        for L in ordered_lengths:
            for c in raw_candidates:
                if len(c) == L and gtin_valid(c):
                    barcode = c
                    break
            if barcode:
                break
        if not barcode:
            # fallback to any candidate if none validates
            barcode = raw_candidates[0]

        # Try to extract GS1 Application Identifier (17) expiry date (YYMMDD)
        expiry_str = None
        gs1_patterns = [
            r"\(17\)\s*(\d{6})",
            r"[^0-9]17\s*(\d{6})",
            
            r"\b17(\d{6})\b",
        ]
        for pat in gs1_patterns:
            m = re.search(pat, full_text)
            if m:
                expiry_str = m.group(1)
                break
    except HTTPException:
        raise
    except Exception as e:
        print(f"Vision OCR barcode extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail="invalid")

    # Query OpenFoodFacts for product
    try:
        api_url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        r = requests.get(api_url, timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"OpenFoodFacts lookup failed: {str(e)}")
        raise HTTPException(status_code=400, detail="invalid")

    if data.get("status") != 1:
        # Store as unknown with barcode
        try:
            save_inventory_item({
                "food_name": "unknown",
                "expiry_date": None,
                "source": "barcode",
                "barcode": barcode,
                "image_id": image_id,
                "quantity": 1,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }, user_key)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="invalid")

    product = data.get("product", {})
    # Use product_name as returned by OpenFoodFacts (do not remap)
    raw_product_name = product.get("product_name") or product.get("generic_name") or "Unknown Product"

    # Determine expiry date: prefer GS1 (17) from barcode text; else OFF fields; else invalid
    parsed_expiry = None
    if expiry_str:
        try:
            yy = int(expiry_str[0:2])
            mm = int(expiry_str[2:4])
            dd = int(expiry_str[4:6])
            year = 2000 + yy  # assume 20xx
            parsed_expiry = datetime(year, mm, dd).strftime("%Y-%m-%d")
        except Exception:
            parsed_expiry = None

    if not parsed_expiry:
        off_expiry = product.get("expiration_date") or product.get("best_before_date") or product.get("minimum_durability_date")
        if off_expiry:
            try:
                # Normalize common formats to YYYY-MM-DD
                # If already in YYYY-MM-DD, keep; if DD/MM/YYYY or similar, try parse
                from dateutil import parser as dateparser  # optional if installed
                parsed = dateparser.parse(off_expiry, dayfirst=False, yearfirst=True)
                parsed_expiry = parsed.strftime("%Y-%m-%d")
            except Exception:
                # If dateutil not available or parse fails, accept as-is if looks like YYYY-MM-DD
                if re.match(r"^\d{4}-\d{2}-\d{2}$", off_expiry):
                    parsed_expiry = off_expiry

    if not parsed_expiry:
        # Store and return with null expiry
        try:
            save_inventory_item({
                "food_name": raw_product_name,
                "expiry_date": None,
                "source": "barcode",
                "barcode": barcode,
                "image_id": image_id,
                "quantity": 1,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }, user_key)
        except Exception:
            pass
        return {
            "food_name": raw_product_name,
            "expiry_date": None
        }

    # Store success entry
    try:
        save_inventory_item({
            "food_name": raw_product_name,
            "expiry_date": parsed_expiry,
            "source": "barcode",
            "barcode": barcode,
            "image_id": image_id,
            "quantity": 1,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }, user_key)
    except Exception:
        pass
    return {
        "food_name": raw_product_name,
        "expiry_date": parsed_expiry
    }

 

async def process_food_prediction(contents: bytes, user_key: str | None = None, image_id: str | None = None):
    """
    Process food prediction from image contents and return only food name and expiry date.
    If not a food image, raise 400 to signal invalid.
    """
    try:
        vision_image = vision.Image(content=contents)
        response = client.label_detection(image=vision_image, max_results=10)
    except Exception as e:
        print(f"Vision API label detection failed: {str(e)}")
        raise HTTPException(status_code=400, detail="invalid")

    labels = response.label_annotations or []

    # Find matching food strictly from keywords
    food_name = None
    for label in labels:
        desc = (label.description or "").lower()
        if desc in FOOD_KEYWORDS:
            food_name = desc
            break

    if not food_name:
        raise HTTPException(status_code=400, detail="invalid")

    expiry_days = FOOD_EXPIRY_DAYS.get(food_name, 7)
    expiry_date = (datetime.today() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
    
    # Store recognized food from image
    try:
        save_inventory_item({
            "food_name": food_name,
            "expiry_date": expiry_date,
            "source": "image",
            "image_id": image_id,
            "quantity": 1,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }, user_key)
    except Exception:
        pass
    return {
        "food_name": food_name,
        "expiry_date": expiry_date
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Food identification service is running"}

@app.get("/test-auth")
async def test_auth():
    """Test endpoint to verify Firebase Admin is working."""
    try:
        # Check if Firebase Admin is initialized
        if not firebase_admin._apps:
            return {"status": "error", "message": "Firebase Admin not initialized"}
        
        # Check if we can access firebase_auth
        if not firebase_auth:
            return {"status": "error", "message": "Firebase Auth not available"}
        
        return {
            "status": "success", 
            "message": "Firebase Admin is working",
            "apps": list(firebase_admin._apps.keys())
        }
    except Exception as e:
        return {"status": "error", "message": f"Firebase Admin test failed: {str(e)}"}

@app.get("/test-vision")
async def test_vision():
    """Test if Vision API client is working"""
    try:
        # Create a simple test image
        test_image = vision.Image(content=b"test")
        # Try to detect labels (this should fail gracefully with test data)
        response = client.label_detection(image=test_image, max_results=1)
        return {"status": "success", "message": "Vision API client is working", "client_type": str(type(client))}
    except Exception as e:
        return {"status": "error", "message": f"Vision API client error: {str(e)}", "client_type": str(type(client))}

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

@app.get("/test-image/{image_id}")
async def test_image_no_auth(image_id: str):
    """Test endpoint without authentication for development"""
    try:
        doc = db.collection('images').document(image_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Image not found")
        
        data = doc.to_dict()
        
        # Create HTML page with embedded image
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
                <h1>Image Viewer (Test Mode)</h1>
                <div class="image-info">
                    <h3>Image Details</h3>
                    <p><strong>Filename:</strong> {data.get('filename', 'Unknown')}</p>
                    <p><strong>Format:</strong> {data.get('format', 'Unknown')}</p>
                    <p><strong>Size:</strong> {data.get('size_bytes', 0)} bytes</p>
                    <p><strong>Dimensions:</strong> {data.get('width', 0)} x {data.get('height', 0)} pixels</p>
                    <p><strong>Uploaded:</strong> {data.get('created_at', 'Unknown')}</p>
                    <p><strong>User:</strong> {data.get('user_key', 'Unknown')}</p>
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

