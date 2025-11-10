from firebase_admin import credentials, db, initialize_app, firestore
import time
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter(
    prefix="/RTDB",
    tags=["rtdb"],
    responses={404: {"description": "Not found"}},
)

# Initialize Firebase Admin SDK once during application startup
cred = credentials.Certificate("smiling-gasket-468408-u8-06ed44d996c7.json")  # Replace with your Firebase service account key file
initialize_app(cred, {
    'databaseURL': 'https://smiling-gasket-468408-u8-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase Realtime Database URL
})
firestore_client = firestore.client()  # Initialize Firestore client

@router.get('/')
async def health_check():
    """
    Health check endpoint to verify the server is running.
    """
    return JSONResponse(content={"status": "Server is running"}, status_code=200)

@router.post('/update_firebase_readings')
async def update_firebase_readings(request: Request):
    """
    Update Firebase Realtime Database with sensor readings.

    Expects a JSON payload with the following keys:
        - pi_key (str): API key of the Pi device.
        - temperature (str): Temperature reading.
        - humidity (str): Humidity reading.
        - door_status (str): Door status.
        - gas_status (str): Gas status.
    """
    try:
        # Parse JSON payload
        data = await request.json()
        pi_key = data['pi_key']
        temperature = data['temperature']
        humidity = data['humidity']
        door_status = data['door_status']
        gas_status = data['gas_status']

        # Query Firestore to get the user_id (uid) associated with the pi_key
        device_query = firestore_client.collection('pi_devices').where('api_key', '==', pi_key).where('status', '==', 'active').stream()
        device = next(device_query, None)
        if not device:
            raise HTTPException(status_code=401, detail="Invalid or inactive Pi API key")
        
        device_data = device.to_dict()
        uid = device_data['user_id']

        # Optionally update the last_used timestamp for the device
        try:
            firestore_client.collection('pi_devices').document(device.id).update({'last_used': time.time()})
        except Exception as e:
            print(f"Failed to update last_used timestamp: {e}")

        # Generate timestamp
        timestamp = int(time.time())

        # Construct database paths
        database_path = f"/UsersData/{uid}/readings"
        parent_path = f"{database_path}/{timestamp}"

        # Update Firebase Realtime Database
        db.reference(f"{parent_path}/temperature").set(temperature)
        db.reference(f"{parent_path}/humidity").set(humidity)
        db.reference(f"{parent_path}/timestamp").set(timestamp)
        db.reference(f"{parent_path}/door_state").set(door_status)
        db.reference(f"{parent_path}/gas_state").set(gas_status)

        return JSONResponse(content={"message": "Readings updated successfully"}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))