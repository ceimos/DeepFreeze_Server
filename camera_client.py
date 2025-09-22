#!/usr/bin/env python3
"""
Raspberry Pi Camera Client for DeepFreeze
Captures images from Pi camera and sends them to the /route endpoint
"""

from picamera2 import Picamera2
import cv2
import requests
import time
import os
import json
from datetime import datetime
from PIL import Image
import io
import argparse

class PiCameraClient:
    def __init__(self, server_url="http://localhost:8000", api_key=None, pi_key=None):
        """
        Initialize the Pi Camera Client
        Args:
            server_url (str): URL of your FastAPI server
            api_key (str): Firebase ID token for authentication
            pi_key (str): Pi device API key for authentication
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.pi_key = pi_key
        self.picam2 = None
        
    def start_camera(self):
        """Start the PiCamera2 camera"""
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_still_configuration(main={"size": (1920, 1080)})
            self.picam2.configure(config)
            self.picam2.start()
            print("PiCamera2 started successfully")
            return True
        except Exception as e:
            print(f"Failed to start PiCamera2: {e}")
            return False
    
    def stop_camera(self):
        """Stop the PiCamera2 camera"""
        if self.picam2:
            self.picam2.stop()
            self.picam2 = None
            print("PiCamera2 stopped")
    
    def capture_image(self, save_path=None):
        """
        Capture a single image from the PiCamera2
        
        Args:
            save_path (str): Optional path to save the captured image
            
        Returns:
            bytes: Image data as bytes
        """
        if not self.picam2:
            raise Exception("PiCamera2 is not started")
        
        # Capture frame as numpy array
        frame = self.picam2.capture_array()
        if frame is None:
            raise Exception("Failed to capture frame")
        
        # Convert to JPEG
        _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_bytes = img_encoded.tobytes()
        
        # Save image if path provided
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(img_bytes)
            print(f"Image saved to: {save_path}")
        
        return img_bytes
    
    def send_image_to_server(self, image_bytes, filename="captured_image.jpg"):
        """
        Send image to the /route endpoint
        Args:
            image_bytes (bytes): Image data
            filename (str): Filename for the upload
        Returns:
            dict: Server response
        """
        if not (self.api_key or self.pi_key):
            raise Exception("API key (Firebase ID token) or Pi device key is required")

        url = f"{self.server_url}/route/"

        # Prepare headers with authentication
        if self.pi_key:
            headers = {"Authorization": f"Pi-Key {self.pi_key}"}
        else:
            headers = {"Authorization": f"Bearer {self.api_key}"}

        # Prepare files for multipart upload
        files = {
            "image": (filename, image_bytes, "image/jpeg")
        }

        try:
            print(f"Sending image to {url}...")
            response = requests.post(url, headers=headers, files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                print("✅ Success! Food identified:")
                print(f"   Food: {result.get('food_name', 'Unknown')}")
                print(f"   Expiry: {result.get('expiry_date', 'Unknown')}")
                return result
            elif response.status_code == 400:
                result = response.json()
                if result.get('message') == 'invalid':
                    print("❌ Image not recognized as food or barcode")
                else:
                    print(f"❌ Error: {result}")
                return result
            else:
                print(f"❌ Server error: {response.status_code}")
                print(f"Response: {response.text}")
                return {"error": f"HTTP {response.status_code}"}

        except requests.exceptions.Timeout:
            print("❌ Request timed out")
            return {"error": "Request timed out"}
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return {"error": str(e)}
    
    def continuous_capture_mode(self, interval=5, max_captures=None):
        """
        Continuous capture mode - takes pictures at regular intervals
        
        Args:
            interval (int): Seconds between captures
            max_captures (int): Maximum number of captures (None for unlimited)
        """
        if not self.start_camera():
            return
        
        try:
            capture_count = 0
            print(f"Starting continuous capture mode (every {interval} seconds)")
            print("Press Ctrl+C to stop")
            
            while True:
                if max_captures and capture_count >= max_captures:
                    print(f"Reached maximum captures ({max_captures})")
                    break
                
                try:
                    # Create timestamp for filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}.jpg"
                    
                    # Capture and send image
                    print(f"\n--- Capture {capture_count + 1} ---")
                    image_bytes = self.capture_image(save_path=filename)
                    
                    # Send to server
                    result = self.send_image_to_server(image_bytes, filename)
                    
                    capture_count += 1
                    
                    # Wait for next capture
                    if max_captures is None or capture_count < max_captures:
                        print(f"Waiting {interval} seconds until next capture...")
                        time.sleep(interval)
                        
                except KeyboardInterrupt:
                    print("\nStopping continuous capture...")
                    break
                except Exception as e:
                    print(f"Error in capture {capture_count + 1}: {e}")
                    time.sleep(interval)
                    
        finally:
            self.stop_camera()
    
    def interactive_mode(self):
        """Interactive mode - press Enter to capture, 'q' to quit"""
        if not self.start_camera():
            return
        
        try:
            print("Interactive mode started!")
            print("Press Enter to capture an image, 'q' to quit")
            
            capture_count = 0
            while True:
                user_input = input(f"\nCapture {capture_count + 1} - Press Enter to capture, 'q' to quit: ").strip().lower()
                
                if user_input == 'q':
                    break
                
                try:
                    # Create timestamp for filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"capture_{timestamp}.jpg"
                    
                    # Capture and send image
                    print("Capturing image...")
                    image_bytes = self.capture_image(save_path=filename)
                    
                    # Send to server
                    result = self.send_image_to_server(image_bytes, filename)
                    
                    capture_count += 1
                    
                except Exception as e:
                    print(f"Error capturing image: {e}")
                    
        finally:
            self.stop_camera()

def main():
    # Use DeepFreezeServerURL environment variable if set, else default to localhost
    default_server_url = os.environ.get("DeepFreezeServerURL", "http://localhost:8000")
    parser = argparse.ArgumentParser(description="Pi Camera Client for DeepFreeze")
    parser.add_argument("--server", default=default_server_url, 
                       help=f"FastAPI server URL (default: {default_server_url})")
    parser.add_argument("--token", required=False, 
                       help="Firebase ID token for authentication")
    parser.add_argument("--pi-key", required=False, 
                       help="Pi device API key for authentication")
    parser.add_argument("--mode", choices=["single", "continuous", "interactive"], 
                       default="single", help="Capture mode (default: single)")
    parser.add_argument("--interval", type=int, default=5, 
                       help="Interval between captures in seconds (default: 5)")
    parser.add_argument("--max-captures", type=int, 
                       help="Maximum number of captures in continuous mode")
    parser.add_argument("--camera-index", type=int, default=0, 
                       help="Camera index (default: 0)")

    args = parser.parse_args()

    # Create client
    client = PiCameraClient(server_url=args.server, api_key=args.token, pi_key=args.pi_key)

    try:
        if args.mode == "single":
            # Single capture mode
            if not client.start_camera(args.camera_index):
                return

            try:
                # Capture image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"

                print("Capturing image...")
                image_bytes = client.capture_image(save_path=filename)

                # Send to server
                result = client.send_image_to_server(image_bytes, filename)

            finally:
                client.stop_camera()

        elif args.mode == "continuous":
            # Continuous capture mode
            client.continuous_capture_mode(interval=args.interval, max_captures=args.max_captures)

        elif args.mode == "interactive":
            # Interactive mode
            client.interactive_mode()

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
