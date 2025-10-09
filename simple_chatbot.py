#!/usr/bin/env python3
"""
Simple Smart Fridge Chatbot
Everything in one file - no complex structure!
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
from datetime import datetime
from typing import List, Dict, Any
import uvicorn

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get API key from environment
GROQ_API_KEY = "***REMOVED***"
if not GROQ_API_KEY:
    print("❌ Please set GROQ_API_KEY environment variable")
    print("   Get your key from: https://console.groq.com/")
    exit(1)

# Firebase configuration
FIREBASE_PROJECT_ID = "smiling-gasket-468408-u8"
FIREBASE_SERVICE_ACCOUNT_PATH = "../inventory_management/smiling-gasket-468408-u8-868e68126259.json"

# =============================================================================
# DATA MODELS
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

# =============================================================================
# FIREBASE SETUP
# =============================================================================

def setup_firebase():
    """Setup Firebase connection"""
    try:
        if firebase_admin._apps:
            print("✅ Firebase already initialized")
            return firestore.client()
        
        print(f"🔍 Looking for service account file at: {FIREBASE_SERVICE_ACCOUNT_PATH}")
        print(f"🔍 File exists: {os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH)}")
        
        if os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
            print("✅ Service account file found")
            cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
        else:
            print("❌ Firebase service account file not found")
            print(f"   Looking for: {FIREBASE_SERVICE_ACCOUNT_PATH}")
            return None
        
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase connected successfully")
        return db
    except Exception as e:
        print(f"❌ Firebase error: {e}")
        return None

# =============================================================================
# AI CHATBOT
# =============================================================================

class SimpleChatbot:
    def __init__(self):
        # Try different models in order of preference
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
                print(f"✅ Successfully initialized with model: {model}")
                break
            except Exception as e:
                print(f"❌ Failed to initialize model {model}: {e}")
                continue
        
        if not self.llm:
            print("❌ Failed to initialize any Groq model")
            raise Exception("No working Groq model found")
            
        self.db = setup_firebase()
    
    def get_user_inventory(self, user_id: str) -> List[Dict[str, Any]]:
        """Get available ingredients from user's fridge with quantity information"""
        if not self.db:
            print("❌ No Firebase connection")
            return []
        
        try:
            print(f"🔍 Looking for inventory for user_id: {user_id}")
            
            # Get inventory from Firestore
            inventory_ref = self.db.collection("users").document(user_id).collection("inventory")
            docs = inventory_ref.stream()
            
            ingredients = []
            doc_count = 0
            all_items = []
            
            for doc in docs:
                doc_count += 1
                data = doc.to_dict()
                # Map the correct field names from React Native app structure
                item_name = data.get('food_name', data.get('name', 'Unknown'))
                quantity = data.get('quantity', 0)
                unit = data.get('unit', '')
                status = data.get('status', 'active')
                
                all_items.append({
                    'name': item_name,
                    'quantity': quantity,
                    'unit': unit,
                    'status': status,
                    'raw_data': data
                })
                
                print(f"📦 Found item: {item_name} - Quantity: {quantity} {unit} - Status: {status}")
                
                # Only include active items with quantity > 0
                if status == 'active' and quantity > 0:
                    ingredients.append({
                        'name': item_name.lower(),
                        'quantity': quantity,
                        'unit': unit,
                        'display_name': item_name
                    })
            
            print(f"✅ Found {doc_count} total documents")
            print(f"📊 All items: {all_items}")
            print(f"🥘 Active ingredients with quantity > 0: {ingredients}")
            return ingredients
        except Exception as e:
            print(f"❌ Error getting inventory: {e}")
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
            
        except Exception as e:
            print(f"❌ AI model error: {e}")
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
            
        except Exception as e:
            print(f"❌ AI model error: {e}")
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

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="Simple Smart Fridge Chatbot", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = SimpleChatbot()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {"message": "Simple Smart Fridge Chatbot", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the AI assistant"""
    try:
        response = chatbot.chat(request.user_id, request.message)
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/inventory")
async def get_inventory(user_id: str):
    """Get user's fridge inventory"""
    if not chatbot.db:
        return {"error": "Firebase not connected", "items": []}
    
    try:
        inventory_ref = chatbot.db.collection("users").document(user_id).collection("inventory")
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
        
        return {"user_id": user_id, "items": items, "total_items": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/inventory")
async def debug_inventory(user_id: str):
    """Debug endpoint to see raw inventory data"""
    if not chatbot.db:
        return {"error": "Firebase not connected", "raw_data": []}
    
    try:
        inventory_ref = chatbot.db.collection("users").document(user_id).collection("inventory")
        docs = inventory_ref.stream()
        
        raw_data = []
        for doc in docs:
            data = doc.to_dict()
            raw_data.append({
                "doc_id": doc.id,
                "data": data
            })
        
        return {
            "user_id": user_id, 
            "total_docs": len(raw_data),
            "raw_data": raw_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# RUN THE APP
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting Simple Smart Fridge Chatbot...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)