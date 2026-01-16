import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HUBSPOT_ACCESS_TOKEN = os.getenv("HUBSPOT_ACCESS_TOKEN")
    
    if not HUBSPOT_ACCESS_TOKEN:
        raise ValueError("❌ HUBSPOT_ACCESS_TOKEN missing from .env file")
    
    print("✅ Environment variables loaded successfully")