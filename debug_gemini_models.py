import google.generativeai as genai
import os

# User's key (used for debugging as permitted by user providing it in chat)
KEY = "AIzaSyBKhCicYB8BA0qz-YIa9jRACOwQAeW6j1k"

genai.configure(api_key=KEY)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
