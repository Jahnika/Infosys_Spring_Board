from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# Replace with your Gemini API Key and endpoint
API_KEY = "AIzaSyBPrZ5rBsNXnDfPUWYUjqPEXhoBU72b-YA"
BASE_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1-translate:generate"

def translate_text_gemini(text, target_language):
    """
    Translate text using the Gemini API.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "target_language": target_language,
    }

    response = requests.post(BASE_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json().get("translation", "Translation unavailable")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

@app.route("/translate", methods=["POST"])
def translate():
    """
    Translate endpoint to process user inputs and return translated text.
    """
    data = request.json
    text = data.get("text")
    target_language = data.get("target_language")

    if not text or not target_language:
        return jsonify({"error": "Invalid input. Provide 'text' and 'target_language'."}), 400

    translated_text = translate_text_gemini(text, target_language)
    if translated_text:
        return jsonify({"translation": translated_text})
    else:
        return jsonify({"error": "Translation failed. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)