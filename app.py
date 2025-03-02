from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
import json
import uuid

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "ubayog_secret_123"
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False

# Enable CORS with credentials
CORS(app, 
     origins=["http://localhost:8000"],
     supports_credentials=True,
     methods=["POST", "OPTIONS"],
     allow_headers=["Content-Type"])

# Load NLP models
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load mock database
try:
    with open('assets.json') as f:
        assets = json.load(f)
except FileNotFoundError:
    assets = []

# Precompute embeddings
asset_embeddings = {asset['id']: sbert_model.encode(asset['description']) for asset in assets}

# Intent samples
INTENTS = {
    "search": ["find bikes", "show apartments", "search for tools"],
    "list": ["i want to list my car", "add new property", "rent out equipment"],
    "faq": ["what is ubayog?", "how do rewards work?"]
}

# Precompute intent embeddings
intent_embeddings = {intent: sbert_model.encode(examples) for intent, examples in INTENTS.items()}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_intent(query):
    query_embed = sbert_model.encode([query])[0]
    scores = {}
    for intent, examples in intent_embeddings.items():
        scores[intent] = max([cosine_similarity(query_embed, ex) for ex in examples])
    return max(scores, key=scores.get)

def extract_entities(query):
    doc = nlp(query)
    return {
        "location": [ent.text for ent in doc.ents if ent.label_ == "GPE"],
        "price": [ent.text for ent in doc.ents if ent.label_ == "MONEY"],
        "type": [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"]
    }

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    user_msg = request.get_data(as_text=True)
    intent = detect_intent(user_msg)
    
    # Handle listing flow
    if intent == "list":
        if 'listing_state' not in session:
            session['listing_state'] = {
                'step': 'asset_type',
                'data': {}
            }
            return _cors_response({"text": "Let's list your asset! What type of asset are you listing? (e.g., bike, apartment)"})
        
        return handle_listing_flow(user_msg)
    
    # Handle search
    if intent == "search":
        query_embed = sbert_model.encode([user_msg])[0]
        scores = {asset['id']: cosine_similarity(query_embed, asset_embeddings[asset['id']]) 
                 for asset in assets}
        top_assets = sorted(assets, key=lambda x: -scores[x['id']])[:3]
        return _cors_response({
            "text": f"Found {len(top_assets)} results:",
            "results": top_assets
        })
    
    return _cors_response({"text": "I can help you search assets or list new ones. Try asking 'Find bikes in Paris'!"})

def handle_listing_flow(user_msg):
    state = session['listing_state']
    
    if state['step'] == 'asset_type':
        state['data']['type'] = user_msg
        state['step'] = 'location'
        session.modified = True
        return _cors_response({"text": "Great! Where is the asset located?"})
    
    elif state['step'] == 'location':
        state['data']['location'] = user_msg
        state['step'] = 'price'
        session.modified = True
        return _cors_response({"text": "What's the daily rental price?"})
    
    elif state['step'] == 'price':
        state['data']['price'] = user_msg
        new_asset = {
            "id": str(uuid.uuid4()),
            "description": f"{state['data']['type']} in {state['data']['location']} for {state['data']['price']}/day",
            **state['data']
        }
        assets.append(new_asset)
        asset_embeddings[new_asset['id']] = sbert_model.encode(new_asset['description'])
        session.pop('listing_state', None)
        return _cors_response({
            "text": f"Asset listed successfully! ID: {new_asset['id']}",
            "preview": new_asset
        })

def _build_cors_preflight_response():
    response = jsonify()
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _cors_response(data, status=200):
    response = jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:8000")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=False)