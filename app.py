from flask import Flask, request, jsonify, session
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
import json
import uuid
import os

app = Flask(__name__)
app.secret_key = "ubayog_secret_123"
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True
)

CORS(app, 
     origins=["http://localhost:8000"],
     supports_credentials=True,
     methods=["POST", "OPTIONS", "GET"],
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Content-Type", "X-Custom-Header"]
)

nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load or initialize assets
ASSETS_FILE = 'assets.json'
if os.path.exists(ASSETS_FILE):
    with open(ASSETS_FILE) as f:
        assets = json.load(f)
else:
    assets = []

asset_embeddings = {asset['id']: sbert_model.encode(asset['description']) 
                   for asset in assets}

INTENT_EXAMPLES = {
    "search": [
        "find bikes under $50",
        "show apartments in paris",
        "search for camera equipment",
        "looking for power tools"
    ],
    "list": [
        "i want to list my car",
        "add new apartment listing",
        "rent out my photography gear"
    ],
    "faq": [
        "how do rewards work?",
        "what payment methods do you accept?"
    ]
}

intent_embeddings = {intent: sbert_model.encode(examples) 
                    for intent, examples in INTENT_EXAMPLES.items()}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def detect_intent(query):
    query_embed = sbert_model.encode([query])[0]
    scores = {}
    for intent, examples in intent_embeddings.items():
        scores[intent] = max(cosine_similarity(query_embed, ex) for ex in examples)
    return max(scores, key=scores.get)

def extract_entities(query):
    doc = nlp(query)
    asset_types = ['bike', 'apartment', 'car', 'tool', 'camera', 'equipment']
    return {
        "location": [ent.text.lower() for ent in doc.ents if ent.label_ == "GPE"],
        "price": [ent.text.lower() for ent in doc.ents if ent.label_ == "MONEY"],
        "type": [token.text.lower() for token in doc 
                if token.text.lower() in asset_types]
    }

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat_handler():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    user_msg = request.get_data(as_text=True).strip().lower()
    intent = detect_intent(user_msg)
    
    # Listing flow
    if intent == "list":
        return handle_listing(user_msg)
    
    # Search handling
    if intent == "search":
        return handle_search(user_msg)
    
    return _cors_response({"text": "How can I help you today?"})

def handle_search(query):
    filters = extract_entities(query)
    
    # Filter assets
    filtered = []
    for asset in assets:
        type_match = not filters['type'] or any(
            t in asset['type'].lower() for t in filters['type']
        )
        loc_match = not filters['location'] or any(
            l in asset['location'].lower() for l in filters['location']
        )
        if type_match and loc_match:
            filtered.append(asset)
    
    # Semantic ranking
    query_embed = sbert_model.encode([query])[0]
    scores = {a['id']: cosine_similarity(query_embed, asset_embeddings[a['id']]) 
             for a in filtered}
    results = sorted(filtered, key=lambda x: -scores[x['id']])[:3]
    
    return _cors_response({
        "text": f"Found {len(results)} matching items:",
        "results": results
    })

def handle_listing(user_msg):
    if 'listing_state' not in session:
        session['listing_state'] = {'step': 'type', 'data': {}}
        return _cors_response({"text": "Let's list your item! What type of item is it?"})
    
    state = session['listing_state']
    
    if state['step'] == 'type':
        state['data']['type'] = user_msg
        state['step'] = 'location'
        session.modified = True
        return _cors_response({"text": "Where is the item located?"})
    
    elif state['step'] == 'location':
        state['data']['location'] = user_msg
        state['step'] = 'price'
        session.modified = True
        return _cors_response({"text": "What's the daily rental price?"})
    
    elif state['step'] == 'price':
        state['data']['price'] = user_msg
        new_item = {
            "id": str(uuid.uuid4()),
            "description": f"{state['data']['type']} in {state['data']['location']}",
            **state['data']
        }
        
        # Persist data
        assets.append(new_item)
        with open(ASSETS_FILE, 'w') as f:
            json.dump(assets, f, indent=2)
        
        # Update embeddings
        asset_embeddings[new_item['id']] = sbert_model.encode(new_item['description'])
        session.pop('listing_state', None)
        
        return _cors_response({
            "text": "✅ Item listed successfully!",
            "preview": new_item
        })

def _build_cors_preflight_response():
    response = jsonify()
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response

def _cors_response(data, status=200):
    response = jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:8000")
    return response

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:8000'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

if __name__ == '__main__':
    app.run(port=5001, debug=False)