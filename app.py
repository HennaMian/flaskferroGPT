from flask import Flask, request, jsonify
from flask_cors import CORS
from answer import getAnswer

app = Flask(__name__)
# CORS(app, resources={r"/process-text": {"origins": "https://sites.gatech.edu"}}, supports_credentials=True)
# CORS(app, resources={r"/process-text": {"origins": "https://electrons.ece.gatech.edu"}}, supports_credentials=True)
CORS(app, resources={r"/process-text": {"origins": ["https://electrons.ece.gatech.edu", "https://sites.gatech.edu"]}}, supports_credentials=True)



@app.route('/')
def home():
    return "Welcome to the Flask API"

@app.route('/process-text', methods=['POST'])
def process_text():
    input_text = request.form.get('search_query', '')

    # Modify the input text as needed
    modified_text = getAnswer(input_text)

    return jsonify(message=modified_text)

if __name__ == '__main__':
    app.run(debug=True)


