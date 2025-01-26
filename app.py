"""
from flask import Flask, request, jsonify
import os
import json
import numpy as np
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Initialize Flask app
app = Flask(__name__)

# Azure Blob Storage setup
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_METADATA_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

EMBEDDINGS_CONTAINER = "weez-files-embeddings"

# Initialize the Sentence Transformer model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@app.route('/search', methods=['POST'])
def search():
    try:
        # Step 1: Parse the query and user ID from the request
        data = request.get_json()
        query = data.get('query')
        user_id = data.get('user_id')

        if not query or not user_id:
            return jsonify({"error": "Query and user_id are required."}), 400

        # Step 2: Perform NER on the query (mocked here, replace with your NER logic)
        # Assuming NER returns a list of relevant entities from the query
        entities = perform_ner(query)

        # Step 3: Generate embeddings for the extracted entities
        query_text = " ".join(entities)
        query_embedding = embedding_model.encode(query_text).reshape(1, -1)

        # Step 4: Access all embeddings for the user
        embeddings_container_client = blob_service_client.get_container_client(EMBEDDINGS_CONTAINER)
        user_prefix = f"{user_id}/"
        blobs = embeddings_container_client.list_blobs(name_starts_with=user_prefix)

        max_similarity = -1
        best_match = None

        for blob in blobs:
            blob_client = embeddings_container_client.get_blob_client(blob.name)
            blob_content = blob_client.download_blob().readall().decode('utf-8')
            embedding_data = json.loads(blob_content)

            file_name = embedding_data.get("file_name")
            file_path = embedding_data.get("file_path")
            file_embeddings = np.array(embedding_data.get("embeddings"))

            # Step 5: Compute cosine similarity
            similarity = cosine_similarity(query_embedding, file_embeddings.reshape(1, -1))[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = {
                    "file_name": file_name,
                    "file_path": file_path,
                    "similarity_score": max_similarity
                }

        if best_match:
            print(max_similarity)
            return jsonify(best_match), 200
        else:
            return jsonify({"message": "No matching files found."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def perform_ner(query):
    messages = [
        {
            "role": "system",
            "content": ""You are a Named Entity Recognition (NER) expert. Your task is to analyze a given text and identify specific
               entities based on the following tags:

               - **B-DOC / I-DOC**: Represents document types like PDF, DOCX, XLSX, PPTX, and their variants. It also includes generic document mentions such as "report", "presentation", or "excel sheet".
               - **B-PER / I-PER**: Represents names of people such as "John Watson", "Elon Musk", or "Mary".
               - **B-TOP / I-TOP**: Represents topics or subjects such as "natural disaster", "machine learning", or "climate change".
               - **B-DATE / I-DATE**: Represents relative or absolute dates, such as "two months ago", "on April 24th", "one year ago", "24/10/24", or "yesterday".

               Each word in the input should be tagged as:
               - `B-TAG`: Beginning of an entity.
               - `I-TAG`: Continuation of an entity.
               - `O`: Not part of any entity.

               ### Additional Guidelines:
               1. The query can be formal or informal.
                  - Example (Formal): "Please provide me with the PDF about natural disasters."
                  - Example (Informal): "Give me a pdf on natural disasters sent by John."
                  - Example (Vague): "Show me the document which contains data on natural disasters."
                  - Example (Direct Data): "Different kinds of natural disasters."
               2. Even if the query directly mentions file content or metadata (like topics, dates, or names), identify and tag all relevant entities.

               ### Example:

               **Input**:  
               "Give me a pdf sent by John on the topic of natural disaster two months ago."

               **Output**:  
               ```plaintext
               Give  O
               me    O
               a     O
               pdf   B-DOC
               sent  O
               by    O
               John  B-PER
               on    O
               the   O
               topic O
               of    O
               natural B-TOP
               disaster I-TOP
               two   B-DATE
               months I-DATE
               ago    I-DATE
               ```
               ""
        },
        {
            "role": "user",
            "content": query
        }
    ]

    # Call OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
        max_tokens=100
    )

    # Return the response content
    return response['choices'][0]['message']['content'].strip()


if __name__ == '__main__':
    app.run(debug=True)
"""

from flask import Flask, request, jsonify
import os
import json
import numpy as np
from azure.storage.blob import BlobServiceClient
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Initialize Flask app
app = Flask(__name__)

# Azure Blob Storage setup
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_METADATA_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

EMBEDDINGS_CONTAINER = "weez-files-embeddings"

# Set up OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/search', methods=['POST'])
def search():
    try:
        # Step 1: Parse the query and user ID from the request
        data = request.get_json()
        query = data.get('query')
        user_id = data.get('user_id')

        if not query or not user_id:
            return jsonify({"error": "Query and user_id are required."}), 400

        # Step 2: Perform NER on the query (mocked here, replace with your NER logic)
        entities = perform_ner(query)

        # Step 3: Generate embeddings for the extracted entities
        query_text = " ".join(entities)
        query_embedding_response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query_text
        )
        query_embedding = np.array(query_embedding_response['data'][0]['embedding']).reshape(1, -1)

        # Step 4: Access all embeddings for the user
        embeddings_container_client = blob_service_client.get_container_client(EMBEDDINGS_CONTAINER)
        user_prefix = f"{user_id}/"
        blobs = embeddings_container_client.list_blobs(name_starts_with=user_prefix)

        max_similarity = -1
        best_match = None

        for blob in blobs:
            blob_client = embeddings_container_client.get_blob_client(blob.name)
            blob_content = blob_client.download_blob().readall().decode('utf-8')
            embedding_data = json.loads(blob_content)

            file_name = embedding_data.get("file_name")
            file_path = embedding_data.get("file_path")
            file_embeddings = np.array(embedding_data.get("embeddings"))

            # Step 5: Compute cosine similarity
            similarity = cosine_similarity(query_embedding, file_embeddings.reshape(1, -1))[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = {
                    "file_name": file_name,
                    "file_path": file_path,
                    "similarity_score": max_similarity
                }

        if best_match:
            return jsonify(best_match), 200
        else:
            return jsonify({"message": "No matching files found."}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def perform_ner(query):
    messages = [
        {
            "role": "system",
            "content": """You are a Named Entity Recognition (NER) expert. Your task is to analyze a given text and identify specific 
               entities based on the following tags:

               - **B-DOC / I-DOC**: Represents document types like PDF, DOCX, XLSX, PPTX, and their variants. It also includes generic document mentions such as "report", "presentation", or "excel sheet".
               - **B-PER / I-PER**: Represents names of people such as "John Watson", "Elon Musk", or "Mary".
               - **B-TOP / I-TOP**: Represents topics or subjects such as "natural disaster", "machine learning", or "climate change".
               - **B-DATE / I-DATE**: Represents relative or absolute dates, such as "two months ago", "on April 24th", "one year ago", "24/10/24", or "yesterday".

               Each word in the input should be tagged as:
               - `B-TAG`: Beginning of an entity.
               - `I-TAG`: Continuation of an entity.
               - `O`: Not part of any entity.

               ### Additional Guidelines:
               1. The query can be formal or informal.
                  - Example (Formal): "Please provide me with the PDF about natural disasters."
                  - Example (Informal): "Give me a pdf on natural disasters sent by John."
                  - Example (Vague): "Show me the document which contains data on natural disasters."
                  - Example (Direct Data): "Different kinds of natural disasters."
               2. Even if the query directly mentions file content or metadata (like topics, dates, or names), identify and tag all relevant entities.

               ### Example:

               **Input**:  
               "Give me a pdf sent by John on the topic of natural disaster two months ago."

               **Output**:  
               ```plaintext
               Give  O
               me    O
               a     O
               pdf   B-DOC
               sent  O
               by    O
               John  B-PER
               on    O
               the   O
               topic O
               of    O
               natural B-TOP
               disaster I-TOP
               two   B-DATE
               months I-DATE
               ago    I-DATE
               ```"""
        },
        {
            "role": "user",
            "content": query
        }
    ]

    # Call OpenAI API for NER
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
        max_tokens=100
    )

    # Return the response content
    return response['choices'][0]['message']['content'].strip().split("\n")

if __name__ == '__main__':
    app.run(debug=True)
