from flask import Flask, request, jsonify
import os
import json
import numpy as np
from azure.storage.blob import BlobServiceClient
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
import threading
import re

app = Flask(__name__)

# Azure Blob Storage setup
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_METADATA_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
EMBEDDINGS_CONTAINER = "weez-files-embeddings"

# Thread-local storage
thread_local = threading.local()


@app.route('/search', methods=['POST'])
def search():
    try:
        # Initialize thread-local storage
        thread_local.matches = []
        thread_local.processed_files = set()

        # Get request data
        data = request.get_json()
        query = data.get('query')
        user_id = data.get('user_id')

        # Validate input
        if not query or not user_id:
            return jsonify({"error": "Query and user_id are required."}), 400

        # 1. NER Processing
        ner_results = perform_ner(query)
        entities = extract_entities_from_ner(ner_results)
        query_text = " ".join(entities) if entities else query
        print(f"[DEBUG] Search text: {query_text}")

        # 2. Embedding Generation
        embedding_client = AzureOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            api_version="2024-02-01",
            azure_endpoint="https://weez-openai-resource.openai.azure.com/"
        )

        try:
            # Generate query embedding
            embedding_response = embedding_client.embeddings.create(
                model="text-embedding-3-large",
                input=query_text
            )
            query_embedding = np.array(embedding_response.data[0].embedding).reshape(1, -1)
        finally:
            embedding_client.close()

        # 3. Blob Processing
        container_client = blob_service_client.get_container_client(EMBEDDINGS_CONTAINER)
        blobs = container_client.list_blobs(name_starts_with=f"{user_id}/")

        for blob in blobs:
            if blob.name in thread_local.processed_files:
                continue
            thread_local.processed_files.add(blob.name)

            try:
                # 4. Filename Extraction
                original_filename = blob.name.split("/")[-1].rsplit(".json", 1)[0]
                print(f"[DEBUG] Processing file: {original_filename}")

                # 5. Embedding Retrieval
                blob_data = json.loads(
                    container_client.get_blob_client(blob.name)
                    .download_blob()
                    .readall()
                    .decode('utf-8')
                )

                # 6. Similarity Calculation
                file_embeddings = np.array(blob_data["embeddings"]).reshape(1, -1)
                similarity = cosine_similarity(query_embedding, file_embeddings)[0][0]
                print(f"[DEBUG] Similarity score for {original_filename}: {similarity:.4f}")

                # 7. Match Threshold
                if similarity > 0.25:  # Adjusted threshold for better results
                    thread_local.matches.append({
                        "file_name": original_filename,
                        "similarity_score": float(similarity),
                        "file_path": blob_data.get("file_path", "")
                    })

            except Exception as e:
                print(f"[ERROR] Processing {blob.name}: {str(e)}")
                continue

        # 8. Result Processing
        thread_local.matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_matches = thread_local.matches[:2]

        return jsonify({
            "matches": top_matches,
            "total_matches": len(thread_local.matches)
        }) if top_matches else jsonify({"message": "No matches found"}), 200

    except Exception as e:
        print(f"[CRITICAL] Search error: {str(e)}")
        return jsonify({"error": "Search failed"}), 500
    finally:
        # Cleanup thread-local storage
        if hasattr(thread_local, 'matches'):
            del thread_local.matches
        if hasattr(thread_local, 'processed_files'):
            del thread_local.processed_files


def extract_entities_from_ner(ner_results):
    """Extract meaningful entities from NER tagged results"""
    entities = []
    current_entity = []
    current_tag = None

    for line in ner_results:
        # Skip empty lines
        if not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) >= 2:
            word = parts[0]
            tag = parts[-1]  # Last part is the tag

            # Check if this is an entity
            if tag != 'O':
                # New entity type
                if tag.startswith('B-') or (current_tag and current_tag != tag):
                    # Save previous entity if exists
                    if current_entity:
                        entities.append(' '.join(current_entity))
                        current_entity = []

                    # Start new entity
                    current_entity.append(word)
                    current_tag = tag
                # Continuation of current entity
                elif tag.startswith('I-'):
                    current_entity.append(word)
                    current_tag = tag
            else:
                # End of an entity
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
                    current_tag = None

    # Add the last entity if exists
    if current_entity:
        entities.append(' '.join(current_entity))

    return entities


def perform_ner(query):
    # Azure OpenAI settings
    endpoint = "https://weez-openai-resource.openai.azure.com/"
    api_key = os.getenv('OPENAI_API_KEY')
    api_version = "2024-12-01-preview"
    deployment = "gpt-4o"  # Using GPT-4 as in the original code

    # Create client with API key
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )

    messages = [
        {
            "role": "system",
            "content": """You are a Named Entity Recognition (NER) expert. Your task is to analyze a given text and identify specific \
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

    # Call Azure OpenAI API for NER using new client syntax
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=0,
        max_tokens=500  # Increased max tokens to handle longer queries
    )

    # Extract and return the response content
    result = response.choices[0].message.content.strip()

    # Check if result contains actual NER output (looking for at least some tagged words)
    if not re.search(r'\b[BI]-[A-Z]+\b', result):
        print(f"NER result might not be in expected format: {result}")
        # Try to parse it anyway or return empty

    # Return the response content split by newlines
    return result.split("\n")

# [Keep the extract_entities_from_ner and perform_ner functions unchanged from original code]

if __name__ == '__main__':
    app.run(debug=True)
