from flask import Flask, request, jsonify
import os
import json
import numpy as np
from azure.storage.blob import BlobServiceClient
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
import threading
import re

# Initialize Flask app
app = Flask(__name__)

# Azure Blob Storage setup
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_METADATA_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

EMBEDDINGS_CONTAINER = "weez-files-embeddings"

# Thread-local storage to ensure data isolation between concurrent requests
thread_local = threading.local()

@app.route('/search', methods=['POST'])
def search():
    try:
        # Initialize thread-local storage for this request
        thread_local.matches = []
        thread_local.processed_files = set()
        
        # Step 1: Parse the query and user ID from the request
        data = request.get_json()
        query = data.get('query')
        user_id = data.get('user_id')

        if not query or not user_id:
            return jsonify({"error": "Query and user_id are required."}), 400

        # Step 2: Perform NER on the query
        ner_results = perform_ner(query)
        
        # Step 3: Extract entities from NER results
        entities = extract_entities_from_ner(ner_results)
        
        # Log entities for debugging
        print(f"Extracted entities: {entities}")
        
        if not entities:
            # Fallback: use the whole query if no entities found
            entities = [query]

        # Step 3: Generate embeddings for the extracted entities using Azure OpenAI client
        query_text = " ".join(entities)
        
        # Azure OpenAI settings for embeddings
        endpoint = "https://weez-openai-resource.openai.azure.com/"
        api_key = os.getenv('OPENAI_API_KEY')
        api_version = "2024-12-01-preview"
        embedding_model = "text-embedding-3-large"
        
        # Create client with API key for embeddings
        embedding_client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        
        # Generate embeddings using the client
        query_embedding_response = embedding_client.embeddings.create(
            model=embedding_model,
            input=query_text
        )
        
        # Extract the embedding vector
        query_embedding = np.array(query_embedding_response.data[0].embedding).reshape(1, -1)
        
        # Close the client
        embedding_client.close()

        # Step 4: Access all embeddings for the user
        embeddings_container_client = blob_service_client.get_container_client(EMBEDDINGS_CONTAINER)
        
        # IMPORTANT: Strong user isolation - only access files with this exact user_id prefix
        user_prefix = f"{user_id}/"
        
        # Safety check - ensure user_id doesn't contain path traversal characters
        if '../' in user_id or '/' in user_id.replace('@', '').replace('.', ''):
            return jsonify({"error": "Invalid user_id format."}), 400
            
        # List only blobs that belong to this specific user
        blobs = embeddings_container_client.list_blobs(name_starts_with=user_prefix)
        
        # Debug: List all blob names
        blob_names = []
        for blob in blobs:
            blob_names.append(blob.name)
        print(f"Found blobs for user {user_id}: {blob_names}")
        
        # If no blobs found, return early
        if not blob_names:
            return jsonify({"message": "No files found for this user."}), 404
        
        # Reset blob iterator
        blobs = embeddings_container_client.list_blobs(name_starts_with=user_prefix)

        for blob in blobs:
            # Additional security check - verify this blob truly belongs to this user
            if not blob.name.startswith(user_prefix):
                continue
                
            # Avoid processing the same file multiple times
            if blob.name in thread_local.processed_files:
                continue
                
            thread_local.processed_files.add(blob.name)
            
            try:
                blob_client = embeddings_container_client.get_blob_client(blob.name)
                blob_content = blob_client.download_blob().readall().decode('utf-8')
                embedding_data = json.loads(blob_content)

                file_name = embedding_data.get("file_name")
                file_path = embedding_data.get("file_path")
                
                # Debug output
                print(f"Processing file: {file_name}, path: {file_path}")
                
                # Verify the file_path also belongs to this user
                if not file_path.startswith(user_prefix):
                    continue
                    
                # Extract embeddings - handle different possible formats
                file_embeddings = embedding_data.get("embeddings")
                
                # Debug output for embeddings format
                print(f"Embeddings type: {type(file_embeddings)}")
                
                # Handle case where embeddings might be nested or in different format
                if isinstance(file_embeddings, list) and isinstance(file_embeddings[0], list):
                    # If it's a list of lists (chunks), concatenate or use the first chunk
                    file_embeddings = np.array(file_embeddings[0])
                elif isinstance(file_embeddings, list):
                    file_embeddings = np.array(file_embeddings)
                else:
                    # Skip if embeddings are not in expected format
                    print(f"Unexpected embeddings format for {file_name}")
                    continue

                # Step 5: Compute cosine similarity
                similarity = cosine_similarity(query_embedding, file_embeddings.reshape(1, -1))[0][0]
                
                # Debug output
                print(f"Similarity score for {file_name}: {similarity}")

                # Add to thread-local matches list
                thread_local.matches.append({
                    "file_name": file_name,
                    "file_path": file_path,
                    "similarity_score": float(similarity)  # Convert numpy float to Python float for JSON serialization
                })
            except Exception as e:
                # Log the error but continue processing other files
                print(f"Error processing blob {blob.name}: {str(e)}")
                continue

        # Sort matches by similarity score in descending order
        thread_local.matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Get top two matches (or fewer if not enough matches)
        top_matches = thread_local.matches[:2] if len(thread_local.matches) >= 2 else thread_local.matches
        total_matches = len(thread_local.matches)
        
        # Create response - using local variables to avoid any shared state
        response_data = {
            "matches": top_matches,
            "total_matches_found": total_matches
        }
        
        if top_matches:
            return jsonify(response_data), 200
        else:
            return jsonify({"message": "No matching files found."}), 404

    except Exception as e:
        # For security, don't expose detailed error information to the client
        print(f"Search error: {str(e)}")
        return jsonify({"error": "An error occurred during search."}), 500
    finally:
        # Clean up thread-local storage to prevent memory leaks
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

if __name__ == '__main__':
    app.run(debug=True)
