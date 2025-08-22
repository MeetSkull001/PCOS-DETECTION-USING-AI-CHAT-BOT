import os
import time
import json
import numpy as np
import requests
from requests.exceptions import ConnectionError, Timeout
from sklearn.metrics.pairwise import cosine_similarity

# Set your API key for Mistral
api_key = "0Q6YuYqsFPpIN7G2ZjlQDZHOJPCppbbI"
model = "mistral-embed"

# Define the Mistral API URL
API_URL = "https://api.mistral.ai/v1/embeddings"
embedding_file = 'embeddings.json'

# Load the JSON data
with open('chat.json', 'r') as file:
    categories = json.load(file)

# Dictionary to store precomputed embeddings
question_embeddings = {}

# Load existing embeddings from the file if it exists
def load_embeddings():
    if os.path.exists(embedding_file):
        with open(embedding_file, 'r') as f:
            return json.load(f)
    else:
        return {}

# Save embeddings to a file
def save_embeddings(embeddings):
    """Save the embeddings to a file after converting them to a serializable format."""
    # Convert numpy arrays to lists to make them JSON serializable
    embeddings = {key: embedding.tolist() for key, embedding in embeddings.items() if embedding is not None}
    
    with open(embedding_file, 'w') as f:
        json.dump(embeddings, f)

# Get embedding with retries and exponential backoff for rate limiting
def get_embedding(text, retries=5):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": model
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, json=data, headers=headers, timeout=5)
            if response.status_code == 200:
                embedding = response.json()['data'][0]['embedding']
                return np.array(embedding)
            elif response.status_code == 429:  # Handle rate limit
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except (ConnectionError, Timeout):
            wait_time = 2 ** attempt
            print(f"Connection error. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    print("Failed to get embedding after retries.")
    return None

# Precompute embeddings for all questions in categories
def precompute_question_embeddings():
    """Precompute and save the embeddings for all example questions."""
    for category in categories:
        category_name = category["CATEGORY"]
        example_questions = category["EXAMPLE_QUESTIONS"]
        
        for question in example_questions:
            if question not in question_embeddings:
                embedding = get_embedding(question)
                if embedding is not None:
                    question_embeddings[question] = embedding

    # Save the embeddings after they are computed
    save_embeddings(question_embeddings)
    print("Embeddings precomputed and saved.")

# Find the best matching category
def find_best_category(input_text):
    """Find the best matching category for the input question based on cosine similarity."""
    input_embedding = get_embedding(input_text)
    if input_embedding is None or np.isnan(input_embedding).any():
        print("Invalid input embedding.")
        return None

    best_category = None
    highest_similarity = 0.8  # Similarity threshold

    # Iterate over each category in the JSON data
    for category in categories:
        category_name = category["CATEGORY"]
        example_questions = category["EXAMPLE_QUESTIONS"]
        
        # Iterate over example questions
        for question in example_questions:
            question_embedding = question_embeddings.get(question)
            if question_embedding is None:
                question_embedding = get_embedding(question)
                if question_embedding is not None:
                    question_embeddings[question] = question_embedding

            if question_embedding is None or np.isnan(question_embedding).any():
                continue  # Skip if there's an invalid embedding

            similarity = cosine_similarity([input_embedding], [question_embedding])[0][0]
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_category = {
                    "CATEGORY": category_name,
                    "SIMILARITY_SCORE": similarity,
                }

    return best_category

def processQuery (best_match, userQuery):
    with open('intent.json', 'r') as file:
        data = json.load(file)
    
    for item in data:
        if item["CATEGORY"] == best_match["CATEGORY"]:
            promptFilePath = item["PROMPT"]
    
    with open(promptFilePath, 'r') as file:
        base_prompt = file.read() 
        
    
    if(best_match["CATEGORY"] == "PCOS_DIET_PLANS" or best_match["CATEGORY"] == "PCOS_WORKOUT_PLANS"):
        formatted_prompt = f"""[INST]
        
        {base_prompt}

        [/INST]
        """
    else:
        formatted_prompt = f"""[INST]
        
        {base_prompt}
        {userQuery}

        [/INST]
        """
    url = "https://api.mistral.ai/v1/chat/completions" 
    
    # Define the JSON payload
    payload = {
        "model": "mistral-large-latest",
        "temperature": 0.1,
        "max_tokens": 128000,
        "messages": [
            {
                "role": "user",
                "content": f"{formatted_prompt}",
            }
        ],
        "response_format": {
            "type": "text"
        },
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"  # Replace with your actual API key
    }

    response = requests.post(url, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Process the response data
        response_data = response.json()
        print("API Response:", response_data["choices"][0]["message"]["content"])
    else:
        # Print the error if the request failed
        print("Request failed with status code:", response.status_code)
        print("Response:", response.text)
    
# Main function to run the program
def main():
    # Load existing embeddings from the file
    global question_embeddings
    question_embeddings = load_embeddings()

    if not question_embeddings:
        print("No embeddings found, fetching from API...")
        precompute_question_embeddings()
    else:
        print(f"Loaded {len(question_embeddings)} precomputed embeddings.")

    while True:
        userQuery = input("Enter a question (type 'quit' or 'exit' to stop): ").strip()

        if userQuery.lower() in ['quit', 'exit']:
            print("Exiting the program.")
            break

        best_match = find_best_category(userQuery)

        if best_match:
            print("Best Matching Category:", best_match["CATEGORY"])
            time.sleep(2)
            processQuery(best_match, userQuery)
        else:
            print("Out Of Bound")

if __name__ == "__main__":
    main()
