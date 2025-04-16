import json
import numpy as np
import faiss
from openai import OpenAI
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
# from routing.router import Router
# from .llm import LLM
from routing.router import Router
import pandas as pd

MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_query_embedding(query_text):
        # Convert a text query to embedding using OpenAI's model
        try:
            response = client.embeddings.create(
                input=query_text,
                model=MODEL
            )
            return np.array([response.data[0].embedding]).astype('float32')
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            raise

def retrieve(query, top_k=5):
    # Path to the FAISS index and processed CSV file
    embeddings_folder = os.path.join('..', 'data', 'embeddings')
    index_path = os.path.join(embeddings_folder, 'faiss_index.idx')
    data_path = os.path.join(embeddings_folder, 'processed_data.csv')

    # Load FAISS index and processed data
    index = faiss.read_index(index_path)
    df = pd.read_csv(data_path)

    # Use Router to get the predicted label for additional context
    router = Router(query)
    predicted_label = router.route()

    # Create an augmented query that includes the predicted label.
    # This helps in generating an embedding with a similar context as your preprocessed data.
    augmented_query = f"Claim: {query} Classification Label: {predicted_label}"
    query_embedding = get_query_embedding(augmented_query)

    # Search FAISS index for top_k results
    # distances: the Euclidean distances between the query and the top-k results
    distances, indices = index.search(query_embedding, top_k)
    # Retrieves the rows from the DataFrame corresponding to the indices 
    # obtained from the FAISS search
    results = df.iloc[indices[0]].copy()

    # new column 'similarity_score'
    # Normalize similarity score: 
    # Smaller distances (higher similarity) should result in a score closer to 1 (best),
    # Larger distances (lower similarity) should result in a score closer to 0 (worst)
    results['similarity_score'] = 1 - distances[0] / (distances[0].max() or 1)
    
    # add the predicted label to the results for reference
    results['predicted_label'] = predicted_label

    # results['citing_paper_id'] = df['citingPaperId']
    return results

if __name__ == "__main__":
    # Example query
    query = "What are the effects of climate change on coral reefs?"
    
    # Retrieve and print the results
    print(retrieve(query))
