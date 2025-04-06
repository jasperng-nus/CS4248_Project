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

# Set your OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY)

def load_scicite_data(json_path="test.jsonl"):
    """Load SciCite dataset and extract citation texts."""
    citations = []
    
    with open(json_path, 'r') as f:
        for line in f:  # Read JSONL file line by line
            entry = json.loads(line.strip())  # Convert each line into a dictionary
            
            citations.append({
                "citation_text": entry["string"],  # The text of the citation
                "label": entry["label"],  # The label for classification (e.g., method, result)
                "section_name": entry.get("sectionName", ""),  # The section name (if available)
                "id": entry["id"]
            })
    
    return citations

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    embeddings = []
    for e in response['data']:
        embeddings.append(e['embedding'])

        #float 32 to work with faiss as is design to work with faiss
    return np.array(embeddings).astype("float32") 


def build_faiss_index(citations):
    """Build FAISS index from SciCite citation embeddings."""

    texts = []

    # Combine citation text, label, and section name
    for citation in citations:
        combined_text = f"{citation['citation_text']} [Label: {citation['label']}] [Section: {citation['section_name']}]"
        texts.append(combined_text)

    print("Generating embeddings for SciCite citations...")  
    embeddings = get_openai_embeddings(texts)  # Get embeddings

    d = embeddings.shape[1]  # Embedding dimension (1536 for OpenAI)
    index = faiss.IndexFlatL2(d)  # L2 distance search
    index.add(embeddings)  # Add embeddings to FAISS index

    return index, embeddings

def retrieve_similar_citations(query, index, citations, k=5):    
    # Use the router to determine the label
    router = Router(query)
    predicted_label = router.route()  # Assuming router returns the predicted label

    # Format query with predicted label
    combined_query = f"{query} [Label: {predicted_label}]"

    # Generate query embedding
    # query_embedding = get_openai_embeddings([combined_query])[0]
    path_to_data = os.path.join('..', 'data', 'embeddings', 'test_embeddings.npy')
    query_embedding = np.load(path_to_data)
    
    # Perform search

    # k: This is the number of nearest neighbors to retrieve (i.e., how many 
    # similar citations you want to get). The search method returns the
    # k most similar entries in the index.
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve corresponding citation data

    # Optionally, use the parsed response (section) to filter or sort results
    # For example, if the section is 'Method', you might prioritize 'Method' citations
    # Retrieve corresponding citation data
    similar_citations = []
    for i in range(k):
        citation = citations[indices[0][i]]  # Get citation by index
        similarity_score = 1 - distances[0][i] / distances[0].max()  # Normalize score

        similar_citations.append({
            "citation_text": citation["citation_text"],
            "label": citation["label"],
            "section_name": citation["section_name"],
            "id": citation["id"],
            "similarity_score": similarity_score
        })
    
    return similar_citations

def get_query_embedding(query_text):
        """Convert a text query to embedding using OpenAI's model"""
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

    index = faiss.read_index(os.path.join(os.path.join('..', 'data', 'embeddings'), 'faiss_index.idx'))

    query_embedding = get_query_embedding(query)

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Convert indices to list
    indices_list = indices[0].tolist()
    
    # Find rows in DataFrame with matching FAISS indices
    # This approach doesn't need pickle or any separate mapping file
    df = pd.read_csv(os.path.join(os.path.join('..', 'data', 'embeddings'), 'processed_data.csv'))
    # results = df[df['faiss_index']].copy()
    results = df.iloc[indices[0]].copy()

    # Add similarity scores
    results['similarity_score'] = 1 - distances[0] / (distances[0].max() or 1)
        
    return results


if __name__ == "__main__":
    # Load the SciCite data
    # citations = load_scicite_data()

    # Build FAISS index from the citations
    # index, embeddings = build_faiss_index(citations)

    # Example query
    query = "What are the effects of climate change on coral reefs?"
    
    # Retrieve similar citations based on the query
    print(retrieve(query))
    # similar_citations = retrieve_similar_citations(query, index, citations, k=5)

    # Print results
    # for citation in similar_citations:
    #     print(f"Citation: {citation['citation_text']}")
    #     print(f"Similarity Score: {citation['similarity_score']}")
    #     print(f"Label: {citation['label']}")
    #     print(f"Section: {citation['section_name']}")
    #     print(f"Query Section: {citation['query_section']}")
    #     print()
