import json
import pandas as pd
import numpy as np
import os
from openai import OpenAI
import faiss

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from llm.llm import LLMEngine

# --- Configuration ---
BATCH_SIZE = 100
MODEL = "text-embedding-3-small"

def load_scicite():
    scicite_sentences = []
    path_to_data = os.path.join(project_root, 'cleaning', 'AugmentedData', 'augmented_data.jsonl')

    with open(path_to_data, "r", encoding='utf-8') as f: 
        for line in f:
            scicite_sentences.append(json.loads(line))
          

    return pd.DataFrame(scicite_sentences)

def preprocess_sentence(df):
    # Extract only relevant fields, starting with citation sentences
    df['text_for_embeddings'] = "Claim: " + df['string'].fillna('')
    
    # Include section name and labels if present
    if 'sectionName' in df.columns:
        df['text_for_embeddings'] += " Section Name: " + df['sectionName'].fillna('')
    if 'sectionName' in df.columns:
        df['text_for_embeddings'] += " Classification Label: " + df['label'].fillna('')
    return df

def generate_embeddings(df):
    all_embeddings = []
    client = LLMEngine()

    for i in range(0, len(df), BATCH_SIZE):
        batch = df['text_for_embeddings'][i:i+BATCH_SIZE].tolist()
        
        try:
            response = client.generate_embeddings(
                query_text=batch,
                model=MODEL,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        except Exception as e:
            print(e)
            print(f"Error generating embeddings for batch {i}")

    embeddings = np.array(all_embeddings).astype('float32')
    path_to_folder = os.path.join(project_root, 'rag', 'data', 'embeddings')
    np.save(os.path.join(path_to_folder, 'augmented_data_embeddings.npy'), embeddings)
    
    return embeddings

def store_embeddings(embeddings):
    # Create FAISS database and store embeddings
    dimension = embeddings.shape[1] 
    faiss_structure = faiss.IndexFlatL2(dimension)
    faiss_structure.add(embeddings)
    # Save FAISS database
    faiss.write_index(faiss_structure, os.path.join(project_root, 'rag', 'data', 'embeddings', 'faiss_index.idx'))

def run_pipeline():
    df = load_scicite()
    df = preprocess_sentence(df)

    # Save the processed data for retrieval
    df.to_csv(os.path.join(project_root, 'rag', 'data', 'embeddings','processed_data.csv'), index=False)

    # Generate embeddings, then store them and build the FAISS index
    embeddings = generate_embeddings(df)
    augmented_embeddings_filepath = os.path.join(project_root, 'rag', 'data', 'embeddings', 'augmented_data_embeddings.npy')
    embeddings = np.load(augmented_embeddings_filepath) 
    store_embeddings(embeddings)

if __name__ == "__main__":
    run_pipeline()