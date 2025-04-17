# CS4248_Project

> **Abstract.** We address the issue of misinformation in AI‑generated scientific content by creating a two‑stage Retrieval‑Augmented Generation (RAG) system that answers queries with citations from research papers. First, we classify the citation intent of the user’s query (background, method, or result), then retrieve supporting evidence from our combined SciCite + PubMed corpus using dense embeddings and FAISS. Finally, an LLM (GPT‑4o‑mini) generates a natural‐language answer grounded in the retrieved citations. In testing, our RAG pipeline achieves a 76.3 % accuracy—far outperforming the 50.9 % baseline of the same LLM without external context.

---

## Features

- **Two‑stage pipeline**  
  1. **Intent classification** via prompt‑steered GPT‑4o‑mini  
  2. **Dense retrieval** with OpenAI text‑embedding‑3‑small + FAISS  
- **Cross‑domain corpus** combining  
  - **SciCite**: 8 243 human‑annotated citation sentences  
  - **PubMed‑200k RCT** (downsampled to 3 538)  
- **Evidence‑grounded answers** with inline citations (paper IDs + similarity scores)  
- **Fallback handling** when no evidence meets retrieval threshold  
- **Extensive evaluation** against a manually curated test set (288 queries)  

---

## Repository Structure
cleaning/  
    ├── AugmentedData/ – Custom preprocessed and augmented data  
    ├── pubmed_20k/ – Cleaned version of the PubMed dataset  
    ├── acl-arc/ – ACL-ARC citation worthiness datasets  
    └── scicite-master/ – Original SciCite codebase, modified for training  

rag/  
    ├── data/ – Processed data and embeddings for RAG  
    └── src/ – Custom FAISS-based retrieval and generation code  

routing/ – Message parsing and query routing logic  

testing/ – Unit tests and evaluation scripts  

outputs.json – JSON file storing model predictions/results  


