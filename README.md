
# CS4248 Team 17

# Cognitia: A Retrieval-Augmented System for Evidence-Grounded Scientific Question Answering

## Abstract

We address misinformation in AI‑generated scientific content by building a two‑stage Retrieval‑Augmented Generation (RAG) system. First, we classify the citation intent of a user’s query (background, method, or result) using a prompt‑steered GPT‑4o‑mini. Next, we retrieve supporting evidence from a combined SciCite + PubMed corpus via dense embeddings (OpenAI text‑embedding‑3‑small) and FAISS. Finally, the LLM generates an answer grounded in the retrieved citations. In our tests, this pipeline achieves 76.3 % accuracy versus 50.9 % for the same LLM without external context.

---

## Table of Contents

1. [Features](#features)  
2. [Repository Structure](#repository-structure)
3. [Key Files & Notebooks](#key-files--notebooks)  
4. [Installation](#installation)  
5. [Training & Indexing](#training--indexing)  
6. [Usage](#usage)  
7. [References](#references)
8. [Acknowledgement](#acknowledgement)   

---

## Features

- **Two‑stage pipeline**  
  1. **Intent classification** via prompt‑steered GPT‑4o‑mini  
  2. **Dense retrieval** with OpenAI text‑embedding‑3‑small + FAISS  
- **Cross‑domain corpus**  
  - **SciCite**: 8 243 curated citation sentences  
  - **PubMed‑200k RCT** (downsampled to 3 538 entries)  
- **Evidence‑grounded answers** with inline citations (paper IDs + similarity scores)  
- **Fallback** to LLM-only response when no evidence meets the threshold  
- **Extensive evaluation** on 288 manually curated scientific queries  

---

## Repository Structure

```text
.
├── .gitattributes
├── .gitignore
├── README.md
├── outputs.json                 # Sample model outputs
├── cleaning/                    # Data cleaning & corpora preparation
│   ├── AugmentedData/           # JSONL of merged SciCite + PubMed
│   ├── pubmed_20k/              # Raw + cleaned PubMed‑200k RCT splits
│   ├── acl-arc/                 # ACL‑ARC citation worthiness scaffolds
│   └── scicite-master/          # Modified SciCite codebase & configs
├── llm/                         # Prompt templates & LLM client wrappers
├── rag/                         # RAG pipeline components
│   ├── data/
│   │   ├── raw/                 # test.jsonl
│   │   └── embeddings/          # .npy embeddings + FAISS index + CSV
│   └── src/faiss/               # embeddings.py, retrieval.py, generation.py
├── routing/                     # Query intent routing & parser
└── testing/                     # Unit & integration tests
    └── test.py
```

--- 
## Key Files & Notebooks

- **`cleaning/CS4248 SciCite Project.ipynb`**  
  Exploratory data analysis of the SciCite dataset—data inspection, class distributions, and initial EDA to inform preprocessing and model design. 

- **`cleaning/scicite-master/scripts/train_local.py`**  
  Training pipeline for the citation‑intent classifier and merging SciCite + PubMed corpora based on a JSON config. 

- **`llm/llm.py`**  
  Wrapper for all LLM calls (GPT‑4o‑mini, GPT‑4.1‑mini): intent classification prompts, answer‑generation prompts, and API utilities. 

- **`rag/src/faiss/embeddings.py`**  
  Generates dense embeddings for each citation sentence (with metadata) using OpenAI’s text‑embedding‑3‑small. 

- **`rag/src/faiss/retrieval.py`**  
  Builds or loads a FAISS index and retrieves the top‑K most similar citations for a given query embedding. 

- **`rag/src/faiss/generation.py`**  
  Feeds retrieved evidence plus query into the LLM to produce a final, citation‑grounded answer. 

- **`routing/parser.py`** & **`routing/router.py`**  
  Parses command‑line arguments, steers the two‑stage pipeline (intent -> retrieval -> generation), and formats the output. 

- **`testing/test.py`**  
  End‑to‑end evaluation harness: runs all 288 test queries with and without RAG, compares LLM‑only vs. RAG‑augmented accuracy, and dumps per‑sample flags to `testing/data.json`. 

---
## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AjayArvind2207/CS4248_Project.git
   cd CS4248_Project
   ```

2. **Create & activate** a Python ≥ 3.9 virtual environment  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure** your OpenAI API key  
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
   Alternatively, create a `.env` file in the root of the project with the following content:
   ```bash
   OPENAI_API_KEY="your_api_key_here"

---
## Training & Indexing

1. **Generate embeddings**  
   ```bash
   python -m rag/src/embeddings.py
   ```

---

## Usage

```bash
python rag/src/generation.py --question "What cells are modulated by the immunomodulatory agent curcumin?"               
```

**Example output:**
**Answer:**
```
"The RAG has referenced claims from the following papers:
1. Furthermore, curcumin has been shown to be a potent immunomodulatory agent that can modulate the activation of T cells, B cells, macrophages, neutrophils, natural killer cells, and dendritic cells (Allam 2009). (Claim obtained from Citing Paper ID: af0fea7198d70421a86e0fa65e31bcc56803de5a. Similarity Score: 0.38)

Curcumin has been shown to modulate the activation of T cells, B cells, macrophages, neutrophils, natural killer cells, and dendritic cells."
```

---
## References

1. Thorp, H.H. (2023). _ChatGPT is fun, but not an author._ Science, 379(6630), 313.  
   https://doi.org/10.1126/science.adg7879  
2. Petroni, F. _et al._ (2019). _Language Models as Knowledge Bases?_ EMNLP.  
   https://arxiv.org/abs/1909.01066  
3. Nogueira, R., & Cho, K. (2019). _Passage Re‑ranking with BERT._ arXiv.  
   https://arxiv.org/abs/1901.04085  
4. Cohan, A. _et al._ (2019). _Structural Scaffolds for Citation Intent Classification._ ACL P19‑1102.  
5. Karpukhin, V. _et al._ (2020). _Dense Passage Retrieval for Open‑Domain QA._ EMNLP.  
   https://arxiv.org/abs/2004.04906  
6. Dernoncourt, F., & Lee, J. (2017). _PubMed 200k RCT: Sequential Sentence Classification._ IJCNLP Short Papers I17‑2052.  
   Dataset: https://www.kaggle.com/datasets/matthewjansen/pubmed-200k-rtc/data
---
## Acknowledgments

We would like to thank our mentor, Ong Han Wei, for his guidance and support throughout this project.
---
