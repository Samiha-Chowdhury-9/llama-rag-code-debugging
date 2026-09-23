# Llama 3.2 + RAG for Automated Code Debugging

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E)
![Llama](https://img.shields.io/badge/Llama%203.2-3B%20Instruct-blue)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-purple)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green)
![RAG](https://img.shields.io/badge/Architecture-RAG-orange)

An experimental code-debugging system built with **Llama 3.2 3B Instruct**, **LoRA fine-tuning**, and **Retrieval-Augmented Generation (RAG)**.

The project explores whether retrieving similar historical buggy/fixed code examples can improve the ability of a fine-tuned LLM to generate corrected source code.

------

## Project Overview

The project contains two main stages:

1. **Fine-tuning Llama 3.2 3B Instruct**
   - Fine-tuned for automated code correction.
   - Parameter-efficient training using LoRA.
   - Evaluated using Exact Match, ROUGE, and SacreBLEU.
2. **RAG-enhanced Code Debugging**
   - Training examples are converted into vector embeddings.
   - Similar buggy-code examples are retrieved using FAISS.
   - Retrieved buggy/fixed examples are added to the LLM prompt.
   - The fine-tuned Llama model generates the final correction.

------

## Architecture

```text
                    ┌──────────────────────┐
                    │      Buggy Code      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ SentenceTransformer  │
                    │ all-MiniLM-L6-v2     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     FAISS Index      │
                    │ Semantic Retrieval   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Top-K Similar Buggy  │
                    │ + Fixed Examples     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     RAG Prompt       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Fine-tuned Llama 3.2 │
                    │      3B Instruct     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Corrected Code    │
                    └──────────────────────┘
```

------

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Llama 3.2 3B Instruct
- PEFT
- LoRA
- SentenceTransformers
- FAISS
- Hugging Face Datasets
- ROUGE
- SacreBLEU
- Pandas
- NumPy
- Matplotlib
- Kaggle GPU environment

------

## Dataset

The experiments use separate training, validation, and test splits.

| Split      | Samples |
| ---------- | ------- |
| Training   | 3,216   |
| Validation | 1,072   |
| Test       | 1,072   |

Each sample includes information such as:

- `buggy_code`
- `fixed_code`
- `bug_type`
- `language`

The **3,216 training samples** are also used as the retrieval knowledge base for the RAG pipeline.

Dataset files are not included in this repository. 

------

## LLM Fine-Tuning

The base model used in the experiment is:

**`meta-llama/Llama-3.2-3B-Instruct`**

LoRA was used instead of full-model fine-tuning to reduce the number of trainable parameters.

### LoRA Configuration

```text
Rank (r):              8
Alpha:                 16
Dropout:               0.0

Target modules:
- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj
```

### Training Configuration

```text
Epochs:                         3
Per-device batch size:          1
Gradient accumulation steps:    16
Effective batch size:           16
Learning rate:                  2e-4
Maximum sequence length:        1024
Precision:                      FP16
Checkpoint interval:            200 steps
Evaluation interval:            200 steps
```

The best model checkpoint was selected using validation loss.

------

## RAG Pipeline

The second experiment integrates RAG with the fine-tuned model.

### 1. Knowledge Base

The 3,216 training examples are used to create a retrieval knowledge base containing:

```text
Buggy Code
Fixed Code
Bug Type
Programming Language
```

### 2. Embeddings

Code examples are embedded using:

**`sentence-transformers/all-MiniLM-L6-v2`**

### 3. Vector Search

The embeddings are stored in a FAISS:

```python
faiss.IndexFlatIP
```

Embedding vectors are L2-normalized before similarity search.

### 4. Retrieval

For each new buggy-code sample, the system retrieves the:

**Top 4 most similar examples**

from the training knowledge base.

### 5. Generation

Retrieved examples are inserted into the prompt before sending it to the LoRA fine-tuned Llama model.

The LoRA adapter is merged with the base model before RAG inference.

------

## Experimental Results

A pilot evaluation was performed on the **same first 50 test examples** for both approaches.

| Metric      | LoRA Fine-Tuned | LoRA + RAG |
| ----------- | --------------- | ---------- |
| Exact Match | 36.00%          | **62.00%** |
| ROUGE-1     | **0.9527**      | 0.8470     |
| ROUGE-2     | **0.9421**      | 0.8484     |
| ROUGE-L     | **0.9474**      | 0.8494     |
| SacreBLEU   | **83.73**       | 59.71      |

RAG increased Exact Match from **36% to 62%** on this 50-example pilot, while ROUGE and SacreBLEU scores decreased.

This suggests that retrieval changed generation behavior in a way that produced more exact corrections for this sample, while reducing overall token-overlap scores.

Because this evaluation contains only 50 test examples, the results should be treated as a **pilot experiment rather than a full benchmark**.

------

## RAG Performance by Bug Type

| Bug Type                | Exact Match |
| ----------------------- | ----------- |
| Algorithm               | 33.33%      |
| Assignment              | 80.00%      |
| Build / Package / Merge | 50.00%      |
| Checking                | 50.00%      |
| Timing / Serialization  | 100.00%     |
| Multiple Error          | 63.64%      |

These values are based on the same 50-example pilot test.

------

## Repository Structure

```text
llama-rag-code-debugging/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_llama32_lora_finetuning.ipynb
│   └── 02_rag_code_debugging.ipynb
│
├── data/
│   └── README.md
│
└──  results/
   ├── baseline/
   │   ├── validation_results.csv
   │   ├── validation_plot.png
   │   └── validation_summary.txt
   │
   └── rag/
       ├── rag_results.csv
       ├── rag_validation_plot.png
       └── summary.txt

```



------

## Skills Demonstrated

This project demonstrates practical experience with:

- Large Language Models
- LLM fine-tuning
- Parameter-Efficient Fine-Tuning
- LoRA
- Retrieval-Augmented Generation
- Semantic embeddings
- Vector databases and similarity search
- FAISS
- Prompt construction
- Transformer inference
- Model evaluation
- NLP evaluation metrics
- GPU-based model training
- Experiment analysis

------

## Limitations

- The reported comparison currently uses only 50 test samples.
- RAG improves Exact Match in the pilot experiment but decreases ROUGE and SacreBLEU.
- Retrieval quality depends on similarity within the training knowledge base.
- The current system uses a relatively simple FAISS similarity retrieval strategy.
- Additional full-test evaluation and retrieval ablation experiments would be required for stronger conclusions.

------

## Future Work

Possible extensions include:

- Evaluation on the complete test set
- Comparison of different embedding models
- Different values of `k` for retrieval
- Retrieval reranking
- Hybrid lexical + semantic retrieval
- RAG ablation experiments
- Retrieval-quality analysis
- Error analysis across programming languages and bug categories

------

## Note

This repository is intended to demonstrate practical experimentation with **LLM fine-tuning, RAG, embeddings, vector retrieval, and automated code debugging**.

Large model checkpoints and trained model weights are intentionally excluded from the repository.
