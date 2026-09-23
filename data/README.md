# Data

This folder contains the datasets used for training, validation, and testing of the Llama-based code debugging model.

## Dataset Structure

The project expects the following main fields:

| Column       | Description                             |
| ------------ | --------------------------------------- |
| `buggy_code` | Source code containing an error or bug  |
| `fixed_code` | Corrected version of the buggy code     |
| `bug_type`   | Category/type of the programming bug    |
| `language`   | Programming language of the code sample |

## Dataset Splits

The experiments used separate training, validation, and test datasets.

- Training samples: **3,216**
- Validation samples: **1,072**
- Test samples: **1,072**

The training data was also used as the retrieval knowledge base for the RAG pipeline.

## Use in the Project

The dataset is used for:

- Fine-tuning Llama 3.2 3B Instruct with LoRA
- Validation and model evaluation
- Building SentenceTransformer embeddings
- Creating the FAISS vector index
- Retrieving similar buggy/fixed code examples for RAG
- Comparing standard LLM inference with RAG-assisted inference

## Data Files

Large or externally sourced dataset files are not included in this repository.

Expected structure:

```text
data/
├── train.csv
├── validation.csv
├── test.csv
└── README.md
```

Place the required dataset files in this directory before running the notebooks.

## Note

Only use and redistribute the dataset according to its original license and usage permissions.