# Mejurix Medical-Legal Embedding Model

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/mejurix-medicallegal-embedder-0329/)
[![Python versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.org/project/mejurix-medicallegal-embedder-0329/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A specialized embedding model for medical and legal document analysis. Using an NER (Named Entity Recognition) based approach to generate text embeddings in medical-legal contexts.

## Overview

This package provides a specialized embedding model for effectively analyzing documents in the medical and legal domains. It leverages NER (Named Entity Recognition) technology to recognize medical and legal entities, thereby generating high-quality text embeddings.

## Installation

### Installation via pip (after publication on PyPI)

```bash
pip install mejurix-medicallegal-embedder-0329
```

### Installation from source

```bash
git clone https://github.com/mejurix/medicallegal-embedder.git
cd medicallegal-embedder
pip install -e .
```

## Model Download

When you first install the package, the model is not automatically included. You can download the model using the following command:

```python
from mejurix_medicallegal_embedder import MejurixMedicalLegalEmbedder

# Download the model
MejurixMedicalLegalEmbedder.download_model()
```

Or use the command line interface:

```bash
medicallegal-embedder download
```

## Usage

### Using the Python API

```python
from mejurix_medicallegal_embedder import MejurixMedicalLegalEmbedder

# Initialize the model
embedder = MejurixMedicalLegalEmbedder()

# Generate text embeddings
text = "The patient was diagnosed with L3 vertebral fracture, and a compensation claim is in progress."
embedding = embedder.encode(text)

# Calculate similarity between documents
text1 = "Diagnosed with L3 spinal fracture."
text2 = "Compensation is needed for lumbar injury."
similarity = embedder.compute_similarity(text1, text2)
print(f"Similarity: {similarity:.4f}")

# Utilize entity type information
entity_type_id = 1  # DISEASE
embedding_with_entity = embedder.encode(text, entity_type_id)

# Check supported entity types
entity_types = embedder.get_entity_types()
print(entity_types)
```

### Batch Processing

```python
# Batch embedding
texts = [
    "The patient was diagnosed with L3 vertebral fracture",
    "Neck pain persisted after the accident",
    "Clinical test results were within normal range"
]
embeddings = embedder.encode(texts, batch_size=8)
print(f"Number of embeddings: {len(embeddings)}")

# Batch similarity calculation
texts1 = ["The patient was diagnosed with a fracture", "Medication was prescribed to the patient"]
texts2 = ["Diagnosed with fracture", "Drug treatment initiated"]
similarity_matrix = embedder.batch_compute_similarity(texts1, texts2)
print(similarity_matrix)
```

### Using Preprocessing Utilities

```python
from mejurix_medicallegal_embedder.utils.preprocessing import preprocess_for_embedding, batch_preprocess

# Preprocess individual text
text = "Pt was Dx with L3 fracture."
processed_text = preprocess_for_embedding(text, entity_type='medical')
print(processed_text)

# Batch preprocessing
texts = [
    "Pt was Dx with L3 fracture.",
    "plf showed similar injury pattern w/ previous accident."
]
entity_types = ['medical', 'legal']
processed_texts = batch_preprocess(texts, entity_types)
print(processed_texts)
```

### Using the Command Line Interface (CLI)

Text encoding:

```bash
# Encode a single text
medicallegal-embedder encode --text "The patient was diagnosed with L3 vertebral fracture" --output "embedding.npy"

# Encode from a file
medicallegal-embedder encode --file "input.txt" --line-by-line --output "embeddings.npy"

# Apply preprocessing
medicallegal-embedder encode --text "Pt was Dx with L3 fracture." --preprocess --entity-type "medical" --output "embedding.npy"
```

Similarity calculation:

```bash
# Calculate similarity between two texts
medicallegal-embedder similarity \
  --text1 "The patient was diagnosed with L3 vertebral fracture" \
  --text2 "Diagnosed with L3 spinal fracture"

# Specify threshold and judgment
medicallegal-embedder similarity \
  --text1 "The patient was diagnosed with L3 vertebral fracture" \
  --text2 "Diagnosed with L3 spinal fracture" \
  --threshold 0.75
```

## Model Features

- **Base Model**: medicalai/ClinicalBERT
- **Embedding Dimension**: 256
- **NER-based Approach**: Optimized for recognizing medical and legal entities
- **Specialized Domain**: Medical-legal document analysis

## Training Results Summary

### Model Information
- **Class**: `NERClinicalBertEmbedder`
- **Base Model**: medicalai/ClinicalBERT
- **Embedding Dimension**: 768 → 256
- **Dropout**: 0.5

### Performance Metrics
- **Final Accuracy**: 0.6082 (validation dataset)
- **Overall Average Similarity across Relationship Types**: 0.8168

### Average Similarity by Relationship Type
| Relationship Type | Average Similarity |
|----------|-----------|
| DISEASE_MEDICATION | 0.8514 |
| SEVERITY_PROGNOSIS | 0.8381 |
| SEVERITY_COMPENSATION | 0.8348 |
| DISEASE_TREATMENT | 0.8359 |
| DIAGNOSIS_TREATMENT | 0.8222 |
| LEGAL_SIMILAR_MEDICAL_DIFFERENT | 0.8236 |
| TREATMENT_OUTCOME | 0.8103 |
| OUTCOME_SETTLEMENT | 0.7951 |
| MEDICAL_SIMILAR_LEGAL_DIFFERENT | 0.7812 |
| SYMPTOM_DISEASE | 0.7819 |

### Key Strengths
1. **Medical-Legal Cross-Concept Connection**: Understanding the connection between medical assessments and legal compensation (Similarity: 0.8348)
2. **Medical Terminology Relationship Recognition**: Recognizing equivalence across various medical expressions (Similarity: 0.8414)
3. **Causality Understanding**: Accurately recognizing cause-and-effect relationships (Similarity: 0.8236)

### Areas for Improvement
1. **Detailed Medical Terminology Differentiation**: Need to improve recognition of severity differences
2. **Temporal Context Understanding**: Need to enhance recognition of temporal difference importance
3. **Negation Handling**: Need to more accurately capture meaning reversal in negations

## License

This project is distributed under the MIT License. See the LICENSE file for details.

## Contact

If you have any issues or suggestions, please register an issue.

## Citation

If you use this model, please cite it as follows:

```
@software{mejurix_medicallegal_embedder,
  author = {Mejurix},
  title = {Mejurix Medical-Legal Embedding Model},
  year = {2025},
  version = {0.1.0}
}
``` 