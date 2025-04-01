# Pretrained Model Files

This directory contains the pretrained model files for the Mejurix Medical-Legal Embedding Model.

## Structure

- `embedder_config.json`: Configuration file for the embedding model
- `embedder_state.pt`: State file of the trained model
- `bert_model/`: Base BERT model files

## Model Information

- **Class**: `MejurixMedicalLegalEmbedder`
- **Base Model**: medicalai/ClinicalBERT
- **Embedding Dimension**: 768 → 256
- **Dropout**: 0.5
- **Special Feature**: NER (Named Entity Recognition) based embedding

## Usage

The model files in this directory are automatically loaded by the `MejurixMedicalLegalEmbedder` class.
Users can use the package without specifying a separate model path.

```python
from mejurix_medicallegal_embedder import MejurixMedicalLegalEmbedder

# Load the default model
embedder = MejurixMedicalLegalEmbedder()

# Or specify a custom model path
# embedder = MejurixMedicalLegalEmbedder(model_path="path/to/custom/model")
```

## Model Updates

The model files are updated along with package updates. It is recommended to update the package
regularly to use the latest performance model.

```bash
pip install --upgrade mejurix-medicallegal-embedder
``` 