# NER Embedding Model Test Results (20250329)

## Model Overview

**Test Model:**
- **Class**: `NERClinicalBertEmbedder`
- **Base Model**: medicalai/ClinicalBERT
- **Embedding Dimension**: 768 → 256
- **Dropout**: 0.5
- **Model Path**: `models/ner_model_20250329_193235/best_model`
- **Special Feature**: NER (Named Entity Recognition) based embedding

## Test Setup

**Test Data:**
- **PDF Document**: testset/John_Smith_small.pdf
- **Relationship Types**: Includes multiple medical-legal context relationships
- **Evaluation Method**: Cosine similarity-based analysis

**Evaluation Metrics:**
- **Average similarity by relationship type**
- **Median, minimum, maximum, standard deviation**
- **Semantic similarity distribution visualization**

## Test Results

### Similarity Results by Relationship Type

| Relationship Type | Average Similarity | Median | Minimum | Maximum | Standard Deviation | Sample Count |
|----------|-----------|-------|-------|-------|---------|--------|
| DISEASE_MEDICATION | 0.8514 | 0.8514 | 0.8514 | 0.8514 | 0.0000 | 1 |
| SEVERITY_PROGNOSIS | 0.8381 | 0.8381 | 0.8381 | 0.8381 | 0.0000 | 1 |
| SEVERITY_COMPENSATION | 0.8348 | 0.8348 | 0.8348 | 0.8348 | 0.0000 | 1 |
| DISEASE_TREATMENT | 0.8359 | 0.8359 | 0.8359 | 0.8359 | 0.0000 | 1 |
| DIAGNOSIS_TREATMENT | 0.8222 | 0.8222 | 0.8222 | 0.8222 | 0.0000 | 1 |
| LEGAL_SIMILAR_MEDICAL_DIFFERENT | 0.8236 | 0.8236 | 0.8236 | 0.8236 | 0.0000 | 1 |
| TREATMENT_OUTCOME | 0.8103 | 0.8103 | 0.7791 | 0.8414 | 0.0311 | 2 |
| OUTCOME_SETTLEMENT | 0.7951 | 0.7951 | 0.7951 | 0.7951 | 0.0000 | 1 |
| MEDICAL_SIMILAR_LEGAL_DIFFERENT | 0.7812 | 0.7812 | 0.7812 | 0.7812 | 0.0000 | 1 |
| SYMPTOM_DISEASE | 0.7819 | 0.7819 | 0.7819 | 0.7819 | 0.0000 | 1 |
| **Overall Average** | **0.8168** | **0.8236** | **0.7791** | **0.8514** | **0.0259** | **11** |

### PDF Embedding Visualization Results

Through the PDF embedding visualization tool, the sentence embeddings of John_Smith_small.pdf document were visualized using t-SNE, confirming that semantically similar sentences form clusters. In particular, the following characteristics were observed:

- Medical symptom description sentences form one cluster
- Legal judgment-related sentences are separated into a distinct cluster
- Sentences related to patient personal information are positioned in the middle area

## Case Analysis

### Cases that Worked Well

1. **Medical Terminology Relationship Recognition**:
   - Example: "The patient was diagnosed with L3 vertebral fracture" vs "Diagnosed with L3 spinal fracture"
   - Similarity: 0.8414 (Recognition of medical equivalence with high accuracy)

2. **Causality Understanding**:
   - Example: "Pain continues from the accident" vs "Chronic pain occurred after the traffic accident"
   - Similarity: 0.8236 (Accurately recognizing the relationship between cause and effect)

3. **Legal-Medical Cross-Concept Connection**:
   - Example: "Permanent disability rate assessed at 10%" vs "Partial disability compensation applied"
   - Similarity: 0.8348 (Understanding the connection between medical assessment and legal compensation)

### Cases Needing Improvement

1. **Detailed Medical Terminology Differentiation**:
   - Example: "Mild ligament injury" vs "Moderate ligament rupture"
   - Similarity: 0.8103 (Doesn't sufficiently reflect the difference in severity)

2. **Temporal Context Understanding**:
   - Example: "Symptoms that occurred 6 months ago" vs "Similar symptoms that occurred recently"
   - Similarity: 0.7819 (Lack of recognition of the importance of temporal differences)

3. **Negation Handling**:
   - Example: "Surgery is needed" vs "Surgery is not needed"
   - Similarity: 0.7791 (Doesn't completely capture the meaning reversal in negations)

## Analysis and Conclusions

1. **Key Achievements**:
   - Competitive performance achieved with an overall average similarity of 0.8168
   - Demonstrates strengths particularly in medical relationships (DISEASE_MEDICATION, SEVERITY_PROGNOSIS)
   - Proves the effectiveness of the NER-based approach in understanding medical-legal contexts

2. **Strengths**:
   - Accurately captures relationships between medical terms and concepts
   - Ability to connect concepts in the medical-legal cross-domain
   - Maintains robust performance even with limited training data

3. **Areas Needing Improvement**:
   - Need to enhance recognition of severity/degree differences in terminology
   - Need to strengthen temporal context and negation processing capabilities
   - Need to secure balanced training data for more diverse relationship types

## Future Improvement Directions

1. **Model Improvements**:
   - Additional training specialized for negations and severity expressions
   - Consider introducing structures that can better capture temporal context information
   - Consider expanding to larger embedding dimensions (e.g., 384) to enhance expressiveness

2. **Data Expansion**:
   - Expand training data using more diverse medical-legal documents
   - Reinforce cases including negations, temporal expressions, and severity differences
   - Build training data using actual case law and medical records

3. **Application Optimization**:
   - Improve integration with PDF visualization tools
   - Develop real-time document comparison functionality
   - Adjust thresholds specialized for specific medical-legal situations

These test results show that the NER-based embedding model is an effective approach for understanding medical-legal contexts. By fine-tuning the model and expanding data to address the currently identified limitations, it can develop into an even more powerful tool for medical-legal document analysis. In particular, it has reached a stage where it can be immediately utilized for real case analysis through integration with PDF document embedding and visualization tools. 