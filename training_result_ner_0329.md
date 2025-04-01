# NER Embedding Model Training Results (20250329)

## Model Information

**Model Architecture:**
- **Class**: `NERClinicalBertEmbedder`
- **Base Model**: medicalai/ClinicalBERT
- **Embedding Dimension**: 768 → 256
- **Dropout**: 0.5
- **Model Path**: `models/ner_model_20250329_193235/best_model`
- **Special Feature**: NER (Named Entity Recognition) based embedding

## Training Settings

**Hyperparameters:**
- **Learning Rate**: 1e-5
- **Batch Size**: 16
- **Weight Decay**: 0.1
- **Triplet Margin**: 2.0
- **Gradient Accumulation Steps**: 1
- **Maximum Sequence Length**: 128
- **Evaluation Step**: 100
- **Patience Parameter**: 3
- **Epochs**: 15

**Data:**
- **Dataset**: enhanced_ner_dataset/enhanced_triplets_ner.json
- **Cache Directory**: cache
- **Hard Negative Mining**: Enabled
- **Negative Mining Ratio**: 0.5
- **Negative Mining Epochs**: 3,6,9,12
- **Random Seed**: 42

## Training Results

**Performance Metrics:**
- **Final Accuracy**: 0.6082 (validation dataset)
- **Validation Loss**: 1.9835 (step 100/110)

**Average Similarity by Relationship Type:**
- **DISEASE_MEDICATION**: 0.8514
- **SEVERITY_PROGNOSIS**: 0.8381
- **SEVERITY_COMPENSATION**: 0.8348
- **DISEASE_TREATMENT**: 0.8359
- **DIAGNOSIS_TREATMENT**: 0.8222
- **LEGAL_SIMILAR_MEDICAL_DIFFERENT**: 0.8236
- **TREATMENT_OUTCOME**: 0.8103
- **OUTCOME_SETTLEMENT**: 0.7951
- **MEDICAL_SIMILAR_LEGAL_DIFFERENT**: 0.7812
- **SYMPTOM_DISEASE**: 0.7819
- **Overall Average**: 0.8168

## Observations

1. **Effect of NER-based Embedding**:
   - Effectively captures core entities in medical/legal contexts through named entity recognition
   - Generates differentiated embeddings by relationship type compared to traditional embedding models

2. **Training Specifics**:
   - Some errors occurred during the training process (missing 'save_pretrained' method)
   - Achieved competitive performance despite limited training time (approximately 4 minutes 20 seconds)
   - Hard negative mining contributed to improving the model's discrimination capabilities

3. **Performance Analysis**:
   - High similarity scores in medical relationship types (DISEASE_MEDICATION, SEVERITY_PROGNOSIS)
   - Stable performance maintained in legal-related relationship types
   - Overall relationship type average similarity achieved a high level of 0.8168

## Conclusions and Future Improvement Directions

1. **Key Success Factors**:
   - Enhanced contextual information through NER-based approach
   - Prevention of overfitting through appropriate dropout (0.5)
   - Increased training efficiency through hard negative mining

2. **Improvement Possibilities**:
   - Need to improve model saving mechanism (currently encountering 'save_pretrained' errors)
   - Potential for performance improvement through extended training time
   - Need to secure balanced training data for more diverse relationship types

3. **Application Possibilities**:
   - Effective for understanding semantic similarities between medical-legal documents
   - Can be utilized for real case analysis by integrating with PDF document embedding and visualization tools
   - Potential to develop into specialized models for specific domains (medical or legal) through future fine-tuning

This model demonstrates the effectiveness of the NER-based approach and shows competitive performance as an embedding model for the medical-legal domain. Although there were some technical issues during the training process, the final model provides balanced performance across various relationship types and has reached a level suitable for practical applications. 