# Biomedical Abbreviation and Long-form Detection using Token Classification

This project investigates multiple deep learning models for identifying biomedical abbreviations (AC) and long forms (LF) using the [PLOD-CW-25 dataset](https://huggingface.co/datasets/surrey-nlp/PLOD-CW-25). It involves EDA, token classification modeling, and evaluation of trade-offs between model accuracy and efficiency.

## 📊 Exploratory Data Analysis (EDA)

1. **Dataset Overview, Completeness & Sentence Counts**
   - Verified token-level annotation consistency
   - Measured document and sentence-level distribution across train, validation, and test splits

2. **Token and Tag Distributions**
   - Analyzed frequency of BIO tags (B-AC, B-LF, I-LF, O)  
   - Investigated token casing and sentence length patterns

3. **Sub-Domain Exploration & Abbreviation Analysis**
   - Grouped entries by biomedical sub-domains  
   - Identified trends in abbreviation usage density

4. **Abbreviation Characteristics & Ambiguity**
   - Checked reuse and ambiguity of abbreviations (e.g., multiple meanings)  
   - Evaluated impact of character length and term frequency on detection complexity

## 🧠 Experiments & Models

### 🔹 Traditional Models
- **Model:** CRF + BiLSTM  
- **Embeddings:** Word2Vec and Word+Char  
- **Result:** Macro F1 ≈ 0.73  

### 🔹 Sequence Models
- **Model:** RNN and Bi-LSTM  
- **Embeddings:** FastText  
- **Result:** Bi-LSTM F1 ≈ 0.67; RNN F1 ≈ 0.52  

### 🔹 Transformer Model
- **Model:** Fine-tuned RoBERTa  
- **Optimizers:** Adam, LION, LAMB  
- **Tokenizer:** RobertaTokenizerFast (BPE)  
- **Result:** RoBERTa + LION: **Micro F1 = 0.8622**, **Macro F1 = 0.855**

## ✅ Evaluation Summary

### A. Can the models fulfil their purpose?  
Yes — all models successfully detected biomedical abbreviations and long forms. RoBERTa + LION outperformed others with superior generalization and convergence speed.

### B. What is a good F1/accuracy threshold?  
F1-scores above **0.75** are considered strong for biomedical NER. Our best model reached **0.855 macro F1**, exceeding this threshold.

### C. How could low-performing models be improved?
- Use BiLSTM instead of plain RNN  
- Incorporate contextual embeddings (e.g., BioBERT) instead of static ones  
- Fine-tune transformers properly; pretrained-only models underperform  
- Avoid LAMB for small-batch training

### D. Tokenization Strategy Impact
- Word-level models used native PLOD tokens  
- RoBERTa used BPE subword tokenization, which improved performance  
- Subword realignment was essential for BIO tagging

### E. Accuracy vs. Efficiency Trade-off  
- RoBERTa + LION is highly accurate but resource-heavy  
- For deployment, distillation or pruning can reduce size with ~2–3% performance drop  
- Critical applications (clinical, research) may favor full model; lightweight tools can trade-off



