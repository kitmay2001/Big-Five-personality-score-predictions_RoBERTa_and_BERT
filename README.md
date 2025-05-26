# Big-Five-personality-score-predictions_RoBERTa_and_BERT 

This repository includes code and data for a research project focused on Personality Trait Prediction using four models: BERT base, BERT large, RoBERTa base and RoBERTa large. 
The objective is to investigate how well these models predict personality traits and identify correlations among them based on user-generated text collected from Reddit (arXiv:2004.04460v3).\
The full paper of this project can be found at the following link:
https://www.mdpi.com/2078-2489/16/5/418

The Python code for the multiple-models approach is included, while the Python code for the single-model approach is not included, as it is identical to `roberta_base_pandora.py`.


The code is organized into the following modules:
- bert_base_pandora : Python code for fine-tuning using BERT base
- bert_large_pandora : Python code for fine-tuning using BERT large
- roberta_base_pandora : Python code for fine-tuning using RoBERTa base
- roberta_large_pandora : Python code for fine-tuning using RoBERTa large
- multiple_models_approach_o.py : Python code for multiple-models approach fine-tuning using RoBERTa base
