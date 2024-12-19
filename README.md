# 101_notebooks
101 Notebooks in Text Classification

## Part 1 - Embedding Models
 * 1.1 - **Scraping and Chunking**
   * 1.1.1_dataset.ipynb; **Parsing and extracting from HTML, splitting with Langchain and Llama-index**
* 1.2 - **Simple Embedding Model Classifier**
  * _1.2.1_vanilla_embed_Sentence_Transformers.ipynb; **Embedding with Sentnece Transformers library**
  * _1.2.2_vanilla_classify_Sentence_Transformers.ipynb; **MLP Classification model in PyTorch**
  * 1.2.1_vanilla_embed.ipynb; **Embedding with Transformers library**
  * 1.2.2_vanilla_classification.ipynb; **MLP Classificiation model PyTorch**
  * 1.2.3_vanilla_predictions.ipynb; **Inference**
  * 1.2.4_vanilla_analysis.ipynb; **Metrics, visualization, results**
* 1.3 - **Soft Mixture of Experts Classifier**
  * 1.3.1_vanilla_moe_e2e_soft_classify.ipynb; **Soft Mixture of Experts classification model in PyTorch**
  * 1.3.2_vanilla_moe_e2e_soft_predictions.ipynb; **Inference**
  * 1.3.3_vanilla_moe_e2e_soft_analysis.ipynb; **Metrics, visualization, results**
* 1.4 - **Hard Mixture of Experts Classifier**
  * 1.4.1_vanilla_moe_e2e_hard_classify.ipynb; **Hard Mixture of Experts classification model in PyTorch**
  * 1.4.2_vanilla_moe_e2e_hard_predictions.ipynb; **Inference**
  * 1.4.3_vanilla_moe_e2e_hard_analysis.ipynb; **Metrics, visualization, results**
* 1.5 - **Hard Mixture of Pre-trained Experts Classifier**
  * 1.5.1_vanilla_moe_hard_pre_classify.ipynb; **Hard Mixture of Pre-trained Experts classification model in PyTorch**
  * 1.5.2_vanilla_hard_pre_predictions.ipynb; **Inference**
  * 1.5.3_vanilla_moe_e2e_forest_analysis.ipynb; **Metrics, visualization, results**
* 1.6 - **Soft Forest of Experts Classifier**
  * 1.6.1_vanilla_moe_e2e_forest_classify.ipynb; **Soft Forest of Experts classification model in PyTorch**
  * 1.6.2_vanilla_moe_e2e_forest_predictions.ipynb; **Inference**
  * 1.6.3_vanilla_moe_e2e_forest_analysis.ipynb; **Metrics, visualization, results**
* 1.7 - **Finetune with Triplet Margin Loss**
  * 1.7.1_embedding_ft_train.ipynb; **Fine-tune embedding model with Tranformers and PyTorch**
  * 1.7.2_embedding_ft_embeddings.ipynb; **Get embeddings from fine-tuned model**
  * 1.7.3_embedding_ft_classify.ipynb; **MLP classification model in PyTorch**
  * 1.7.4_embedding_ft_predictions.ipynb; **Inference**
  * 1.7.5_embedding_ft_analysis.ipynb; **Metrics, vizualization, results**
* 1.8 - **Finetune on Classification Task**
  * 1.8.1_direct_ft_classify.ipynb; **Fine-tune embedding model in PyTorch on classification task**
  * 1.8.2_direct_ft-predictions.ipynb; **Inference**
  * 1.8.3_direct_ft_analysis.ipynb; **Metrics, vizualization, results**
  * 1.8.4_direct_ft_soft_moe_classify.ipynb; **Fine-tune embedding model with Mixture of Experts Head**
  * 1.8.5_direct_ft_soft_moe_predictions; **Inference**
  * 1.8.6_direct_ft_analysis; **Metrics, vizualization, results**
* 1.9 - **Observations Part One**
  * 1.9.1_vanilla_moe_hard_pre_expert_check; **Specialization of Experts for Hard Pre-Trained MoE Model**
  * 1.9.2_vanilla_moe_e2e_soft_expert_check; **Specialization of Experts for Soft MoE model**
  * 1.9.3_vanilla_moe_e2e_hard_expert_check; **Specialization of Experts for hard MoE model**

## Part 2 - LLM Models
* 2.1 - **QLoRA**
  * 2.1.1_llm_instruct_data.ipynb; **Organizing data for Transformer fine-tuning**
  * 2.1.2_llm_instruct_finetune.ipynb; **Fine-tune Mistral 7B Istruct model with QLoRA**
  * 2.1.3_llm_instruct_ots_inference.ipynb; **Inference with the base model**
  * 2.1.4_llm_instruct_ft_inference.ipynb; **Inference with the finetuned model**
* 2.2 - **GPT**
  * 2.2.1_inference_GPT_40_mini.ipynb; **Inference with OpenAI model**
  * 2.2.2_finetune_GPT_4o_mini.ipynb; **Finetune OpenAI model through API**
  * 2.2.3_inference_GPT_4o_mini_ft.ipynb; **Inference with the finetuned model**
  T.B.C

## Part 3 - Graph Neural Networks
* 3.1 - **GNN**
  * 3.1.1_adjacency_matrix.ipynb; **Construction of adjacency matrices**
  * 3.1.2_GNN_train.ipynb; **Creating and training the GNN model**
  * 3.1.3_GNN_inference; **Inference**  
  T.B.C



    
