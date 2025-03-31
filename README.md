# 101_notebooks
101 Notebooks in Text Classification

## Part 0 - Miscellaneous
 * 0.1 - **Misc**
   * 0.1.1_data_augmentation_comparisson.ipynb; **Statistical comparisson original vs augmented data**     

## Part 1 - Embedding Models
 * 1.1 - **Scraping and Chunking**
   * 1.1.1_dataset.ipynb; **Parsing and extracting from HTML, splitting with Langchain and Llama-index**
   * 1.1.1a_augmented_dataset.ipynb; **Parsing, extracting, and splitting augmented data**
* 1.2 - **Simple Embedding Model Classifier**
  * _1.2.1_vanilla_embed_Sentence_Transformers.ipynb; **Embedding with Sentnece Transformers library**
  * _1.2.2_vanilla_classify_Sentence_Transformers.ipynb; **MLP Classification model in PyTorch**
  * 1.2.1_vanilla_embed.ipynb; **Embedding with Transformers library**
  * 1.2.1a_augmented_vanilla_embed; **Embedding augmented data**
  * 1.2.2_vanilla_classification.ipynb; **MLP Classificiation model PyTorch**
  * 1.2.2a_augmented_vanilla_classify; **Clasification model with augmented data**
  * 1.2.2aft_augmeted_ft_embed.ipynb; **Classification model with augmented data and fintuned embeddings**
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
  * 1.5.1_vanilla_moe_hard_pre_classify.ipynb; **Hard Mixture of Pre-trained Experts classification model**
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
  * 1.8.4a_direct_ft_soft_moe_classify_augmented.ipynb; **Fine-tube embeddings / MoE head / augmented data**
  * 1.8.5_direct_ft_soft_moe_predictions; **Inference**
  * 1.8.6_direct_ft_analysis; **Metrics, vizualization, results**
* 1.9 - **Observations**
  * 1.9.1_vanilla_moe_hard_pre_expert_check; **analysis of Experts for Hard Pre-Trained MoE Model**
  * 1.9.2_vanilla_moe_e2e_soft_expert_check; **analysis of Experts for Soft MoE model**
  * 1.9.3_vanilla_moe_e2e_hard_expert_check; **analysis of Experts for Hard MoE model**

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
* 2.3 - **NER**
  * 2.3.1_ner_dataset.ipynb; **Creation of NER fine-tuning dataset**
  * 2.3.2_ner_gpt_4o_mini.apynb; **Evaluation and metrics off-the-shelf GPT model**
  * 2.3.3_ner_finetune_gpt_4o_mini.ipynb; **Fine-tune GPT model for NER task**
  * 2.3.4_ner_test_gpt_4o_mini_ft.ipynb; **Evaluation and metrics fine-tuned model**
  * 2.3.5_ner_batch_inference_gpt_4o_mini_ft.ipynb; **OpenAI batch API,for NER inference on entire dataset**
  
## Part 3 - Graph Neural Networks
* 3.1 - **GNN**
  * 3.1.1_adjacency_matrix.ipynb; **Construction of adjacency matrices**
  * 3.1.2_GNN_train.ipynb; **Creating and training the GNN model**
  * 3.1.3_GNN_inference; **Inference**
  * 3.1.4_GNN_analysis; **Metrics, visualization, results**
  * 3.1.5_GNN_distance_sampler_train.ipynb; **Creating a GNN model that samples neighbors by distance**
  * 3.1.6_GNN_trace.ipynb; **Tracing GNN for computational efficiency**
* 3.2 - **GAT**
  * 3.2.1_GAT_train.ipynb; **Training a Graph Attention Network**
  * 3.3.2_GAT_geo_distance.ipynb; **Training a GAT model that samples neighbors by distance**
* 3.3 - **PyTorch Geometric**
  * 3.3.1_GNN_geometric.ipynb; **Training a GNN with torch geometric**
  * 3.3.2_GNN_geo_distance.ipynb; **Training a GNN with torch geometric that samples enighbors by distance**

## Part 4 - Agents
* 4.1 - **LangGraph**
  * 4.1.1_router_agent.ipynb; **LangGraph router with best models in project**  
 



    
