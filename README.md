# HerO


# Code for replication

### Hypothetical fact-checking documents (HyDE-FC) generation
```python3
python hyde_fc_generation.py --target_data "dev.json" --output_json "dev_hyde_fc.json"
```

### Evidence retrieval and reranking
```python3
ptyhon retrieval.py --knowledge_store_dir "knowledge_store/dev" --target_data "dev_hyde_fc.json" --json_output "dev_retrieval_top_k.json"

python reranking.py --target_data "dev_retrieval_top_k.json" --json_output "dev_reranking_top_k.json"
```
### Question generation
```python3
python question_generation.py --reference_corpus "train.json" --top_k_target_knowledge "dev_reranking_top_k.json" --output_questions "dev_top_k_qa.json"
```
### Verdict prediction

