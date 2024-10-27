# HerO at AVeriTeC: The Herd of Open Large Language Models for Verifying Real-World Claims

This repository provides the code for our paper, ["HerO at AVeriTeC: The Herd of Open Large Language Models for Verifying Real-World Claims"](https://arxiv.org/abs/2410.12377) to be published at Seventh Workshop on Fact Extraction and VERification (FEVER), 2024. (co-located with EMNLP)

## [AVeriTeC shared task](https://fever.ai/task.html)
- Given a claim and its metadata, the systems must retrieve evidence that supports and/or refutes the claim, either from the Web or from the document collection provided by the organizers.
- Using this evidence, label the claim as Supported, Refuted given the evidence, Not Enough Evidence (if there isn't sufficient evidence to either support or refute it) or Conflicting Evidence/Cherry-picking (if the claim has both supporting and refuting evidence).


## Method: HerO
We present HerO, a herd of open large language models for verifying real-world claims.
<p align="center"><img src="https://github.com/user-attachments/assets/6cc0d0ea-78ec-4b84-b9cc-f905916dd972" width="900" height="400"></p>
This figure illustrates the inference pipeline of our system. We configure three modules using only open LLMs to fact-check real-world claims in the AVeriTeC dataset: evidence retrieval, question generation, and veracity prediction.


#### The main features of the three modules
- Evidence retrieval: By leveraging [HyDE](https://aclanthology.org/2023.acl-long.99/), we expand the query by generating hypothetical fact-checking documents that rely on the LLM's parametric knowledge. We retrieve 2-stage using BM25 and [SFR-Embedding-2_R](https://huggingface.co/Salesforce/SFR-Embedding-2_R).
- Question generation: We add a claim at LLM input.
- Veracity prediction: We fully fine-tune the LLM to generate justifications and verdicts using the training set of the AVeriTeC dataset.

## Code for replication

### Hypothetical fact-checking documents (HyDE-FC) generation
```python3
python hyde_fc_generation.py --target_data "dev.json" --json_output "dev_hyde_fc.json"
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
```python3
```

### Citation
```
@article{yoon2024hero,
  title={HerO at AVeriTeC: The Herd of Open Large Language Models for Verifying Real-World Claims},
  author={Yoon, Yejun and Jung, Jaeyoon and Yoon, Seunghyun and Park, Kunwoo},
  journal={arXiv preprint arXiv:2410.12377},
  year={2024}
}
```
