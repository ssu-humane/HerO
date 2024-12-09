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

## Model for replication
We use Finetuned 8b LLM for question generation and 70b LLM for veracity prediction. models and datasets are available at huggingface 🤗

- [humane-lab/AVeriTeC-HerO](https://huggingface.co/datasets/humane-lab/AVeriTeC-HerO) is our training dataset for the veraity prediction and justification generation model. We modify the [AVeriTeC dataset](https://huggingface.co/chenxwh/AVeriTeC) to be used for instruction training.

- [humane-lab/Meta-Llama-3.1-8B-HerO](https://huggingface.co/humane-lab/Meta-Llama-3.1-8B-HerO) is our fine-tuned 8b model for veracity prediction and justification generation. We use Meta-Llama-3.1-8B for base model.

- [humane-lab/Meta-Llama-3.1-70B-HerO](https://huggingface.co/humane-lab/Meta-Llama-3.1-70B-HerO) is our fine-tuned 70b model for veracity prediction and justification generation. We use Meta-Llama-3.1-70B for base model.

You can use our provided models or train your own models using the training dataset we provide.

## Code for replication
We use [vllm](https://github.com/vllm-project/vllm) to infer from LLMs and [axolotl](https://github.com/axolotl-ai-cloud/axolotl) to train LLMs.

Our repository use gated models like Llama, so you might need a authentication token of huggingface.

We also provide the result file of each steps in [data_store/baseline](https://github.com/ssu-humane/HerO/tree/main/data_store/baseline) directory.

### Installation
```bash
git clone https://github.com/ssu-humane/HerO.git
cd HerO
pip install -r requirements.txt
```

### Data Preparation
Download the AVeriTeC dataset and place it in the `data_store/averitec` directory. More details can be found in the [data_store/averitec/README.md](https://github.com/ssu-humane/HerO/tree/main/data_store/averitec)

### Evidence Retrieval
#### Hypothetical fact-checking documents (HyDE-FC) generation
```python3
python hyde_fc_generation.py --target_data "data_store/averitec/dev.json" --json_output "data_store/dev_hyde_fc.json"
```

#### Evidence retrieval and reranking
```python3
python retrieval.py --knowledge_store_dir "knowledge_store/dev" --target_data "data_store/dev_hyde_fc.json" --json_output "data_store/dev_retrieval_top_k.json"

python reranking.py --target_data "data_store/dev_retrieval_top_k.json" --json_output "data_store/dev_reranking_top_k.json"
```

> HyDE-FC generation, evidence retrieval and reranking takes about 6 hours in two H100.

### Question generation
```python3
python question_generation.py --reference_corpus "data_store/averitec/train.json" --top_k_target_knowledge "data_store/dev_reranking_top_k.json" --output_questions "data_store/dev_top_k_qa.json" --model "meta-llama/Meta-Llama-3-8B-Instruct"
```

> Generate questions for the dev set (8b LLM) takes about 25 minutes in two H100.

### Veracity prediction
```python3
python veracity_prediction.py --target_data "data_store/dev_top_k_qa.json" --output_file "data_store/dev_veracity_prediction.json" --model "humane-lab/Meta-Llama-3.1-70B-HerO"
```

> Veracity prediction for the dev set (70b Finetuned LLM) takes about 12 minutes in two H100.

### Evaluation
```python3
python averitec_evaluation.py --prediction_file "data_store/dev_veracity_prediction.json" --label_file "data_store/averitec/dev.json"
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
