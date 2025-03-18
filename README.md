# HerO at AVeriTeC: The Herd of Open Large Language Models for Verifying Real-World Claims
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hero-at-averitec-the-herd-of-open-large/fact-checking-on-averitec)](https://paperswithcode.com/sota/fact-checking-on-averitec?p=hero-at-averitec-the-herd-of-open-large) [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat)](https://arxiv.org/abs/2410.12377)

This repository provides the code for 🌟HerO🌟, the runner-up :runner: for the AveriTeC shared task. 

The system description paper is published in the proceedings of the 7th FEVER workshop (co-located with EMNLP 2024) [[paper]](https://aclanthology.org/2024.fever-1.15/).

Our method has been selected as a [baseline for the 8th FEVER workshop shared task](https://github.com/Raldir/FEVER-8-Shared-Task) (co-located with ACL 2025)

## Task: AVeriTeC

- The AVeriTeC task is to verify a real-world claim by retrieving evidence from the web. Given a claim and its metadata, a system needs to retrieve evidence that supports and/or refutes the claim, either from the Web or from the document collection provided along with the dataset.
- This code is for our fact-checking pipeline that utilizes open large language models for the shared task hosted by the 7th FEVER workshop (co-located with EMNLP). For more details about the task and dataset, please refer to [the shared task paper](https://aclanthology.org/2024.fever-1.1/).

## Method: HerO

- HerO, the herd of open large language models for real-world claims, is our pipelined system for verifying real-world claims.
- :tada: Our system achieved 2nd place in the shared task! As the winner utilizes GPT-4o for their pipeline, HerO is the best one among those using open LLMs.
<p align="center"><img src="https://github.com/user-attachments/assets/6cc0d0ea-78ec-4b84-b9cc-f905916dd972" width="900" height="400"></p>

- The above figure illustrates our system's inference pipeline. We configure three modules using only open LLMs: evidence retrieval, question generation, and veracity prediction.
  + Evidence retrieval: We implement a 2-stage retrieval pipeline using BM25 and [SFR-Embedding-2_R](https://huggingface.co/Salesforce/SFR-Embedding-2_R). We expand the query by prompting an LLM to generate hypothetical fact-checking documents.
  + Question generation: We use an LLM to generate a verifying question for an answer candidate. We improve the baseline prompt by using the claim as an additional context.
  + Veracity prediction: We fully fine-tune an LLM to generate justifications and verdicts.

## Veracity Prediction Model and Fine-tuning Dataset
The model checkpoints and instruction datasets are available at Hugging Face Hub 🤗

### Veracity Prediction Model Checkpoints
We fine-tune the 8b model and the 70b model for veracity prediction.

- [humane-lab/Meta-Llama-3.1-8B-HerO](https://huggingface.co/humane-lab/Meta-Llama-3.1-8B-HerO) is our fine-tuned 8b model for veracity prediction and justification generation. We use Meta-Llama-3.1-8B for the base model.

- [humane-lab/Meta-Llama-3.1-70B-HerO](https://huggingface.co/humane-lab/Meta-Llama-3.1-70B-HerO) is our fine-tuned 70b model for veracity prediction and justification generation. We use Meta-Llama-3.1-70B as the base model.

### Fine-tuning Dataset
We created our fine-tuning dataset using our own prompts along with AVeriTeC justifications and verdicts to train the veracity prediction model

- [humane-lab/AVeriTeC-HerO](https://huggingface.co/datasets/humane-lab/AVeriTeC-HerO) is our training dataset for the veraity prediction and justification generation model. We modify the [AVeriTeC dataset](https://huggingface.co/chenxwh/AVeriTeC) to be used for instruction training.

## How to Run

### Installation
```bash
git clone https://github.com/ssu-humane/HerO.git
cd HerO
pip install -r requirements.txt
```

### AVeriTeC Data Preparation
Download the AVeriTeC dataset and place it in the `data_store/averitec` directory. More details can be found in the [data_store/averitec/README.md](https://github.com/ssu-humane/HerO/tree/main/data_store/averitec)

### Evidence retrieval

#### Hypothetical fact-checking documents (HyDE-FC)

```python3
python hyde_fc_generation.py --target_data "data_store/averitec/dev.json" --json_output "data_store/dev_hyde_fc.json"
```

#### Retrieval and reranking

```python3
python retrieval.py --knowledge_store_dir "knowledge_store/dev" --target_data "data_store/dev_hyde_fc.json" --json_output "data_store/dev_retrieval_top_k.json"
python reranking.py --target_data "data_store/dev_retrieval_top_k.json" --json_output "data_store/dev_reranking_top_k.json"
```

> The evidence retrieval pipeline takes about 6 hours in two H100.

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
python averitec_evaluate.py --prediction_file "data_store/dev_veracity_prediction.json" --label_file "data_store/averitec/dev.json"
```

> You can also evaluate using hidden test set at https://eval.ai/web/challenges/challenge-page/2285/overview

## License \& Attribution

The code and dataset are shared under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0).
```
@inproceedings{yoon-etal-2024-hero,
    title = "{H}er{O} at {AV}eri{T}e{C}: The Herd of Open Large Language Models for Verifying Real-World Claims",
    author = "Yoon, Yejun  and
      Jung, Jaeyoon  and
      Yoon, Seunghyun  and
      Park, Kunwoo",
    booktitle = "Proceedings of the Seventh Fact Extraction and VERification Workshop (FEVER)",
    month = nov,
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.fever-1.15",
    pages = "130--136",
}
```
