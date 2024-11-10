# HerO at AVeriTeC: The Herd of Open Large Language Models for Verifying Real-World Claims

This repository provides the code for our paper titled ["HerO at AVeriTeC: The Herd of Open Large Language Models for Verifying Real-World Claims"](https://aclanthology.org/2024.fever-1.15/).

## Task: AVeriTeC

- The AVeriTeC task is to verify a real-world claim by retrieving evidence from the web. Given a claim and its metadata, a system needs to retrieve evidence that supports and/or refutes the claim, either from the Web or from the document collection provided along with the dataset.
- This code is for our fact-checking pipeline that utilizes open large language models for the shared task hosted by the 7th FEVER workshop (co-located with EMNLP). For more details about the task and dataset, please refer to [the paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html) or [the website](https://fever.ai/task.html).

## Method: HerO

We present HerO, a herd of open large language models for verifying real-world claims.
<p align="center"><img src="https://github.com/user-attachments/assets/6cc0d0ea-78ec-4b84-b9cc-f905916dd972" width="900" height="400"></p>
This figure illustrates the inference pipeline of our system. We configure three modules using only open LLMs to fact-check real-world claims in the AVeriTeC dataset: evidence retrieval, question generation, and veracity prediction.


#### The main features of the three modules

- Evidence retrieval: We implement a 2-stage retrieval pipeline using BM25 and [SFR-Embedding-2_R](https://huggingface.co/Salesforce/SFR-Embedding-2_R). We expand the query by generating hypothetical fact-checking documents that rely on the LLM's parametric knowledge.
- Question generation: We use an LLM to generate a verifying question for an answer candidate. We improve the baseline prompt by using the claim as an additional context.
- Veracity prediction: We fully fine-tune an LLM to generate justifications and verdicts.

## Model for replication
We use the 8b model for question generation and the 70b model for veracity prediction. We make the model checkpoints and datasets available at Huggingface 🤗

- [humane-lab/AVeriTeC-HerO](https://huggingface.co/datasets/humane-lab/AVeriTeC-HerO) is our training dataset for the veraity prediction and justification generation model. We modify the [AVeriTeC dataset](https://huggingface.co/chenxwh/AVeriTeC) to be used for instruction training.

- [humane-lab/Meta-Llama-3.1-8B-HerO](https://huggingface.co/humane-lab/Meta-Llama-3.1-8B-HerO) is our fine-tuned 8b model for veracity prediction and justification generation. We use Meta-Llama-3.1-8B for the base model.

- [humane-lab/Meta-Llama-3.1-70B-HerO](https://huggingface.co/humane-lab/Meta-Llama-3.1-70B-HerO) is our fine-tuned 70b model for veracity prediction and justification generation. We use Meta-Llama-3.1-70B as the base model.


## Code for replication
We use [vllm](https://github.com/vllm-project/vllm) to infer from LLMs and [axolotl](https://github.com/axolotl-ai-cloud/axolotl) to train LLMs.

Our repository uses gated models like Llama so that you might need an authentication token of huggingface.

We also provide the result file of each step in the [data_store/baseline](https://github.com/ssu-humane/HerO/tree/main/data_store/baseline) directory.

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
python averitec_evaluation.py --prediction_file "data_store/dev_veracity_prediction.json" --reference_file "data_store/averitec/dev.json"
```

### Citation

The code and dataset are shared under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0). Please cite our paper if you use our code.
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
