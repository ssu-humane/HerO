import os
import argparse
import time
import json
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
import torch
from vllm import LLM, SamplingParams

def claim2prompts(example): 
  claim = example["claim"]
  claim_str = "Example [NUMBER]:||Claim: " + claim + "||Evidence: "

  for question in example["questions"]:
    q_text = question["question"].strip()
    if len(q_text) == 0:
      continue

    if not q_text[-1] == "?":
      q_text += "?"

    answer_strings = []

    for a in question["answers"]:
      if a["answer_type"] in ["Extractive", "Abstractive"]:
        answer_strings.append(a["answer"])
      if a["answer_type"] == "Boolean":
        answer_strings.append(a["answer"]  + ", because " + a["boolean_explanation"].lower().strip())

    for a_text in answer_strings:
      if not a_text[-1] in [".", "!", ":", "?"]:
        a_text += "."

      prompt_lookup_str = a_text
      this_q_claim_str = claim_str + a_text.strip() + "||Question: " + q_text 
      yield (prompt_lookup_str, this_q_claim_str.replace("\n", " ").replace("||", "\n")[:1500]) 


def main(args):
  # few-shot learning from the training set
  with open(args.reference_corpus, "r", encoding="utf-8") as json_file:
    train_examples = json.load(json_file)

  prompt_corpus, tokenized_corpus = [], []

  for example in train_examples:
    for lookup_str, prompt in claim2prompts(example):
      entry = nltk.word_tokenize(lookup_str)
      tokenized_corpus.append(entry)
      prompt_corpus.append(prompt)

  prompt_bm25 = BM25Okapi(tokenized_corpus)
  
  gpu_count = torch.cuda.device_count()
  llm = LLM(model=args.model,
            tensor_parallel_size=gpu_count,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            trust_remote_code=True
  )
  llm.get_tokenizer().pad_token = "<|end_of_text|>"
  
  sampling_params = SamplingParams(
                      temperature=0.6,
                      top_p=0.9,
                      top_k=1,
                      early_stopping=False,
                      skip_special_tokens=False,
                      max_tokens=512,
                      stop=['<|end_of_text|>', '</s>', '<|im_end|>', '[INST]', '[/INST]','<|eot_id|>','<|end|>','<|endoftext|>']
  )

  start_time = time.time()
  with torch.no_grad():
    with open(args.output_questions, "w", encoding="utf-8") as output_file:
      done = 0
      with open(args.top_k_target_knowledge, "r", encoding="utf-8") as json_file:
        for i, line in enumerate(json_file):
          data = json.loads(line)
          top_k_sentences_urls = data[f"top_{args.top_k}"]
          claim = data["claim"]
          claim_id = data["claim_id"]

          bm25_qau = []  # question, answer, url
          # Generate questions for those top k:
          for sent_i, sentences_urls in enumerate(top_k_sentences_urls):
            prompt_lookup_str = sentences_urls["sentence"]
            url = sentences_urls["url"]

            prompt_s = prompt_bm25.get_scores(
                nltk.word_tokenize(prompt_lookup_str)
            )
            prompt_n = 10
            prompt_top_n = np.argsort(prompt_s)[::-1][:prompt_n]
            prompt_docs = [prompt_corpus[i] for i in prompt_top_n]                            
            
            evidence = prompt_lookup_str.replace("\n", " ")
            temp_prompt = "\n\n".join(prompt_docs)
            for k in range(1, temp_prompt.count("[NUMBER]")+1): temp_prompt = temp_prompt.replace("[NUMBER]", f"{k}", 1)
            claim_prompt = "Your task is to generate a question based on the given claim and evidence. The question should clarify the relationship between the evidence and the claim\n\n"
            
            prompt = claim_prompt + temp_prompt + "\n\nNow, generate a question that links the following claim and evidence:" + f"\n\nClaim: {claim}" + f"\nEvidence: {evidence}"

            messages = [{"role":"user", "content":prompt}]
            
            inputs = llm.get_tokenizer().apply_chat_template(messages, tokenize=False)
            inputs += "<|start_header_id|>assistant<|end_header_id|>\n\nQuestion: "

            st = time.time()
            outputs = llm.generate(inputs, sampling_params)
            outputs = outputs[0].outputs[0].text.strip()
            
            print(f"Generated QA for sent {sent_i} in file {i}. Time elapsed: {time.time() - st}")

            qau_pair = [
              outputs.strip().split("?")[0].replace("\n", " ") + "?",
              prompt_lookup_str.replace("\n", " "),
              url,
            ]

            bm25_qau.append(qau_pair)

          evidence = [
            {
              "question": bm25_qau[i][0],
              "answer": bm25_qau[i][1],
              "url": bm25_qau[i][2],
            }
            for i in range(args.top_k)
          ]
          
          json_data = {
              "claim_id": claim_id,
              "claim": claim,
              "evidence": evidence,
          }
          output_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
          done += 1
          output_file.flush()
  print(time.time()-start_time)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Use a prompt to generate questions that could be answered by top-k retrieved evidence. Output generated questions.")
  parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
  parser.add_argument("--reference_corpus", default="data/original/train.json")
  parser.add_argument(
    "-i",
    "--top_k_target_knowledge",
    default="dev_reranking_top_k.json",
    help="Directory where the sentences for the scraped data is saved.",
  )
  parser.add_argument(
    "-o",
    "--output_questions",
    default="dev_top_k_qa.json",
    help="Directory where the sentences for the scraped data is saved.",
  )
  parser.add_argument(
    "--top_k",
    default=10,
    type=int
  )
  
  args = parser.parse_args()
  
  main(args)
