import os
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import argparse
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def preprocess_sentences(sentence1, sentence2):
    vectorizer = TfidfVectorizer().fit_transform([sentence1, sentence2])
    vectors = vectorizer.toarray()
    
    cosine_sim = cosine_similarity(vectors)
    similarity_score = cosine_sim[0][1]
    return similarity_score

def remove_trailing_special_chars(text):
    return re.sub(r'[\W_]+$', '', text)

def remove_special_chars_except_spaces(text):
    return re.sub(r'[^\w\s]+', '', text)

def select_top_k(claim, results, top_k):
  '''
  remove sentence of similarity claim
  '''
  dup_check = set()
  top_k_sentences_urls = []
  
  i = 0
  claim = remove_special_chars_except_spaces(claim).lower()
  while len(top_k_sentences_urls) < top_k:
    sentence = remove_special_chars_except_spaces(results[i]['sentence']).lower()
    
    if sentence not in dup_check:
      if preprocess_sentences(claim, sentence) > 0.97:
        dup_check.add(sentence)
        continue
      
      if claim in sentence:
        if len(claim) / len(sentence) > 0.92:
          dup_check.add(sentence)
          continue 
      
      top_k_sentences_urls.append({
        'sentence': results[i]['sentence'],
        'url': results[i]['url']}
      )
    i += 1
    
  return top_k_sentences_urls
      

def main(args):
  device = "cuda" if torch.cuda.is_available() else 'cpu'
  model = SentenceTransformer("Salesforce/SFR-Embedding-2_R", device=device)
  
  target_examples = []
  with open(args.target_data, "r", encoding="utf-8") as json_file:
      for line in json_file:
          example = json.loads(line)
          target_examples.append(example)


  if args.end == -1:
    args.end = len(target_examples)
    print(args.end)
  
  files_to_process = list(range(args.start, args.end))
  total = len(files_to_process)

  
  task = 'Given a web search query, retrieve relevant passages that answer the query'
  with open(args.json_output, "w", encoding="utf-8") as output_json:
    done = 0
    for idx, example in enumerate(target_examples):
      if idx in files_to_process:
        print(f"Processing claim {example['claim_id']}... Progress: {done + 1} / {total}")

        claim = example['claim']
        query = [get_detailed_instruct(task, claim)] + [get_detailed_instruct(task, le) for le in example['hypo_fc_docs'] if len(le.strip()) > 0]
        query_length = len(query)
        sentences = [sent['sentence'] for sent in example[f'top_{args.retrieved_top_k}']]
        
        st = time.time()
        with torch.no_grad():
          embeddings = model.encode(query + sentences, batch_size=args.batch_size,show_progress_bar=False)

          avg_emb_q = np.mean(embeddings[:query_length], axis=0)
          hyde_vector = avg_emb_q.reshape((1, len(avg_emb_q)))
          
          scores = model.similarity(hyde_vector, embeddings[query_length:])[0].numpy()
          
          top_k_idx = np.argsort(scores)[::-1]
          results = [example['top_10000'][i] for i in top_k_idx]
          top_k_sentences_urls = select_top_k(claim, results, args.top_k)
          print(f"Top {args.top_k} retrieved. Time elapsed: {time.time() - st}.")

          json_data = {
          "claim_id": example['claim_id'],
          "claim": claim,
          f"top_{args.top_k}": top_k_sentences_urls
          }
          output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
          done += 1
          output_json.flush()
      

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--target_data", default="data_store/dev_retrieval_top_k.json")
  parser.add_argument("--retrieved_top_k", type=int, default=10000)
  parser.add_argument("--top_k", type=int, default=10)
  parser.add_argument("-o", "--json_output", type=str, default="data_store/dev_reranking_top_k.json")
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("-s", "--start", type=int, default=0)
  parser.add_argument("-e", "--end", type=int, default=-1)
  args = parser.parse_args()
    
  main(args)
  