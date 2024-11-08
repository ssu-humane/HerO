import tqdm
import argparse
import torch
import transformers
import json
from vllm import LLM, SamplingParams

LABEL = [
  "Supported",
  "Refuted",
  "Not Enough Evidence",
  "Conflicting Evidence/Cherrypicking",
]

def main(args):
  try:
    with open(args.target_data) as f:
      examples = json.load(f)
  except:
    examples = []
    with open(args.target_data) as f:
      for line in f:
        examples.append(json.loads(line))

  predictions = []

  tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

  gpu_counts = torch.cuda.device_count()

  llm = LLM(
    model=args.model,
    tensor_parallel_size=gpu_counts,
    max_model_len=4096,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
    trust_remote_code=True
  )

  sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.7,
    top_k=1,
    early_stopping=False,
    skip_special_tokens=False,
    max_tokens=512,
    stop=['<|endoftext|>', '</s>', '<|im_end|>', '[INST]', '[/INST]','<|eot_id|>','<|end|>']
  )

  for example in tqdm.tqdm(examples):
    # reverse the list
    
    prompt = "Your task is to predict the verdict of a claim based on the provided question-answer pair evidence. Choose from the labels: 'Supported', 'Refuted', 'Not Enough Evidence', 'Conflicting Evidence/Cherrypicking'. Disregard irrelevant question-answer pairs when assessing the claim. Justify your decision step by step using the provided evidence and select the appropriate label."
    example["input_str"] = prompt + "\n\nClaim: " + example["claim"] + "\n\n" + "\n\n".join([f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}" for i, qa in enumerate(example["evidence"])])

    messages = [
      {"role": "user", "content": example["input_str"]},
    ]
    
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False)

    label = None
    while label == None:
      output = llm.generate(input_ids, sampling_params)
      
      output = output[0].outputs[0].text.strip()

      if "Not Enough Evidence" in output:
        label = "Not Enough Evidence"
      elif "Conflicting Evidence/Cherrypicking" in output or "Cherrypicking" in output or "Conflicting Evidence" in output:
        label = "Conflicting Evidence/Cherrypicking"
      elif "Supported" in output or "supported" in output:
        label = "Supported"
      elif "Refuted" in output or "refuted" in output:
        label = "Refuted"
      else:
        label = None
        sampling_params = SamplingParams(
          temperature=0.9,
          top_p=0.7,
          top_k=2,
          early_stopping=False,
          skip_special_tokens=False,
          max_tokens=512,
          stop=['<|endoftext|>', '</s>', '<|im_end|>', '[INST]', '[/INST]','<|eot_id|>','<|end|>']
        )
        print("Error: could not find label in output.")
        print(output)

    print(output)

    json_data = {
      "claim_id": example["claim_id"],
      "claim": example["claim"],
      "evidence": example["evidence"],
      "pred_label": label,
      "llm_output": output,
    }
    predictions.append(json_data)

  with open(args.output_file, "w", encoding="utf-8") as output_file:
    json.dump(predictions, output_file, ensure_ascii=False, indent=4)

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default="meta-llama/Meta-Llama-3-8B-Instruct")
  parser.add_argument("-i", "--target_data", default="llm_dev.json")
  parser.add_argument("-o", "--output_file", default="dev_veracity_prediction.json")
  args = parser.parse_args()    
  
  main(args)