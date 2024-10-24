from vllm import LLM, SamplingParams
import json
import torch
import time
import argparse
import tqdm

class VLLMGenerator:
    def __init__(self, model_name, n=8, max_tokens=512, temperature=0.7, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
        self.device_count = torch.cuda.device_count()
        self.llm = LLM(model=model_name,
            tensor_parallel_size=self.device_count,
            max_model_len=4096,
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            trust_remote_code=True,
            dtype="bfloat16"
        )
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success
        
    @staticmethod
    def parse_response(response):
        to_return = []
        for _, g in enumerate(response[0].outputs):
            text = g.text.strip()
            logprob = sum(logprob_obj.logprob for item in g.logprobs for logprob_obj in item.values())
            to_return.append((text, logprob))
        texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
        return texts
    
    @torch.no_grad()
    def generate(self, prompt):
        sampling_params = SamplingParams(
            n=self.n,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
            logprobs=1
        )

        get_result = False
        while not get_result:
            try:
                result = self.llm.generate(prompt, sampling_params=sampling_params)
                get_result = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e            
        return self.parse_response(result)
      
def main(args):
  with open(args.target_data, 'r', encoding='utf-8') as json_file:
    examples = json.load(json_file)
  
  generator = VLLMGenerator(model_name=args.model)  
  
  data = []
  for _, example in tqdm.tqdm(enumerate(examples), ncols=100):
    claim = example["claim"]
    
    prompt = f"Please write a fact-checking article passage to support, refute, indicate not enough evidence, or present conflicting evidence regarding the claim.\nClaim: {claim}"
    messages = [{"role":"user", "content":prompt}]
    inputs = generator.llm.get_tokenizer().apply_chat_template(messages, tokenize=False)
    inputs += "<|start_header_id|>assistant<|end_header_id|>\n\nPassage: "
    outputs = generator.generate(inputs)

    
    example['hypo_fc_docs'] = outputs
    data.append(example)

  with open(args.output_json, "w", encoding="utf-8") as output_file:
    json.dump(data, output_file, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--target_data', default='dev.json')
  parser.add_argument('-o', '--output_json', default='hyde_fc.json')
  parser.add_argument('-m','--model', default="meta-llama/Meta-Llama-3.1-70B-Instruct")
  args = parser.parse_args()
  main(args)
  