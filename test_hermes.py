import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("teknium/Puffin-Phi-v2", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("teknium/Puffin-Phi-v2", trust_remote_code=True, torch_dtype=torch.bfloat16)
inputs = tokenizer(f"### Instruction:\nThis is Arnav's work:  Led a team of 38 international judges and 20 in-person volunteers to run MakeUC 2022 for 800+ hackers, organizing hacker spaces, food, networking, professional mentorship, and live project judging with feedback. What do you think about him? Do not make up information. \n### Response:\n", return_tensors="pt", return_attention_mask=False).to(device)
outputs = model.generate(**inputs, max_length=128, do_sample=True, temperature=0.2, top_p=0.9, use_cache=True, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
text = tokenizer.batch_decode(outputs)[0]
print(text)
