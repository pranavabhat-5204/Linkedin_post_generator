prompt = "Compose a LinkedIn post announcing the launch of the Llama 3 Tool Use 8B and 70B models. Highlight its top ranking on the Berkeley Function Calling Leaderboard, surpassing all other open-source and proprietary models. Encourage readers to read the full blog for insights and benchmark results and to start using the model via the GroqCloud Dev Hub or by downloading from Hugging Face. Acknowledge the contributions from the Groq and Glaive teams and emphasize the model's performance on the BFCL benchmark. Include the link https://hubs.la/Q02GSbkv0 and the hashtag #OpenAlwaysWins."
#prompt=str(input()) for getting the prompt from user
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your fine-tuned model and tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained("my_fine_tuned_t5_small_model")
loaded_model = AutoModelForSeq2SeqLM.from_pretrained("my_fine_tuned_t5_small_model")

inputs = loaded_tokenizer(prompt, return_tensors="pt", padding=True).to(loaded_model.device)
output = loaded_model.generate(**inputs, max_new_tokens=500, do_sample=True, top_p=0.5, temperature=0.5)
print(loaded_tokenizer.decode(output[0], skip_special_tokens=True))
#the output for the above input was """We’re proud to unveil the latest fine-tuned versions of Llama 3, purpose-built for tool use and function calling—and they’re already making waves. These models now rank #1 on the Berkeley Function Calling Leaderboard, outperforming every other open-source and proprietary model out there. Crafted on synthetic data and optimized for real-world developer needs, this release wouldn’t be possible without the incredible collaboration between Groq and Glaive teams—huge kudos to everyone involved!""" 
