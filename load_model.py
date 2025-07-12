prompt = "Generate a LinkedIn post announcing that Daphne Koller, founder and CEO of insitro, will be speaking at the #Pezcoller23 Symposium on October 19th. The post should mention that she will present cutting-edge #MachineLearning models designed to create meaningful representations of human pathophysiological states. Highlight that these models enable the identification of novel targets and biomarkers for coherent patient segments, thus accelerating the development of effective therapeutic interventions. Emphasize that this is the last week to register for the event and provide the registration link: https://lnkd.in/etYSdNMX. Also, note that this is a European School of Oncology recommended event and express enthusiasm for participating with world-leading scientists in Oncology, all striving towards #curecancer."
#prompt=str(input()) for getting the prompt from user
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your fine-tuned model and tokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained("my_fine_tuned_t5_small_model")
loaded_model = AutoModelForSeq2SeqLM.from_pretrained("my_fine_tuned_t5_small_model")

inputs = loaded_tokenizer(prompt, return_tensors="pt", padding=True).to(loaded_model.device)
output = loaded_model.generate(**inputs, max_new_tokens=500, do_sample=True, top_p=0.5, temperature=0.5)
print(loaded_tokenizer.decode(output[0], skip_special_tokens=True))
