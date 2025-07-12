import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/ShayanShamsi/prompt_to_linkedin_post/" + splits["train"])
dataset=Dataset.from_pandas(df)
def preprocess_function(examples):
    inputs = ['generate post for:' + str(doc) for doc in examples["prompt"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    output=[f"{doc}" for doc in examples["output"]]
    labels = tokenizer(text_target=output, max_length=1024, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_dataset = dataset.map(preprocess_function, batched=True)
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)

train_dataset = train_test_split['train']
test_dataset = train_test_split['test']
training_args = TrainingArguments(output_dir="./t5-linkedin",eval_strategy="epoch",learning_rate=2e-4,per_device_train_batch_size=8,per_device_eval_batch_size=8,num_train_epochs=1,weight_decay=0.01,save_strategy="epoch",logging_dir="./logs",fp16=True )
trainer = Trainer(model=model,args=training_args,train_dataset=train_dataset,eval_dataset=test_dataset,tokenizer=tokenizer)
trainer.train()
trainer.save_model("my_fine_tuned_t5_small_model")



