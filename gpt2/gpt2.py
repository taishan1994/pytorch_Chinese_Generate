# coding=utf-8
"""
!pip install --upgrade transformers
!pip install tensorboardX
!pip install rouge
!pip install datasets
"""

import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
  BertTokenizer, 
  GPT2LMHeadModel, 
  TextGenerationPipeline,
  TrainingArguments,
  Trainer
)

def get_eb_data(path):
  with open(path, "r", encoding="utf-8") as fp:
    data = fp.read().strip().split("\n")
  for d in data:
    print(d)
    break
  data = [eval(x)["tgt_text"] for x in data]
  return data
  


class GPT2Dataset(Dataset):
  def __init__(self, data, tokenizer, max_seq_len):
    self.data = data
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len 

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    text = self.data[idx]
    inputs = self.tokenizer(text,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
    input_ids = inputs["input_ids"]
    label_ids = input_ids[0, :][input_ids[0, :] > 0][1:]
    label_ids = torch.tensor([label_ids.numpy().tolist() + [-100] * (self.max_seq_len - len(label_ids))])
    inputs["labels"] = label_ids
    return inputs


def main():
  data_path = "data/eb"
  train_path = os.path.join(data_path, "train_data.json")
  test_path = os.path.join(data_path, "test_data.json")
  train_data = get_eb_data(train_path)
  test_data = get_eb_data(test_path)


  max_seq_len = 256
  tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
  model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

  train_dataset = GPT2Dataset(train_data, tokenizer, max_seq_len)
  test_dataset = GPT2Dataset(train_data, tokenizer, max_seq_len)

  training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    save_total_limit=1,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    prediction_loss_only=True,
  )

  trainer.train()

  # trainer.save_model()

  """
  https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface
  from transformers import pipeline
  chef = pipeline('text-generation',model='./gpt2-gerchef', tokenizer='anonymous-german-nlp/german-gpt2',config={'max_length':800})
  result = chef('Zuerst Tomaten')[0]['generated_text']
  """

if __name__ == "__main__":
  main()
  
  """
  from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
  import torch
  tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
  model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
  model2 = GPT2LMHeadModel.from_pretrained("output/checkpoint-6000/")
  text_generator = TextGenerationPipeline(model2, tokenizer)   
  text_generator("型男时尚秋季撩妹90后棉衣，", max_length=256, do_sample=True)
  """
