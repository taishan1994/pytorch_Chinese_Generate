import json
import pandas as pd
import numpy as np
import lawrouge
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline, pipeline
from transformers import HfArgumentParser, TrainingArguments, Trainer, set_seed
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset

do_train = False
do_predict = True

if do_train:
  # 加载数据集
  train_path = "data/couplet_small/train.csv"
  train_dataset = Dataset.from_csv(train_path)
  test_path = "data/couplet_small/test.csv"
  test_dataset = Dataset.from_csv(test_path)

  # 转换为模型需要的格式
  def tokenize_dataset(tokenizer, dataset, max_len):
    def convert_to_features(batch):
      text1 = batch["text1"]
      text2 = batch["text2"]
      inputs = tokenizer.batch_encode_plus(
        text1,
        max_length=max_len,
        padding="max_length",
        truncation=True,
      )
      targets = tokenizer.batch_encode_plus(
        text2,
        max_length=max_len,
        padding="max_length",
        truncation=True,
      )
      outputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "target_ids": targets["input_ids"],
        "target_attention_mask": targets["attention_mask"]
      }
      return outputs
    
    dataset = dataset.map(convert_to_features, batched=True)
    # Set the tensor type and the columns which the dataset should return
    columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
    dataset.with_format(type='torch', columns=columns)
    dataset = dataset.rename_column('target_ids', 'labels')
    dataset = dataset.rename_column('target_attention_mask', 'decoder_attention_mask')
    dataset = dataset.remove_columns(['text1', 'text2'])
    return dataset

  tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
  max_len = 24
  train_data = tokenize_dataset(tokenizer, train_dataset, max_len)
  test_data = tokenize_dataset(tokenizer, test_dataset, max_len)

  # model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
  model = AutoModelForSeq2SeqLM.from_pretrained("fnlp/bart-base-chinese")

  def compute_metrics(outputs):
    predictions, labels = outputs
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
    decoded_labels = ["".join(label.replace(" ", "")) for label in decoded_labels]
    decoded_preds = [decoded_preds[i][:len(text)] for i, text in enumerate(decoded_labels)]
    print("\n")
    print(decoded_preds)
    print(decoded_labels)
    rouge = lawrouge.Rouge()
  
    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}

    result = {key: value * 100 for key, value in result.items()}
    return result

  training_args = Seq2SeqTrainingArguments(
    output_dir='./results',         # output directory 结果输出地址
    num_train_epochs=10,          # total # of training epochs 训练总批次
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=128,  # batch size per device during training 训练批大小
    per_device_eval_batch_size=64,   # batch size for evaluation 评估批大小
    logging_dir='./logs',    # directory for storing logs 日志存储位置
    learning_rate=3e-4,             # 学习率
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=1,
    predict_with_generate=True,
    generation_max_length=max_len, # 生成的最大长度
    generation_num_beams=1, # beam search，大于1为集束搜索
    load_best_model_at_end=True,
    metric_for_best_model="rouge-1"
  )

  trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
  )
  train_result = trainer.train()
  # 打印测试集上的结果
  print(trainer.evaluate(test_data))
  # 保存最优模型
  # trainer.save_model("results/best")

  
if do_predict:
  test_path = "data/couplet_small/test.csv"
  test_data = pd.read_csv(test_path)
  texts = test_data["text1"].values.tolist()
  labels = test_data["text2"].values.tolist()
  tokenizer = BertTokenizer.from_pretrained("fnlp/bart-large-chinese")
  max_len = 24

  # 方法一
  generator = pipeline('text2text-generation', model='results/checkpoint-1000/', tokenizer=tokenizer)
  for text, label in zip(texts, labels):
    print("上联：", text)
    print("真实下联：", label)
    print("预测下联：", "".join(generator(text)[0]["generated_text"].split(" ")))
    print("="*100)

  # 方法二
  # 加载训练好的模型
  model = BartForConditionalGeneration.from_pretrained("results/checkpoint-1000/")
  model = model.to("cuda")
  # 从测试集中挑选4个样本
  inputs = tokenizer(
          texts,
          padding="max_length",
          truncation=True,
          max_length=max_len,
          return_tensors="pt",
      )
  input_ids = inputs.input_ids.to(model.device)
  attention_mask = inputs.attention_mask.to(model.device)
  # 生成
  outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_len)
  # 将token转换为文字
  output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  output_str = [s.replace(" ","") for s in output_str]
  for text, label, pred in zip(texts, labels, output_str):
    print("上联：", text)
    print("真实下联：", label)
    print("预测下联：", pred)
    print("="*100)

  # 方法三
  model = BartForConditionalGeneration.from_pretrained("results/checkpoint-1000/")
  generator = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)
  results = generator(texts)
  for text, label, res in zip(texts, labels, results):
    print("上联：", text)
    print("真实下联：", label)
    print("预测下联：", "".join(res["generated_text"].split(" ")))
    print("="*100)


  
