import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.common_utils import sequence_padding
import codecs
import ast

class ListDataset(Dataset):
    def __init__(self, file_path=None, data=None, **kwargs):
        self.kwargs = kwargs
        if isinstance(file_path, (str, list)):
            self.data = self.load_data(file_path)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError('The input args shall be str format file_path / list format dataset')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load_data(file_path):
        return file_path



# 加载数据集
class CnewsDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        examples = []
        with codecs.open(filename, 'r') as f:
            raw_examples = f.readlines()
            print(len(raw_examples))
        # 这里是从json数据中的字典中获取
        for i, item in enumerate(raw_examples):
            item = item.strip().split('	')
            label = item[0]
            text = item[1].split(' ')
            if len(text[0]) > 60:  # 过滤掉标题长度太长的
              continue
            if len(text) == 2 and text[0]:
              examples.append((text[0], text[1])) # (标题，内容)
              
        return examples

class WeiboDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        examples = []
        with codecs.open(filename, 'r') as f:
            raw_examples = f.readlines()
            print(len(raw_examples))
        # 这里是从json数据中的字典中获取
        for i, item in enumerate(raw_examples):
            item = ast.literal_eval(item)
            label = item['tgt_text']
            text = item['src_text']
            if len(label) > 60:  # 过滤掉标题长度太长的
              continue
            examples.append((label, text)) # (标题，内容)
        return examples

class Collate:
  def __init__(self, max_len, device, tokenizer):
      self.maxlen = max_len
      self.device = device
      self.tokenizer = tokenizer

  def collate_fn(self, batch):
      
      batch_text_ids = []
      batch_summary_ids = []
      batch_attention_mask = []
      batch_decoder_attention_mask = []
      batch_text_length = []
      batch_summary_length = []
      batch_titles = []
      for i, (title, content) in enumerate(batch):
          # 对句子对进行编码，注意tokenizer.encode_plus的使用
          text_ids = self.tokenizer.encode(content, max_length=self.maxlen, truncation='only_first')
          batch_text_ids.append(text_ids)
          batch_text_length.append(len(text_ids))
          summary_ids = self.tokenizer.encode(title, max_length=self.maxlen, truncation='only_first')
          batch_summary_length.append(len(summary_ids))
          batch_summary_ids.append(summary_ids)
          attention_mask = [1] * len(text_ids)
          batch_attention_mask.append(attention_mask)
          decoder_attention_mask = [1] * len(summary_ids)
          batch_decoder_attention_mask.append(decoder_attention_mask)
          batch_titles.append(title)
      text_max_length = max(batch_text_length)
      summary_max_length = max(batch_summary_length)
      batch_text_ids = torch.tensor(sequence_padding(batch_text_ids, length=text_max_length), dtype=torch.long, device=self.device)
      batch_summary_ids = torch.tensor(sequence_padding(batch_summary_ids, length=summary_max_length), dtype=torch.long, device=self.device)
      batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask, length=text_max_length), dtype=torch.long, device=self.device)
      batch_decoder_attention_mask = torch.tensor(sequence_padding(batch_decoder_attention_mask, length=summary_max_length), dtype=torch.long, device=self.device)
      return batch_text_ids, batch_summary_ids, batch_attention_mask, batch_decoder_attention_mask, batch_titles

if __name__ == "__main__":
  from transformers import BertTokenizer
  from utils import common_utils
  max_len = 512
  
  tokenizer = common_utils.T5PegasusTokenizer.from_pretrained('model_hub/chinese_t5_pegasus_small/vocab.txt')
  train_dataset = CnewsDataset(file_path='data//cnews/cnews.train.txt')
  print(train_dataset[0])

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  collate = Collate(max_len=max_len, device=device, tokenizer=tokenizer)
  # collate.collate_fn(train_dataset[:16])
  batch_size = 2
  train_dataset = train_dataset[:10]
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn) 

  for i, batch in enumerate(train_dataloader):
    for j in range(len(batch)-1):
      print(batch[j].shape)
    print(batch[-1])
    break