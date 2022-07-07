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
                examples.append((text[0], text[1]))  # (标题，内容)
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
            examples.append((label, text))  # (标题，内容)
        return examples


class DuilianDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        in_filename = filename[0]
        out_filename = filename[1]
        examples = []
        ins = []
        with codecs.open(in_filename, 'r') as f:
            in_raw_examples = f.readlines()
        with codecs.open(out_filename, 'r') as f:
            out_raw_examples = f.readlines()
        print(len(in_raw_examples))
        print(len(out_raw_examples))
        # 这里是从json数据中的字典中获取
        for i, (item1, item2) in enumerate(zip(in_raw_examples, out_raw_examples)):
            item1 = "".join(item1.strip().split(" "))
            item2 = "".join(item2.strip().split(" "))
            if len(item2) > 60:  # 过滤掉标题长度太长的
                continue
            examples.append((item2, item1))  # (标题，内容)
        return examples


class Collate:
    def __init__(self, max_len, device, tokenizer):
        self.maxlen = max_len
        self.device = device
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        batch_token_ids = []
        batch_token_type_ids = []
        for i, (title, content) in enumerate(batch):
            # 对句子对进行编码，注意tokenizer.encode_plus的使用
            output = self.tokenizer.encode_plus(
                text=content,
                text_pair=title,
                max_length=self.maxlen,
                padding="max_length",
                truncation="longest_first",
                return_token_type_ids=True,
            )
            token_ids = output['input_ids']
            token_type_ids = output["token_type_ids"]
            batch_token_ids.append(token_ids)
            batch_token_type_ids.append(token_type_ids)
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=self.maxlen), dtype=torch.long,
                                       device=self.device)
        token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids, length=self.maxlen), dtype=torch.long,
                                      device=self.device)
        # print(batch_token_ids.shape)
        # print(token_type_ids.shape)
        return batch_token_ids, token_type_ids


if __name__ == "__main__":
    from transformers import BertTokenizer

    max_len = 512
    tokenizer = BertTokenizer.from_pretrained('model_hub/chinese-bert-wwm-ext/vocab.txt')
    train_dataset = CnewsDataset(file_path='data//cnews/cnews.train.txt')
    print(train_dataset[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate = Collate(max_len=max_len, device=device, tokenizer=tokenizer)
    # collate.collate_fn(train_dataset[:16])
    batch_size = 2
    train_dataset = train_dataset[:10]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)

    for i, batch in enumerate(train_dataloader):
        for j in range(len(batch)):
            print(batch[j].shape)
        break