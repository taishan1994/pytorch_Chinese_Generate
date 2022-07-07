import os
import json
import torch
import torch.nn as nn
from transformers import BertConfig 
from .model_unilm import BojoneModel


def build_model(config_path=None,
        checkpoint_path=None,
        keep_tokens=None, # 用于减少词表
        model='bert'):

  if model == 'bert':
    bert_config = BertConfig.from_pretrained(config_path)
    tmp_state_dict = torch.load(os.path.join(checkpoint_path), map_location="cpu")

    if keep_tokens:
      tmp_state_dict['bert.embeddings.word_embeddings.weight'] = torch.index_select(tmp_state_dict['bert.embeddings.word_embeddings.weight'], 0, torch.tensor(keep_tokens, dtype=torch.long))
    # 这里将需要的参数都映射成bert开头的，并和自定义模型相对应
    tmp_state_dict["bert.cls.transform.dense.weight"] = tmp_state_dict["cls.predictions.transform.dense.weight"]
    tmp_state_dict["bert.cls.transform.dense.bias"] = tmp_state_dict["cls.predictions.transform.dense.bias"]
    tmp_state_dict["bert.cls.transform.LayerNorm.weight"] = tmp_state_dict["cls.predictions.transform.LayerNorm.weight"]
    tmp_state_dict["bert.cls.transform.LayerNorm.bias"] = tmp_state_dict["cls.predictions.transform.LayerNorm.bias"]
    bert_model = BojoneModel.from_pretrained(pretrained_model_name_or_path=checkpoint_path, config=bert_config, state_dict=tmp_state_dict)

  return bert_model, bert_config