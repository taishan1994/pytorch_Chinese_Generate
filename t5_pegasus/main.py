import sys
sys.path.append('.')
import json
import os
import torch.nn as nn
import logging
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, MT5ForConditionalGeneration
from tensorboardX import SummaryWriter
# import config
from data_loader import CnewsDataset, Collate, WeiboDataset
from utils.common_utils import set_seed, set_logger, T5PegasusTokenizer, sequence_padding
from utils.train_utils import build_optimizer_and_scheduler, save_model, load_model_and_parallel
from utils.generate_utils import beam_search
from utils.metric_utils import evaluate
# from models.t5_pegasus import T5ForGenerate
import config


args = config.Args().get_parser()
set_seed(args.seed)
logger = logging.getLogger(__name__)

if args.use_tensorboard:
  writer = SummaryWriter(log_dir='./tensorboard')


class T5ForSeq2Seq:
    def __init__(self, args, train_loader, dev_loader, tokenizer, idx2tag, model, device):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.tokenizer = tokenizer
        self.args = args
        self.idx2tag = idx2tag
        self.model = model
        self.device = device
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        if train_loader is not None:
          self.t_total = len(self.train_loader) * args.train_epochs
          self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = 100 #每多少个step打印损失及进行验证
        best_rouge_1 = 0
        best_rouge_2 = 0
        for epoch in range(1, self.args.train_epochs+1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for batch in batch_data[:-1]:
                    batch = batch.to(self.device)
                text_ids = batch_data[0]
                summary_ids = batch_data[1]
                attention_mask = batch_data[2]
                decoder_attention_mask = batch_data[3]
                output = self.model(input_ids=text_ids, 
                           decoder_input_ids=summary_ids,
                           attention_mask=attention_mask,
                           decoder_attention_mask=decoder_attention_mask)
                output = output[0]
                # decoder_attention_mask
                mask = decoder_attention_mask[:, 1:].reshape(-1).bool()
                output = output[:, :-1]
                output = output.reshape((-1, output.size(-1)))[mask]
                labels = summary_ids[:, 1:].reshape(-1)[mask]
                loss = self.loss_fct(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{}/{} {}/{} loss:{:.4f}'.format(epoch, self.args.train_epochs, global_step, self.t_total, loss.item()))
                
                if args.use_tensorboard:
                  writer.add_scalar('data/loss', loss.item(), global_step)
                global_step += 1
                if global_step % eval_steps == 0:
                
                  metrics = self.test(self.model, self.tokenizer, self.dev_loader, gen_max_len=args.gen_max_len)
                  logger.info('='*100)
                  logger.info(metrics)
                  logger.info('='*100)
                  rouge_1 = metrics['rouge-1']
                  rouge_2 = metrics['rouge-2']
                  if best_rouge_1 < rouge_1 and best_rouge_2 < rouge_2:
                      save_model(self.args, self.model, model_name + '_' + data_name, global_step)
                      best_rouge_1 = rouge_1
                      best_rouge_2 = rouge_2
                      logger.info("best: rough-1：{} rough-2：{}".format(best_rouge_1, best_rouge_2))
                      logger.info('='*100)
                
                

    def test(self, model, tokenizer, dev_loader, gen_max_len=30):
        model.eval()
        true_titles = []
        pred_titles = []
        with torch.no_grad():
            for step, batch_data in enumerate(dev_loader):
              for batch in batch_data[:-1]:
                  batch = batch.to(self.device)
              text_ids = batch_data[0]
              attention_mask = batch_data[2]
              title = batch_data[-1]
              gen = model.generate(max_length=args.gen_max_len,
                          eos_token_id=tokenizer.sep_token_id,
                          decoder_start_token_id=tokenizer.cls_token_id,
                          input_ids=text_ids,
                          attention_mask=attention_mask)
              gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
              gen = [item.replace(' ', '') for item in gen]
              if step == 0:
                batch_size = text_ids.shape[0]
                for i in range(batch_size):
                  if i < 10:
                    print("真实：", title[i])
                    print("预测：", gen[i])
                    print("="*100)
              true_titles.extend(title)
              pred_titles.extend(gen)
        metrics = evaluate(true_titles, pred_titles)
        # print(metrics)
        return metrics
            

    def generate(self, raw_text, model, tokenizer, args):   
        """gen_max_len：最大生成长度
        """     
        model.eval()
        pred_results = []
        with torch.no_grad():
          batch_input_ids = []
          batch_attention_mask = []
          batch_max_length = []
          for text in raw_text:
            text_ids = tokenizer.encode(text, max_length=args.max_seq_len, truncation='only_first')
            attention_mask = [1] * len(text_ids)
            if len(batch_input_ids) < args.batch_size:
              batch_input_ids.append(text_ids)
              batch_attention_mask.append(attention_mask)
            else:
              text_max_length = max(batch_max_length)
              batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=args.max_seq_len), dtype=torch.long, device=self.device)
              batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask, length=args.max_seq_len), dtype=torch.long, device=self.device)
              gen = model.generate(max_length=args.gen_max_len,
                          eos_token_id=tokenizer.sep_token_id,
                          decoder_start_token_id=tokenizer.cls_token_id,
                          input_ids=batch_input_ids,
                          attention_mask=batch_attention_mask)
              gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
              gen = [item.replace(' ', '') for item in gen]
              pred_results.extend(gen)
              batch_input_ids = []
              batch_attention_mask = []
              batch_max_length = []
          if batch_input_ids:
            batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, length=args.max_seq_len), dtype=torch.long, device=self.device)
            batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask, length=args.max_seq_len), dtype=torch.long, device=self.device)
            gen = model.generate(max_length=args.gen_max_len,
                        eos_token_id=tokenizer.sep_token_id,
                        decoder_start_token_id=tokenizer.cls_token_id,
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            pred_results.extend(gen)

        return pred_results
    

if __name__ == '__main__':
    data_name = args.data_name
    model_name = args.model_name

    set_logger(os.path.join(args.log_dir, '{}_{}.log'.format(model_name, data_name)))

    # class Args:
    #     lr = 1e-5
    #     train_epochs = 40
    #     weight_decay = 1e-5
    #     adam_epsilon = 1e-8
    #     batch_size = 16
    #     warmup_proportion = 0.01
    #     max_grad_norm = 5.0
    #     max_seq_len = 256
    #     output_dir = "./checkpoints/"
    #     continue_train = False
    #     bert_dir = 'model_hub/chinese-bert-wwm-ext/'


    bert_dir = args.bert_dir
    config_path = os.path.join(bert_dir, 'config.json')
    checkpoint_path = os.path.join(bert_dir, 'pytorch_model.bin')
    vocab_path = os.path.join(bert_dir, 'vocab.txt')
    
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size
    
    word2id = {}
    id2word = {}

    with open(vocab_path, 'r', encoding='utf-8') as fp:
      vocab = fp.read().strip().split('\n')
    for i,j in enumerate(vocab):
      word2id[j] = i
      id2word[i] = j

    tokenizer = T5PegasusTokenizer.from_pretrained(args.bert_dir)
    t5ForGenerate = MT5ForConditionalGeneration.from_pretrained(args.bert_dir)

      
    if args.continue_train:  # 是否加载训练好的模型继续训练
        model_path = './checkpoints/{}_{}/model.pt'.format(model_name, data_name)
        t5ForGenerate = load_model_and_parallel(t5ForGenerate, ckpt_path=model_path)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t5ForGenerate.to(device)

    collate = Collate(max_len=max_seq_len, device=device, tokenizer=tokenizer)

    if data_name == "cnews": 
        train_dataset = CnewsDataset(file_path='data/cnews/cnews.train.txt')
        print(train_dataset[0])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate.collate_fn) 

        dev_dataset = CnewsDataset(file_path='data/cnews/cnews.val.txt')
        print(dev_dataset[0])
        tmp_dev_dataset = random.sample(list(dev_dataset), 100) # 随机选择多少条数据
        dev_loader = DataLoader(tmp_dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn) 

        test_dataset = CnewsDataset(file_path='data/cnews/cnews.test.txt')
        tmp_test_dataset = random.sample(list(test_dataset), 5)  # 随机选择多少条数据
        print(tmp_test_dataset[0])
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn) 
    elif data_name == "weibo":
        train_dataset = WeiboDataset(file_path='data/weibo/train_data.json')
        print(train_dataset[0])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate.collate_fn) 

        # dev_dataset = WeiboDataset(file_path='data/cnews/cnews.val.txt')
        # print(dev_dataset[0])
        # dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn) 

        test_dataset = WeiboDataset(file_path='data/weibo/test_data.json')
        tmp_test_dataset = random.sample(list(test_dataset), 100)  # 随机选择多少条数据
        print(tmp_test_dataset[0])
        test_loader = DataLoader(tmp_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn) 

        dev_loader = test_loader

    t5ForSeq2Seq = T5ForSeq2Seq(args, train_loader, dev_loader, tokenizer, id2word, t5ForGenerate, device)
    # 训练
    # ================================
    if args.do_train:
      t5ForSeq2Seq.train()
    # ================================

    # 加载训练好的模型
    # ================================
    if args.do_test or args.do_generate:
      model_path = './checkpoints/{}_{}/model.pt'.format(model_name, data_name)
      t5ForGenerate = load_model_and_parallel(t5ForGenerate, ckpt_path=model_path)
      t5ForGenerate.to(device)
      # ================================

      # 测试
      # ================================
      if args.do_test:
        metrics = t5ForSeq2Seq.test(t5ForGenerate, tokenizer, dev_loader, gen_max_len=args.gen_max_len)
        logger.info('='*100)
        logger.info(metrics)
        logger.info('='*100)   
      # ================================

      # 预测
      # ================================
      if args.do_generate:
        texts = tmp_test_dataset
        contents = [i[1] for i in texts]
        titles = [i[0] for i in texts]
        results = t5ForSeq2Seq.generate(contents, t5ForGenerate, tokenizer, args)
        for content, title, result in zip(contents, titles, results):
          print("文本：", content)
          print("真实：", title)
          print("预测：", result)
          print("="*100)
      # ================================
        
