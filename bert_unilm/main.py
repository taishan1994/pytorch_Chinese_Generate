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
from transformers import BertTokenizer
from tensorboardX import SummaryWriter
# import config
from data_loader import CnewsDataset, Collate, WeiboDataset, DuilianDataset
from utils.common_utils import set_seed, set_logger
from utils.train_utils import build_optimizer_and_scheduler, save_model, load_model_and_parallel
from utils.generate_utils import beam_search
from utils.metric_utils import evaluate
from models import model_utils
import config

args = config.Args().get_parser()
set_seed(args.seed)
logger = logging.getLogger(__name__)

if args.use_tensorboard:
    writer = SummaryWriter(log_dir='./tensorboard')


class BertForSeq2Seq:
    def __init__(self, args, train_loader, tmp_test_dataset, tokenizer, idx2tag, model, device):
        self.train_loader = train_loader
        self.tmp_test_dataset = tmp_test_dataset
        self.tokenizer = tokenizer
        self.args = args
        self.idx2tag = idx2tag
        self.model = model
        self.device = device
        if train_loader is not None:
            self.t_total = len(self.train_loader) * args.train_epochs
            self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, model, self.t_total)

    def train(self):
        # Train
        global_step = 0
        self.model.zero_grad()
        eval_steps = 100  # 每多少个step打印损失及进行验证
        best_rouge_1 = 0
        best_rouge_2 = 0
        for epoch in range(1, self.args.train_epochs + 1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for batch in batch_data:
                    batch = batch.to(self.device)
                input_ids = batch_data[0]
                tmp_input_ids = input_ids
                input_ids = input_ids[:, :-1]
                token_type_ids = batch_data[1]
                labels_mask = token_type_ids[:, 1:].contiguous()

                token_type_ids = token_type_ids[:, :-1]
                labels = tmp_input_ids[:, 1:].contiguous()  # 错开输入为标签
                # print(labels_mask[0])
                # print(token_type_ids[0])
                # print(labels[0])

                # print(input_ids.shape, token_type_ids.shape, labels.shape)
                _, loss = self.model(input_ids, token_type_ids=token_type_ids, labels=labels, labels_mask=labels_mask)

                # loss.backward(loss.clone().detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                logger.info('【train】 epoch:{}/{} {}/{} loss:{:.4f}'.format(epoch, self.args.train_epochs, global_step,
                                                                           self.t_total, loss.item()))
                if args.use_tensorboard:
                    writer.add_scalar('data/loss', loss.item(), global_step)
                global_step += 1
                if global_step % eval_steps == 0:

                    metrics = self.test(self.model, self.tokenizer, self.tmp_test_dataset, gen_max_len=50)
                    logger.info('=' * 100)
                    logger.info(metrics)
                    logger.info('=' * 100)
                    rouge_1 = metrics['rouge-1']
                    rouge_2 = metrics['rouge-2']
                    if best_rouge_1 < rouge_1 and best_rouge_2 < rouge_2:
                        save_model(self.args, self.model, model_name + '_' + data_name, global_step)
                        best_rouge_1 = rouge_1
                        best_rouge_2 = rouge_2
                        logger.info("best: rough-1：{} rough-2：{}".format(best_rouge_1, best_rouge_2))
                        logger.info('=' * 100)

    def test(self, model, tokenizer, tmp_test_dataset, gen_max_len=30):
        model.eval()
        true_titles = []
        pred_titles = []
        with torch.no_grad():
            for i, raw_text in enumerate(tmp_test_dataset):
                pred_title = self.generate(raw_text, model, tokenizer, gen_max_len=gen_max_len)
                if i < 5:
                    print("真实：", raw_text[0])
                    print("预测：", "".join(pred_title.split(" ")))
                    print("=" * 100)
                true_titles.append(raw_text[0])
                pred_titles.append("".join(pred_title.split(" ")))
        metrics = evaluate(true_titles, pred_titles)
        # print(metrics)
        return metrics

    def generate_by_beam_search(self, raw_text, model, tokenizer, gen_max_len=50, beam_size=3):
        """gen_max_len：最大生成长度
        """
        model.eval()
        # 第一个句子的最大长度
        first_length = self.args.max_seq_len - gen_max_len - 2
        label, text = raw_text
        # print(text)
        # print(label)
        sep_id = 102
        if len(text) > first_length:
            text = text[:first_length]
        with torch.no_grad():
            result = []
            encode_dict = tokenizer.encode_plus(text=text,
                                                max_length=first_length,
                                                padding='max_length',
                                                return_token_type_ids=True,
                                                return_tensors="pt")
            token_ids = encode_dict['input_ids'].to(self.device)
            token_type_ids = encode_dict['token_type_ids'].to(self.device)

            result = beam_search(model, token_ids, token_type_ids, first_length, beam_size=2, device=self.device)
            # print(tokenizer.decode(result))
            return tokenizer.decode(result)

    def generate(self, raw_text, model, tokenizer, gen_max_len=30):
        """gen_max_len：最大生成长度
        """
        model.eval()
        # 第一个句子的最大长度
        first_length = self.args.max_seq_len - gen_max_len - 2
        label, text = raw_text
        sep_id = 102
        # if len(text) > first_length:
        # text = text[:first_length]
        with torch.no_grad():
            encode_dict = tokenizer.encode_plus(text=text,
                                                max_length=first_length,
                                                truncation="longest_first",
                                                return_token_type_ids=True,
                                                return_tensors="pt")
            token_ids = encode_dict['input_ids'].to(self.device)
            token_type_ids = encode_dict['token_type_ids'].to(self.device)
            # print(token_ids.shape)
            # print(token_type_ids.shape)
            # print(token_ids.shape, token_type_ids.shape)
            output = []
            for step in range(gen_max_len):
                predicates = model(token_ids, token_type_ids=token_type_ids)
                # 贪心解码
                out_id = np.argmax(predicates[:, -1, :].cpu().numpy(), -1).tolist()
                # print(out_id)
                if out_id == [sep_id]:
                    break
                if out_id != [100]:  # 如果不是[UNK]
                    output.extend(out_id)
                # 加上该步
                # print(token_ids.shape, torch.tensor([out_id]).shape)
                # print(token_type_ids.shape, torch.tensor([[1]]).shape)

                token_ids = torch.cat((token_ids, torch.tensor([out_id]).to(self.device)), 1)
                token_type_ids = torch.cat((token_type_ids, torch.tensor([[1]]).to(self.device)), dim=1)
        # print(output)
        # print(tokenizer.decode(output))
        return tokenizer.decode(output)


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
    for i, j in enumerate(vocab):
        word2id[j] = i
        id2word[i] = j

    bert_seq2seq, bert_config = model_utils.build_model(
        config_path,
        checkpoint_path,
        keep_tokens=None,
        model="bert")

    if args.continue_train:  # 是否加载训练好的模型继续训练
        model_path = './checkpoints/{}_{}/model.pt'.format(model_name, data_name)
        bert_seq2seq = load_model_and_parallel(bert_seq2seq, ckpt_path=model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_seq2seq.to(device)

    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    collate = Collate(max_len=max_seq_len, device=device, tokenizer=tokenizer)

    if data_name == "cnews":
        train_dataset = CnewsDataset(file_path='data/cnews/cnews.train.txt')
        print(train_dataset[0])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate.collate_fn)

        dev_dataset = CnewsDataset(file_path='data/cnews/cnews.val.txt')
        print(dev_dataset[0])
        # dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)

        test_dataset = CnewsDataset(file_path='data/cnews/cnews.test.txt')
        tmp_test_dataset = random.sample(list(test_dataset), 100)  # 随机选择多少条数据
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
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)
    elif data_name == "duilian":
        train_dataset = DuilianDataset(file_path=['data/duilian/in.txt', 'data/duilian/out.txt'])
        train_dataset = train_dataset[1000:]
        test_dataset = train_dataset[:1000]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate.collate_fn)

        # dev_dataset = WeiboDataset(file_path='data/cnews/cnews.val.txt')
        # print(dev_dataset[0])
        # dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)

        tmp_test_dataset = random.sample(list(test_dataset), 100)  # 随机选择多少条数据
        print(tmp_test_dataset[0])
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate.collate_fn)

    bertForSeq2Seq = BertForSeq2Seq(args, train_loader, tmp_test_dataset, tokenizer, id2word, bert_seq2seq, device)
    # 训练
    # ================================
    if args.do_train:
        bertForSeq2Seq.train()
    # ================================

    # 加载训练好的模型
    # ================================
    if args.do_test or args.do_generate:
        model_path = './checkpoints/{}_{}/model.pt'.format(model_name, data_name)
        bert_seq2seq = load_model_and_parallel(bert_seq2seq, ckpt_path=model_path)
        bert_seq2seq.to(device)
        # ================================

        # 测试
        # ================================
        if args.do_test:
            metrics = bertForSeq2Seq.test(bert_seq2seq, tokenizer, tmp_test_dataset, gen_max_len=args.gen_max_len)
            print(metrics)
        # ================================

        # 预测
        # ================================
        if args.do_generate:
            texts = tmp_test_dataset[:10]
            for text in texts:
                print("文本：", text[1])
                print("真实：", text[0])
                print("预测：", "".join(
                    bertForSeq2Seq.generate(text, bert_seq2seq, tokenizer, gen_max_len=args.gen_max_len).split(" ")))
                # print("预测：", "".join(bertForSeq2Seq.generate_by_beam_search(text, bert_seq2seq, tokenizer).split(" ")))
                print("=" * 100)
        # ================================

