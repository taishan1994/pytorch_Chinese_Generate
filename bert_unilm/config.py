import argparse



def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='./checkpoints/',
                            help='the output dir for model checkpoints')

        parser.add_argument('--bert_dir', default='model_hub/chinese-bert-wwm-ext/',
                            help='bert dir for uer')
        parser.add_argument('--data_name', default='cnews',
                            help='data dir for uer')
        parser.add_argument('--model_name', default='bert',
                            help='模型的名字')
        parser.add_argument('--log_dir', default='./logs/',
                            help='log dir for uer')

        # other args
        parser.add_argument('--seed', type=int, default=110, help='random seed')

        parser.add_argument('--max_seq_len', default=256, type=int)

        parser.add_argument('--batch_size', default=16, type=int)

        # train args
        parser.add_argument('--train_epochs', default=15, type=int,
                            help='Max training epoch')
        # 2e-5
        parser.add_argument('--lr', default=3e-5, type=float,
                            help='学习率')

        # 0.5
        parser.add_argument('--max_grad_norm', default=5, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.01, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--weight_decay', default=1e-5, type=float)
        parser.add_argument('--gen_max_len', default=50, type=int,
                            help='生成标题的最大长度')
        parser.add_argument('--continue_train', default=False, type=str2bool)
        parser.add_argument('--do_train', default=False, type=str2bool)
        parser.add_argument('--do_test', default=False, type=str2bool)
        parser.add_argument('--do_generate', default=False, type=str2bool)
        parser.add_argument('--use_tensorboard', default=False, type=str2bool)


        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()