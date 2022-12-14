import numpy as np
import pandas as pd
from tqdm import tqdm

data = pd.read_csv("item_desc_dataset.txt",
                   header=None,
                   sep="\t",
                   names=["title", "desc"])

np.random.seed(123)
# 随机采样10万条数据
train_data = data.sample(100000)
dev_data = data.sample(100)


def save_data(in_data, out_path):
    i = 0
    fp = open(out_path, "w", encoding="utf-8")
    for d in tqdm(in_data.iterrows(), total=len(in_data)):
        d = d[1]
        title = d["title"]
        desc = d["desc"]
        tmp = {}
        tmp["src_text"] = title
        tmp["tgt_text"] = desc
        # 我们要根据标题来生成描述
        fp.write(str(tmp) + "\n")

    fp.close()


save_data(train_data, "train_data.json")
save_data(dev_data, "test_data.json")