这里记录的是另外一种方法,使用[UER-py](https://github.com/dbiir/UER-py/wiki/%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B%E4%BD%BF%E7%94%A8%E7%A4%BA%E4%BE%8B)<br>
数据可以去这里下载:git clone https://www.modelscope.cn/datasets/minisnow/couplet_samll.git
### 步骤
1、下载UER-PY
```python
git clone https://github.com/dbiir/UER-py.git
```
2、获取数据
```python

cd UER-py/corpora/
git clone https://www.modelscope.cn/datasets/minisnow/couplet_samll.git
```
然后使用以下代码生成couplet.txt，其中一行是一句对联：
```python
def get_couplet_data(path):
  import pandas as pd
  data = pd.read_csv(path)
  data["text"] = data["text1"] + "-" + data["text2"] + "。"
  data = data["text"].values.tolist()
  return data

path = "couplet_samll/train.csv"
with open("couplet.txt", "w", encoding="utf-8") as fp:
  fp.write("\n".join(get_couplet_data(path)))
```
3、数据预处理
回到UER-py目录下，执行：
```python
python preprocess.py --corpus_path corpora/couplet.txt \
                      --vocab_path models/google_zh_vocab.txt \
                      --dataset_path couplet_lm_seq32_dataset.pt \
                      --seq_length 32 --processes_num 32 --data_processor lm 

```
4、开始继续预训练
首先下载预训练模型：```wget https://huggingface.co/uer/gpt2-chinese-couplet/resolve/main/pytorch_model.bin```
```python
CUDA_VISIBLE_DEVICES=0 python pretrain.py --dataset_path couplet_lm_seq32_dataset.pt \
                    --vocab_path models/google_zh_vocab.txt \
                    --config_path models/gpt2/config.json \
                    --pretrained_model_path pytorch_model.bin \
                    --output_model_path models/couplet_gpt2_seq32_model.bin \
                    --world_size 1 --gpu_ranks 0 \
                    --total_steps 5000 --save_checkpoint_steps 1000 --report_steps 100 \
                    --learning_rate 1e-4 --batch_size 128
```
注意这里使用单卡训练，设置world_size为GPU的数目，显卡为第0块显卡。
5、转换为hungging face对应的模型格式
```python
python scripts/convert_gpt2_from_uer_to_huggingface.py \
      --input_model_path /content/drive/MyDrive/UER-py/models/couplet_gpt2_seq32_model.bin-5000 \
      --output_model_path /content/drive/MyDrive/UER-py/models/gpt2/pytorch_model.bin \
      --layers_num 12
```
6、加载模型并进行预测
```python
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline, AutoConfig
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-couplet")
config = AutoConfig.from_pretrained("/content/drive/MyDrive/UER-py/models/gpt2/config.json")
config.vocab_size = len(tokenizer)
print(config)
model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/UER-py/models/gpt2/", config=config)
text_generator = TextGenerationPipeline(model, tokenizer)   
text = "泼墨吟诗，银发人生添雅兴"
text = " ".join([i for i in text]) + " -"
length = len(text)
text = "[CLS]" + text
print(text)
print(text_generator(text, max_length=length+2, do_sample=True))
```
