from rouge import Rouge
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

rouge = Rouge()
smooth = SmoothingFunction().method1
best_bleu = 0.
def evaluate(true_title, pred_title):
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for t_title, p_title in tqdm(zip(true_title, pred_title), total=len(true_title), ncols=100):
        total += 1
        t_title = ' '.join(t_title).lower()
        p_title = ' '.join(p_title).lower()
        # print(t_title, t_title)
        if p_title.strip():
            scores = rouge.get_scores(hyps=p_title, refs=t_title)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[t_title.split(' ')],
                hypothesis=p_title.split(' '),
                smoothing_function=smooth
            )
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    return {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l,
        'bleu': bleu,
    }
