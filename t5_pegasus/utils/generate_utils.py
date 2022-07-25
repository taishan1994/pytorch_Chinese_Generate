import torch


def beam_search(model, 
        token_ids, 
        token_type_ids, 
        gen_max_len=30,
        beam_size=1, 
        device="cpu"):
    """
    beam-search操作
    """
    sep_id = 102
    # 用来保存输出序列
    output_ids = [[]]
    # 用来保存累计得分
    output_scores = torch.zeros(token_ids.shape[0], device=device)
    for step in range(gen_max_len):
        
        scores = model(token_ids, token_type_ids=token_type_ids)
        # print(scores.shape)
        # scores: 1*226*vocab_size
        if step == 0:
            # 重复beam-size次 输入ids。为什么需要重复三次？？？
            token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
            token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
        ## 计算log 分值 (beam_size, vocab_size)
        logit_score = torch.log_softmax(scores, dim=-1)[:, -1]
        # [:, -1] 取出第2个维度上最后一个向量，最后维度：[1, 21128]
        logit_score = output_scores.view(-1, 1) + logit_score # 累计得分
        # output_scores.view(-1, 1) shape：3*1，logit_score shape：3*21128
        # 3表示基于当前每个单词预测得到的下一个单词的概率分布，求和后再取topK


        ## 取topk的时候我们是展平了然后再去调用topk函数
        # 展平
        logit_score = logit_score.view(-1)
        hype_score, hype_pos = torch.topk(logit_score, beam_size)  # 取出概率最大的三个预测单词
        indice1 = hype_pos // scores.shape[-1] # 行索引
        indice2 = hype_pos % scores.shape[-1] # 列索引

        # 下面需要更新一下输出了
        new_hype_scores = []
        new_hype_ids = []
        # 为啥有这个[],就是因为要过滤掉结束的序列。
        next_chars = [] # 用来保存新预测出来的一个字符，继续接到输入序列后面，再去预测新字符
        for i_1, i_2, score in zip(indice1, indice2, hype_score):
            # 遍历三次
            # 行索引，列索引(单词的预测位置)，单词预测得分
            i_1 = i_1.item()
            i_2 = i_2.item()
            score = score.item()
            # print(i_1, i_2)
            hype_id = output_ids[i_1] + [i_2] # 保存所有输出的序列，而不仅仅是新预测的单个字符

            if i_2 == sep_id:
                # 说明解码到最后了
                if score == torch.max(hype_score).item():
                    # 说明找到得分最大的那个序列了 直接返回即可
                    return hype_id[: -1]
                else:
                    # 完成一个解码了，但这个解码得分并不是最高，因此的话需要跳过这个序列
                    beam_size -= 1
            else :
                new_hype_ids.append(hype_id)
                new_hype_scores.append(score)
                next_chars.append(i_2) # 收集一下，需要连接到当前的输入序列之后

        output_ids = new_hype_ids
        
        output_scores = torch.tensor(new_hype_scores, dtype=torch.float32, device=device)
        # 现在需要重新构造输入数据了，用上一次输入连接上这次新输出的字符，再输入bert中预测新字符
        token_ids = token_ids[:len(output_ids)].contiguous() # 截取，因为要过滤掉已经完成预测的序列
        token_type_ids = token_type_ids[: len(output_ids)].contiguous()

        next_chars = torch.tensor(next_chars, dtype=torch.long, device=device).view(-1, 1)
        next_token_type_ids = torch.ones_like(next_chars, device=device)

        # 将已预测字符和context部分拼接起来
        # print(token_ids.shape, next_chars.shape)
        # print(token_type_ids.shape, next_token_type_ids.shape)
        token_ids = torch.cat((token_ids, next_chars), dim=1)
        token_type_ids = torch.cat((token_type_ids, next_token_type_ids), dim=1)

        if beam_size < 1:
            break

    # 如果达到最大长度的话 直接把得分最高的输出序列返回把
    return output_ids[output_scores.argmax().item()] 