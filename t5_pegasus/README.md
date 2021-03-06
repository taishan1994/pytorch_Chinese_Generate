# T5_Pegasus
- 1、将数据保存在data文件夹下。
- 2、在data_loader.py中定义数据的读取方式，具体可参考其它数据的格式。
- 3、在main.py里面新增数据加载的方式，最后使用相关的指令运行即可。

预训练的模型下载地址：

- Small版本：
	- 百度网盘：https://pan.baidu.com/s/1Kc6xFqJZoVxKLBGx924zgQ 提取码：hvq9
	- Google Drive：https://drive.google.com/file/d/1fCL7f_f8I6YuezoYQ3EvT9h-S1WngKZo/view?usp=sharing

训练好的模型和数据下载：<br>

链接：https://pan.baidu.com/s/1etGXjRzyYIUMdwf_lxRXuQ?pwd=47oa <br>
提取码：47oa<br>

# 依赖

```python
torch==1.8.0
transformers==4.15.0
tensorboardX
rouge
```

# 运行

## cnews数据集

### 训练

```python
python main.py \
--bert_dir="model_hub/chinese_t5_pegasus_small/" \
--batch_size=16 \
--train_epochs=40 \
--max_seq_len=512 \
--gen_max_len=64 \
--data_name="cnews" \
--model_name="t5_pegasus" \
--do_train="True" \
--do_test="False" \
--do_generate="False" \
--use_tensorboard="True" 
```

### 测试和预测

```python
python main.py \
--bert_dir="model_hub/chinese_t5_pegasus_small/" \
--batch_size=16 \
--train_epochs=40 \
--max_seq_len=512 \
--gen_max_len=64 \
--data_name="cnews" \
--model_name="t5_pegasus" \
--do_train="False" \
--do_test="True" \
--do_generate="True" \
--seed=123  # 默认seed为110，这里通过指定seed来测试不同的数据（部分数据）
```

### 结果

这里在运行7个epoch后手动停止了。

```python
{'rouge-1': 0.3630059543818625, 'rouge-2': 0.23552465150931073, 'rouge-l': 0.3484842659597319, 'bleu': 0.12693138432438517}

文本： 京城四少看谁风流又倜傥导语：何谓“京城四少”：内地有不少喜欢和女明星传绯闻的富二代，其中几个年轻帅气的被戏称为“京城四少”即汪小菲、汪雨、王烁、王珂。“京城四少”意味着倜傥、风流，万事有人打点帮办。论貌，气宇轩昂，可王珂还不如新娘子挺拔高挑。论才，知诗书会风雅，谈吐不凡，可汪小菲连微博(http://t.sina.com.cn)文字都写不清。这样的“京城四少”满足的是民间八卦游戏。如同顶尖奢侈品不会在电视里向最广大群众投播广告，最漂亮的女人不会抛头露面演戏给大众看，真正有身份的少爷自有封闭的小圈子，绝不屑与外人道。当初还跟张雨绮在一起的时候是出双入对，笔直的西装，陪着心爱的女人出入到各个场所，力挺的白衬衫+灰色西装，十分的绅士。性情中人的汪小菲，生活中也很自恋的说，摆酷酷的pose，穿着笔挺的西服，或是漆黑锃亮的夹克，又或是将衬衫的扣子开到第三颗，都显露出他男性的魅力所在。深V领的黑色针织衫，简洁大方，没有丝毫的点缀与装饰，深沉的色调彰显出汪小菲的熟男魅力，一点点的胡渣也够吸引！
真实： 组图：汪小菲耍帅NO.1
预测： 汪小菲汪雨王烁王珂
====================================================================================================
文本： 指望巴龙对抗湖人双塔？新浪体育讯北京时间4月13日，开拓者总经理里奇-乔今天宣布，球队已经正式签下了中锋厄尔-巴龙，这位7尺的的中锋将代表开拓者征战本赛季剩余的比赛。乔在今天的发布会上表示，希望巴龙的到来能帮助开拓者增加内线的板凳深度。以目前的西部排名来看，暂列第6位的开拓者季后赛首轮就将碰上卫冕冠军湖人，这样一来，面对对方内线双塔保罗-加索尔和安德鲁-拜纳姆的联手冲击，开拓者的内线将遭遇不小的挑战。开拓者是联盟中出了名的“中锋坟场”，格雷格-奥登、马库斯-坎比以及昔日的白人内线乔尔-普尔兹比拉，都曾中了魔咒似的的相继受伤。而本赛季，拉马库斯-阿尔德里奇甚至被主教练麦克米兰不止一次临时提到中锋位置客串。目前球队的主力中锋坎比已经缺席了4月6日之后的两场比赛，而季后赛之前看样子坎比也不会复出，所以此时开拓者火线签下巴龙救急，也同时是在为季后赛找到了一个备胎。本赛季巴龙曾效力过太阳和雄鹿两支球队，一共打了19场比赛，场均有3.8分和3.3个篮板的表现。(小林)
真实： 开拓者火速签下内线备胎
预测： 开拓者正式签下中锋巴龙
====================================================================================================
文本： 体重暴涨55磅谁还敢签新浪体育讯北京时间4月12日消息，据雅虎体育专家阿德里安-沃伊纳罗斯基透露，由于埃迪-库里的身体状况不佳，迈阿密热火已经决定放弃签约这位7尺中锋。沃伊纳罗斯基在个人Twitter中表示，库里将不会在这个赛季加盟热火。据悉，在尼克斯裁掉库里之后，热火的确是果断跟进，希望能签下这名昔日第4顺位的中锋，来在季后赛为自己的球队增加内线板凳深度。但事实上，库里的身体状况十分糟糕，已经有两个多赛季没怎么打比赛的库里，身体发福的令人吃惊。2001年初入联盟时，库里的身高为7英尺，体重295磅。但如今，在身高保持不变的情况下，库里的体重已经暴涨至350多磅，完全超出了一名职业运动员可以承受的范畴。以这种身体状况在对抗激烈的NBA联盟打球，库里想不受伤几乎都是奢望。而在看到如此肥胖的库里，热火管理层显然也是大跌眼镜，他们不会傻到把钱白白扔给那些只能坐在板凳席看球的球员。因此球队还是果断放弃了签约库里的机会。据报道称，虽然库里本赛季想要站上NBA赛场的机会已经微乎其微。但在联盟中，仍然有一些球队对库里寄予希望。包括热火等几支NBA球队，都表示有耐心等待库里完成身体恢复，而这些球队都希望能够邀请到库里来球队的夏季训练营，来面对面查看库里的身体状况。作为2001届的4号新秀，库里曾是被寄予取代奥尼尔之厚望的年轻中锋，但事与愿违的是，入行多年库里的防守和篮板不见任何提高，却总能得到球队开出的千万工资。而不止一家美国媒体在评选NBA烂合同时，把库里的名字列在其中。但库里似乎并不以此为动力，反而越发沉沦。近两个赛季，库里总共只代表尼克斯打了10场比赛，其余时间不是在伤病中度过，就是在板凳席上两眼发呆。如今的库里，又因为体重问题失去了为一支联盟强队效力的机会，如果继续就此沉沦下去，即便有球队愿意邀请他进入夏季训练营，库里又拿什么来打动对方给你开出一份新合同呢？(小林)
真实： 曝热火终放弃签昔日小鲨鱼
预测： 热火欲签下7尺中锋
====================================================================================================
文本： 据台湾《联合晚报》报道，台湾当局经济主管部门30日下午正式公布“大陆地区人民来台投资许可办法”，同时也开放陆资入台设分公司或办事处，即日实施。这次首批开放陆资来台投资项目有100项，但公布显示的项数会再多一些，而中草药、面板、半导体以及营造业，都不在第一阶段开放清单。至于陆资定义，台湾经济主管尹启铭指出，除了陆资证券投资超过10%即视为直接投资，适用陆资入台相关规定外，来台外商企业只要陆资股权达30%，就视为陆资。这次首批开放陆资来台投资项目，包括制造业、服务业与公共建设(限BOT)，总项目100项，但由于两岸在商谈服务业开放时，采取WTO初始承诺表的分类，而经济主管部门正式公布的总项数会更多一些，只是实质内容则是相同。开放清单中，陆企有兴趣的中草药以及目前台当局列为台商赴大陆投资负面表列的面板、半导体等均未列入，有关官员表示将在下批开放时才一并检讨与研究。另外，除高科技，大陆方面原本高度期待的公共建设，在这批开放公告中“限缩”到只限公共建设有开放BOT的范围。例如航空城周遭设施有开放，所以桃园航城的展览馆如果开放BOT，陆资可以参与招商竞标作业，但必须先在台湾成立相关的公司；又例如高铁由于并没有列在开放项目，所以陆资不能参加高铁的BOT。
真实： 台湾开放首批100项大陆企业赴台投资项目中新网6月30日电
预测： 台湾开放陆资入台设分公司或办事处中新网10月30日电
====================================================================================================
文本： Big3真害怕捅这跟软肋新浪体育讯上一次回到克里夫兰打球，勒布朗-詹姆斯是毫不留情地在旧东家头上砍了38分，率队大比分获胜。很久之后，詹姆斯谈起那场比赛都还认为那是一场让热火开始腾飞的胜利。如今，在常规赛快要结束的当口，热火再度造访克里夫兰，若还能赢球，詹姆斯会不会认为这将是让热火在季后赛腾飞的胜利？只可惜的是，恐怕谁都没能想到，詹姆斯和他的热火居然在比赛中一度落后23分。最终，在自己最熟悉的快贷球馆，在曾经三次击败的对手面前，詹姆斯的热火以90-102败北。客观地说，今天这场失利，并不是詹姆斯的责任。全场，他拿到27分、12助攻和10篮板的三双数据，而且自从第一节得到13分之后，他再度得分就已经是第三节后半段了。这之间，他一直在为队友做球。遗憾的是，热火的老毛病今天又犯了。五连败期间，这个毛病就是备受诟病，后来的连胜暂时掩盖住了。今天碰上不要命的骑士，热火替补们的不堪大用又被无情揭开。可能会有球迷质疑，怎么每次输了球就是替补的责任？的确，从今天的赛场表现来看，热火先发中也有球员发挥的很可耻，比如说克里斯-波什，那真是思想有多软，波什就有多软。而且，今天对阵骑士，热火主观上也难免有轻敌思想。也不奇怪，东部前三碰上垫底球队，很难从一开始就打起精神。怎料，进不了季后赛的骑士今天完全是把比赛当季后赛在打。所以从开场热火的节奏就被骑士打乱了。替补的作用就应该在这时候体现，所谓旁观者清，替补就是应该在这种时刻发挥作用。结果，热火的替补们可能是在场边看球太过投入，上场之后同样是完全没有感觉。新科三分王詹姆斯-琼斯在首节剩不到三分钟的时候进场，埃里克-斯波尔斯特拉大概是指望用他的外线进攻来把比分拉近一些。可倒好，琼斯的第一次外线出手就被阿朗佐-吉封盖。第二节，热火还是指望用替补来冲击一下比分，改变一下比赛胶着的尴尬。同样，乔尔-安东尼近距离出手不中，埃迪-豪斯中距离跳投偏出，琼斯则继续在外线放风筝。第二节前半段，热火一直是由替补们在主攻，但始终无法得分，只能依靠德维恩-韦德孤掌难鸣。而骑士方面恰恰相反，这段时间内得分全部来自于替补，一举打出13-4的得分小高潮。从赛后数据来看，骑士5名替补出战，除了乔伊-格拉汉姆是象征性地走过场之外，其他4人出场时间都在20分钟以上，合力贡献了32分12助攻8篮板。而热火的5名替补，共计只得到6分1助攻和9篮板。板凳席上就差了26分，再加上波什今天实在不给力，詹姆斯和韦德两人去填这么大一坑？这真是勉为其难了。不是说输球就该热火替补承担责任，但客观事实已经说明了，有三巨头，特别是詹姆斯和韦德的存在，热火是全联盟空位投篮机会最多、投篮效率最高的球队之一。对豪斯和琼斯等人来说，他们的工作已经被极大地简化了。但今天这比赛，热火替补们无疑是表现不及格的，像新科三分王琼斯，三分线外居然5投0中。表现如此糟糕，输球还能逃得了罪责？(XWT185)
真实： 热火命门竟被鱼腩无限放大
预测： 热火前瞻：詹姆斯vs骑士前瞻：詹姆斯vs骑士
```

# 参考

> 模型参考：https://github.com/SunnyGJing/t5-pegasus-chinese

