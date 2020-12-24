# CCF-2020-BeiKe
CCF 2020的比赛——[房产行业聊天问答匹配](https://www.datafountain.cn/competitions/474)
没有太多时间去尝试一些trick，提交次数也不多，只是简单尝试了一些中文预训练模型，主要是哈工大讯飞联合实验室（HFL）发布的。我用了其中的两个——BERT-wwm-ext和RoBERTa-wwm-ext-large，实验结果如下，权当是Baseline。

&nbsp;|model|折数|epochs|result(%)
:--:|:--:|:--:|:--:|:--:|
1|BERT-wwm-ext|0|3|0.769
2|RoBERTa-wwm-ext-large|0|3|0.770
3|roberta-base|5|5|0.775
4|roberta-large|5|5|0.776

可以后续尝试的trick：
- [对抗训练](https://zhuanlan.zhihu.com/p/91269728)
- 伪标签
- [focal loss](https://zhuanlan.zhihu.com/p/49981234)
- [模型融合](https://blog.csdn.net/weixin_39505820/article/details/111393476)
