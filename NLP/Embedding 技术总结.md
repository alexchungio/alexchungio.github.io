# Embedding  技术总结



 ## 文本向量化

**文本向量化（ vectorize）**是指将文本转换为数值张量的过程。

* 标记

  讲文本分解而成的单元(单词、字符或n-gram)叫**标记（token）**

* 分词

  讲文本分解成标记的的过程叫做**分词（tokenization）**。

**文本向量化就是应用某种分词方案讲数值向量与生成的标记相关联。** 主要有**one-hot** 和**词嵌入（word embedding）**两种方法。

## Word2Vec

### Skip-Gram

### CBOW

CBOW(Continuous Bag-of-Words)