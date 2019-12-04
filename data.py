import tensorflow as tf

MAX_LEN = 50  # 限定句子的最大单词数量
SOS_ID = 0   # 目标语言词汇表中<sos>的ID


# 使用Dataset从一个文件读取一个语言的数据
# 数据格式为每行一句话， 单词已经转化为单词编号
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 读入的是文本，字符格式的，要转成数字格式
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


def FilterLength(src_tuple, trg_tuple):
    ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
    src_len_ok = tf.logical_and(tf.greater(src_len, 1),
                                tf.less_equal(src_len, MAX_LEN))
    trg_len_ok = tf.logical_and(tf.greater(trg_len, 1),
                                tf.less_equal(trg_len, MAX_LEN))
    return tf.logical_and(src_len_ok, trg_len_ok)


def MakeTrgInput(src_tuple, trg_tuple):
    ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
    trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
    return (src_input, src_len), (trg_input, trg_label, trg_len)


# 读入源语言和目标语言文件，也就是英文和中文文件，并padding和batching
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    # 删除内容空的和过长的
    dataset = dataset.filter(FilterLength)

    # 生成<sos>, x, y, z形式的句子并加入到dataset中
    dataset = dataset.map(MakeTrgInput)

    dataset = dataset.shuffle(10000)

    # 规定填充后的数据维度
    padding_shapes = (
        (tf.TensorShape([None]),
         tf.TensorShape([])),
        (tf.TensorShape([None]),
         tf.TensorShape([None]),
         tf.TensorShape([]))
    )
    batched_dataset = dataset.padded_batch(batch_size, padding_shapes)

    return batched_dataset

