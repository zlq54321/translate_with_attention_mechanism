import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants as convert_to_constant
import codecs
import sys

CHECKPOINT_PATH = './save_attention/seq2seq.ckpt-5900'
HIDDEN_SIZE = 1024
DECODER_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
SHARE_EMB_AND_SOFTMAX = True
# 词汇表文件
SRC_VOCAB = "./data/ted.en.vocab"
TRG_VOCAB = "./data/ted.zh.vocab"
SOS_ID = 0
EOS_ID = 1


class NMTModel(object):
    def __init__(self):
        # lstm cell
        basic_cell = tf.nn.rnn_cell.BasicLSTMCell

        self.enc_cell_fw = basic_cell(HIDDEN_SIZE)
        self.enc_cell_bw = basic_cell(HIDDEN_SIZE)
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([basic_cell(HIDDEN_SIZE) for _ in range(DECODER_LAYERS)])

        # word embedding
        self.src_embedding = tf.get_variable('src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # softmax
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable('softmax_bias', [TRG_VOCAB_SIZE])

    def inference(self, src_input):

        # 将句子整理为batch为1的数据
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope('encoder'):
            # sequence_length=src_size 的解释见书P248
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw,
                                                                     self.enc_cell_bw,
                                                                     src_emb,
                                                                     src_size,
                                                                     dtype=tf.float32)
            # 上面的outputs是前向和后向rnn的输出的元组(output_fw , output_bw),
            # 每一个维度是(batch, time, hidden), 下面即是将第三维度拼接起来
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], axis=-1)
        # 解码器
        with tf.variable_scope('decoder'):
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_SIZE,
                                                                       enc_outputs,
                                                                       memory_sequence_length=src_size)
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell,
                                                                 attention_mechanism,
                                                                 attention_layer_size=HIDDEN_SIZE)

        # 设置解码的最长输出，避免翻译出无限长的中文
        MAX_DEC_LEN = 100

        with tf.variable_scope('decoder/rnn/attention_wrapper'):
            # 使用一个变长的tensorarray来存储生成的句子
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            # 填写第一个单词<sos>作为解码器输入
            init_array = init_array.write(0, SOS_ID)
            # 构建初始的循环状态，
            # 循环状态包含，attention_decoder初始零状态（不再是编码层输出的状态了），保存生成中文的tensorarray，以及记录解码步数的一个整数step
            init_loop_var = (attention_cell.zero_state(batch_size=1, dtype=tf.float32), init_array, 0)

            # tf.while_loop的循环继续条件
            # 解码到结束符或者超出最大长度
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(tf.not_equal(trg_ids.read(step), EOS_ID),
                                                    tf.less(step, MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                # 读取器最后一步的输出并读取其词向量
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
                # 使用attention_cell计算
                dec_outputs, next_state = attention_cell.call(state=state, inputs=trg_emb)
                # 计算每个可能的输出单词的对应的logits，并取值最大的作为这一步的输出
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                trg_ids = trg_ids.write(step+1, next_id[0])
                return next_state, trg_ids, step+1

            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var
            )
            return trg_ids.stack()


def main(argv=None):
    with tf.variable_scope('nmt_model', reuse=None):
        model = NMTModel()

    test_en_text = 'I don not like this test . <eos>'
    print(test_en_text)

    # 根据英文词汇表，将测试句子转为单词ID。
    with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))

    test_en_ids = []
    for token in test_en_text.split():
        if token in src_id_dict:
            test_en_ids.append(src_id_dict[token])
        else:
            test_en_ids.append(src_id_dict['<unk>'])
    print(test_en_ids)

    output_op = model.inference(test_en_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    # 从存储的参数继续训练
    ckpt = tf.train.get_checkpoint_state('./save_attention/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # 存储为pb文件，以后可更快的执行
    graph_def = tf.get_default_graph().as_graph_def()
    out_graph_def = \
        convert_to_constant(sess,
                            graph_def,
                            ['decoder/rnn/attention_wrapper/TensorArrayStack/TensorArrayGatherV3'])
    with tf.gfile.GFile('./model.pb', 'wb') as f:
        f.write(out_graph_def.SerializeToString())

    output_ids = sess.run(output_op)
    print(output_ids)

    # 根据中文词汇表，将翻译结果转换为中文文字。
    with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]

    output_text = ''.join([trg_vocab[x] for x in output_ids])

    # 输出翻译结果。
    print(output_text.encode('utf8').decode(sys.stdout.encoding))
    sess.close()


if __name__ == "__main__":
    main()




