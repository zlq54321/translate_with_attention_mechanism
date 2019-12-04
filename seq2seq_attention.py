from data import MakeSrcTrgDataset
import tensorflow as tf

SRC_TRAIN_DATA = './data/train.en'
TRG_TRAIN_DATA = './data/train.zh'
CHECK_POINT_PATH = './save_attention/seq2seq.ckpt'
HIDDEN_SIZE = 1024
DECODER_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
BATCH_SIZE = 100
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True


# 定义NMTModel类
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

    # 前向传播
    # 输入即为MakeSrcTrgDataset的输出张量
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]
        # 转为词向量
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
        # 加dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # 编码器
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

            dec_output, _ = tf.nn.dynamic_rnn(attention_cell, trg_emb, trg_size, dtype=tf.float32)

        # log perplexity
        # output_dim = (b_size*batch_max_len, h_size)
        output = tf.reshape(dec_output, [-1, HIDDEN_SIZE])
        # bias广播到合适维度， 得到logits_dim = (b_size*batch_max_len, vocab_size)
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias  #
        # trg_label是单词编号,所以维度是(b_size, batch_max_len,) reshape (b_size*batch_max_len,)
        # loss_dim = (b_size*batch_max_len,)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]),
                                                              logits=logits)

        # 填充位置权重为0， 方便计算字符平均损失
        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        # label_weights_dim = (b_size*batch_max_len,), 这样和loss可以点乘，去掉padding出的无效loss
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 反向传播， 这里优化的是样本平均损失
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        return cost_per_token, train_op


# 训练一个世代
def run_epoch(session, cost_op, train_op, saver, step):
    while True:
        try:
            cost, _ = session.run([cost_op, train_op])
            step += 1
            if step % 10 == 0:
                print('After %d steps, per token cost is %.3f' % (step, cost))
            if step % 100 == 0:
                saver.save(session, CHECK_POINT_PATH, global_step=step)
        except tf.errors.OutOfRangeError:
            break
    return step


def main(argv=None):
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope('nmt_model', reuse=None, initializer=initializer):
        train_model = NMTModel()

    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

    saver = tf.train.Saver()
    step = 0
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        tf.global_variables_initializer().run()

        # 从存储的参数继续训练
        ckpt = tf.train.get_checkpoint_state('./save_attention/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            step = int(step)

        for i in range(NUM_EPOCH):
            print('In iterator: %d' % (i+1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == '__main__':
    main()




