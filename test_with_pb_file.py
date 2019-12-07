from seq2seq_test import NMTModel
import tensorflow as tf
import codecs
from tensorflow.python.framework.graph_util import convert_variables_to_constants as convert_to_constant
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
IMPORT_RETURN_NAME = 'decoder/rnn/attention_wrapper/TensorArrayStack/TensorArrayGatherV3:0'


def main(argv=None):

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

    with tf.Session() as sess:
        with tf.gfile.FastGFile('./model.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        output_op = tf.import_graph_def(graph_def, return_elements=[IMPORT_RETURN_NAME])

        output_ids = sess.run(output_op)
        output_ids = output_ids[0].tolist()
        print(output_ids)

        # 根据中文词汇表，将翻译结果转换为中文文字。
        with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
            trg_vocab = [w.strip() for w in f_vocab.readlines()]

        output_text = ''.join([trg_vocab[x] for x in output_ids])

        # 输出翻译结果。
        print(output_text.encode('utf8').decode(sys.stdout.encoding))


if __name__ == "__main__":
    main()
