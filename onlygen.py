import tensorflow as tf
import getsave
import numpy as np

# 训练步长
seq_length = 30

# 使用训练好的模型来生成自己的小说
# 生成文本的长度
gen_length = 1000

# 文章开头的字，指定一个即可，这个字必须是在训练词汇列表中的
prime_word = '正'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载保存过的session
    loader = tf.train.import_meta_graph(getsave.load_dir + '.meta')
    loader.restore(sess, getsave.load_dir)

    # 通过名称获取缓存的tensor
    input_text, initial_state, final_state, probs = getsave.get_tensors(loaded_graph)

    # 准备开始生成文本
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # 开始生成文本
    for n in range(gen_length):
        dyn_input = [[getsave.vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = getsave.pick_word(probabilities[dyn_seq_length - 1], getsave.int_to_vocab)

        gen_sentences.append(pred_word)

    # 将标点符号还原
    novel = ''.join(gen_sentences)
    for key, token in getsave.token_dict.items():
        ending = ' ' if key in ['\n', '（', '“'] else ''
        novel = novel.replace(token.lower(), key)
    novel = novel.replace('\n ', '\n')
    novel = novel.replace('（ ', '（')

    print(novel)

# Input -> LSTM -> Dropout -> LSTM -> Dropout -> Fully Connected