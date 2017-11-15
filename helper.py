import pickle
import os

def load_text(path):
    input_file = os.path.join(path)

    with open(input_file, 'r') as f:
        text_data = f.read()

    return text_data


def preprocess_and_save_data(text, token_lookup, create_lookup_tables):
    token_dict = token_lookup()
    
    for key, token in token_dict.items():
        text = text.replace(key, '{}'.format(token))

    text = list(text)
    
    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    return pickle.load(open('params.p', mode='rb'))

# 工作
def create_lookup_tables(input_data):
    vocab = set(input_data)
    # 文字到数字的映射
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}
    # 数字到文字的映射
    int_to_vocab = dict(enumerate(vocab))
    return vocab_to_int, int_to_vocab
	
def token_lookup():
    # 创建一个符号查询表，把逗号，句号等符号与一个标志一一对应，用于将『我。』和『我』这样的类似情况区分开来，排除标点符号的影响。
    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])
    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R"]
    return dict(zip(symbols, tokens))

def pretreatment():
    # 读取文件
    text = load_text('D:/TCproject/NLP/Chinese-novel-generation-master/123.txt')
    lines_of_text = text.split('\n')
    # 用以上方法预处理数据并保存到磁盘
    preprocess_and_save_data(''.join(lines_of_text), token_lookup, create_lookup_tables)

# pretreatment()
# 读取数据
int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
