import tensorflow as tf
import numpy as np
import helper

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()

def get_tensors(loaded_graph):
    inputs = loaded_graph.get_tensor_by_name("inputs:0")
    initial_state = loaded_graph.get_tensor_by_name("init_state:0")
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    probs = loaded_graph.get_tensor_by_name("probs:0")
    return inputs, initial_state, final_state, probs
	
def pick_word(probabilities, int_to_vocab):
    chances = []
    for idx, prob in enumerate(probabilities):
        if prob >= 0.05:
            chances.append(int_to_vocab[idx])
    rand = np.random.randint(0, len(chances))
    return str(chances[rand])