# Question Answer Summarization
# CS388: Natural Language Processing 
# Authors: Nidhi Kadkol, Shivangi Mahto

import argparse
import sys
from model import *
from data import *

def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='FFNN', help='model to run (FF or FANCY)')
    parser.add_argument('--word_vecs_path', type=str, default='glove.6B.300d-relativized.txt', help='path to word vectors file')
    parser.add_argument('--train_path', type=str, default='quora_duplicate_questions.tsv', help='path to train set (you should not need to modify)')
    parser.add_argument('--para_ques_url', type=str, default='https://trec.nist.gov/data/qa/2017_LiveQA/questionParaphrases.xml')
    parser.add_argument('--answer_file', type=str, default='./Answer_file.txt')
    parser.add_argument('--no_reverse_input', dest='reverse_input', default=True, action='store_false', help='disable_input_reversal')
    parser.add_argument('--emb_dropout', type=float, default=0, help='input dropout rate')
    parser.add_argument('--input_dim', type=int, default=300, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=300, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')
    parser.add_argument('--rnn_dropout', type=float, default=0, help='dropout rate internal to encoder RNN')
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional LSTM')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    word_vectors = read_word_embeddings(args.word_vecs_path)
    quora_exs = read_and_index_sentiment_examples(args.train_path, word_vectors.word_indexer)
    model_enc = train_model(quora_exs, word_vectors, device)
    ID_dict, train_ques_list, dev_ques_list, num_classes =  read_and_index_question_examples(args.para_ques_url, word_vectors.word_indexer)
    classifier = train_question_classifier(train_ques_list, dev_ques_list, num_classes, device, len(word_vectors.word_indexer), word_vectors.vectors.shape[1])
    #decoder = train_model_encdec(answer_exs, answer_exs, word_vectors.word_indexer, args, device)
