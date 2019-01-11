# Question Answer Summarization
# CS388: Natural Language Processing 
# Authors: Nidhi Kadkol, Shivangi Mahto

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random 
import math
import copy
import os
import torch.autograd as autograd
import time
from collections import defaultdict

class EmbeddingLayer(nn.Module):
    # Parameters: dimension of the word embeddings, number of words, and the dropout rate to apply
    # (0.2 is often a reasonable value)
    def __init__(self, input_dim, full_dict_size, embedding_dropout_rate):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    # Takes either a non-batched input [sent len x input_dim] or a batched input
    # [batch size x sent len x input dim]
    def forward(self, input_vec):
        embedded_words = self.word_embedding(input_vec)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings
    
def siamese(hid1, hid2):
  return torch.exp(-(torch.sum(torch.abs(torch.add(hid1, -hid2)))))
    
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def form_input(x):
    return torch.from_numpy(x).float()

def evaluate(device, model_enc, train_exs):
    with torch.no_grad():
        model_enc.eval()
        truth_res = []
        pred_res = []
        train_exs0 = train_exs[0]
        train_exs1 = train_exs[1]
        for ex in train_exs0:
            train_vector_one = torch.from_numpy(np.array(ex.indexed_q_one)).to(device)
            train_vector_two = torch.from_numpy(np.array(ex.indexed_q_two)).to(device)
            if len(train_vector_one) == 0 or  len(train_vector_two) == 0:
                    continue          
            truth_res.append(ex.label)
            label = ex.label
            enc_vec_one = model_enc(train_vector_one)
            enc_vec_two = model_enc(train_vector_two)
                    
            distance = siamese(enc_vec_one, enc_vec_two)
            
            pred_res.append(round(distance.item()))

        for ex in train_exs1:
            train_vector_one = torch.from_numpy(np.array(ex.indexed_q_one)).to(device)
            train_vector_two = torch.from_numpy(np.array(ex.indexed_q_two)).to(device)
            if len(train_vector_one) == 0 or  len(train_vector_two) == 0:
                    continue          
            truth_res.append(ex.label)
            label = ex.label
            enc_vec_one = model_enc(train_vector_one)
            enc_vec_two = model_enc(train_vector_two)
                    
            distance = siamese(enc_vec_one, enc_vec_two)
            pred_res.append(round(distance.item()))
        acc = get_accuracy(truth_res, pred_res)
        return acc

class BiLSTMEncoderQuora(nn.Module):

    def __init__(self, device, embedding_dim, hidden_dim, vocab_size):
        super(BiLSTMEncoderQuora, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True).to(device)
        self.device = device
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden = self.init_hidden()
        self.init_weight()
    
    def init_weight(self):
        # ih: input - hidden, hh: hidden - hidden, l0: layer 0, l1: layer 1
        # weight_... : weights of input_size x hidden_size or hidden_size x hidden_size (depending on whether ih or hh is specified) are initialized
        # bias_... : bias of size hidden_size initialized
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=1)
        nn.init.constant_(self.lstm.bias_hh_l0, 0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).to(self.device),   
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).to(self.device))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence).to(self.device)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        y = self.linear(lstm_out[-1])
        return y #self.softmax(y)

class QuestionClassifier(nn.Module):
  
  def __init__(self, encoding_dim, hidden_dim_a, hidden_dim_b, hidden_dim_c, output_size):
    super(QuestionClassifier, self).__init__()
    self.linear_a = nn.Linear(encoding_dim , hidden_dim_a)
    self.linear_b = nn.Linear(hidden_dim_a, hidden_dim_b)
    self.linear_c = nn.Linear(hidden_dim_b, hidden_dim_c)
    self.to_class_size = nn.Linear(hidden_dim_c, output_size)
    #self.softmax = nn.LogSoftmax(dim=-1)
    
    nn.init.xavier_uniform_(self.linear_a.weight)
    nn.init.xavier_uniform_(self.linear_b.weight)
    nn.init.xavier_uniform_(self.linear_c.weight)
    nn.init.xavier_uniform_(self.to_class_size.weight)

  def forward(self, enc_vec):
    out_a = F.relu(self.linear_a(enc_vec))
    out_b = F.relu(self.linear_b(out_a))
    out_c = F.relu(self.linear_c(out_b))
    return self.to_class_size(out_c)

class BiLSTMEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN
    def __init__(self, input_size, hidden_size, dropout):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout, bidirectional=True)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        # ih: input - hidden
        # hh: hidden - hidden 
        # l0: layer 0
        # l1: layer 1
        # weight_... : weights of input_size x hidden_size or hidden_size x hidden_size (depending on whether ih or hh is specified) are initialized
        # bias_... : bias of size hidden_size initialized
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
        nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)
            
    # embedded_words should be a [batch size x sent len x input dim] tensor
    # input_lens is a tensor containing the length of each input sentence
    # Returns output (each word's representation), context_mask (a mask of 0s and 1s
    # reflecting where the model's output should be considered), and h_t, a *tuple* containing
    # the final states h and c from the encoder for each sentence.
    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        # input_lens is a pytorch Variable which is a wrapping over a tensor that makes it differentiable. 
        # To unpack the underlying tensor, we use input_lens.data.
        max_length = input_lens.data[0].item()
        #context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.

        # output - (sentence_length,batch_size,hid_size*num_directions)
        # hn[0] - (num_layers*num_directions, batch_size, hid_size)
        # hn[1] - (num_layers*num_directions, batch_size, hid_size)

        h, c = hn[0], hn[1]
        # Grab the representations from forward and backward LSTMs
        h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
        # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
        # as the hidden size in the encoder
        new_h = self.reduce_h_W(h_)
        new_c = self.reduce_c_W(c_)
        h_t = (new_h, new_c)
        return (output, h_t)
      
      
def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def train_model(train_exs, word_vectors, device):
    torch.backends.cudnn.benchmark = True    

    train_exs0 = train_exs[0]
    train_exs1 = train_exs[1]
    print(len(train_exs0), len(train_exs1))

    train_exs0 = train_exs0[:len(train_exs1)]
    print(len(train_exs0), len(train_exs1))

    dev_exs = [train_exs0[-15000:], train_exs1[-15000:]]
    print(len(dev_exs)) # Should be 2 - as it is a list of 2 lists

    EMBEDDING_DIM = word_vectors.vectors.shape[1]
    HIDDEN_DIM = 100
    OUT_DIM = 1
    EPOCH = 5 
    
    model_enc = BiLSTMEncoderQuora(device, embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_vectors.vectors))
    model_enc.to(device)
    model_enc.word_embeddings.weight.data = form_input(word_vectors.vectors).to(device)
    optimizer = optim.Adam(model_enc.parameters(),lr = 1e-3)
   
    best_dev_acc = 0.0001
    no_up = 0
    for epoch in range(EPOCH):
        model_enc.train()
        truth_res = []
        pred_res = []
        avg_loss = 0.0
        count = 0
        perm0 = np.arange(len(train_exs1))[:-15000]
        random.shuffle(perm0)
        start_time = time.time()
        print("EPOCH ", epoch)
        for i in perm0:
            model_enc.train()
            ex0 = train_exs0[i]
            ex1 = train_exs1[i]
            loss = nn.BCELoss()
            loss.to(device)
            for ex in [ex0, ex1]:
                train_vector_one = torch.from_numpy(np.array(ex.indexed_q_one)).long().to(device)
                train_vector_two = torch.from_numpy(np.array(ex.indexed_q_two)).long().to(device)
                if len(train_vector_one) == 0 or  len(train_vector_two) == 0:
                        continue          
                truth_res.append(ex.label)
                label = ex.label
                
                model_enc.hidden = model_enc.init_hidden()
                model_enc.zero_grad()
                
                enc_vec_one = model_enc(train_vector_one)
                enc_vec_two = model_enc(train_vector_two)
                
                distance = siamese(enc_vec_one, enc_vec_two)
                                
                pred_loss = loss(distance, torch.tensor(label).float().to(device))
                
                avg_loss += pred_loss.item()
                pred_loss.backward()
                optimizer.step()
                count = count + 1

            if (count%10000==0):
                end_time = time.time()
                print("Time taken for 10k iterations: %0.1f"% (end_time - start_time))
                start_time = end_time
    
            if (count%50000==0):
                dev_acc = evaluate(device, model_enc, dev_exs)
                print("Loss: %0.3f, Dev accuracy: %0.3f"% (avg_loss/float(count), dev_acc))
    
        dev_acc = evaluate(device, model_enc, dev_exs)
        print("Loss: %0.3f, Dev accuracy: %0.3f"% (avg_loss/float(count), dev_acc))
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            #os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!', best_dev_acc)
            torch.save(model_enc.state_dict(), 'm2r_best_model_acc_' + str(int(dev_acc)) + '.bilstm.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()

    return model_enc

def train_question_classifier(train_exs, dev_exs, num_classes, device, vocab_size, dict_embed_dim):
    EMBEDDING_DIM = dict_embed_dim
    HIDDEN_DIM = 100
    hid_a = 400
    hid_b = 400 
    hid_c = 400
    out = num_classes
    
    print("len(train_exs):", len(train_exs), "len(dev_exs):", len(dev_exs), "num_classes:", num_classes) 

    model_enc = BiLSTMEncoderQuora(device, embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,vocab_size=vocab_size).to(device)
    #model_enc_pre_train = './mr_best_model_acc_0.bilstm.model'
    #model_enc.load_state_dict(torch.load(model_enc_pre_train, map_location={'cuda:0': 'cpu'}))
    #model_enc.load_state_dict(torch.load(model_enc_pre_train, device))
    
    ques_classifier = QuestionClassifier(encoding_dim=HIDDEN_DIM, hidden_dim_a=hid_a, hidden_dim_b=hid_b, hidden_dim_c = hid_c, output_size=out).to(device)

    best_dev_acc = 0.0001

    optimizer = torch.optim.Adam(list(ques_classifier.parameters()) + list(model_enc.parameters()), lr=1e-3)

    for epoch in range(200):
        print("epoch", epoch)
        epoch_loss = 0.0
        
        indices = np.arange(len(train_exs))
        random.shuffle(indices)
        
        truth_res = []
        pred_res = []
        i = 0
        for idx in indices:
            i += 1
            criterion = nn.CrossEntropyLoss()
            model_enc.train()
            ques_classifier.train()
            model_enc.zero_grad()
            ques_classifier.zero_grad()
            optimizer.zero_grad()
            model_enc.hidden = model_enc.init_hidden()

            enc_ques  = model_enc(torch.from_numpy(np.array(train_exs[idx].ques)).to(device))
            label = train_exs[idx].label

            pred = ques_classifier(enc_ques)
            pred_idx = torch.max(pred, dim = 1)[1]

            pred_res.append(pred_idx.item())
            truth_res.append(label)
            ex_loss = criterion(pred, torch.LongTensor([label]).to(device))
            #print(i, ":", train_exs[idx].ques, label)
            #print(i, "label:", label, "predicted", pred_idx.item())
            #print(ex_loss.item())

            ex_loss.backward()
            optimizer.step()
            epoch_loss += ex_loss.item()

        acc = get_accuracy(truth_res, pred_res)
        print("epoch_loss", epoch_loss/len(indices))
        print("Train Accuracy:", acc) 
        
        # Evaluate on Dev Set
        dev_truth_res = []
        dev_pred_res = []
        j = 0
        for ex in dev_exs:
            j += 1
            with torch.no_grad():
                model_enc.eval()
                ques_classifier.eval()
                model_enc.hidden = model_enc.init_hidden()
                dev_enc_ques  = model_enc(torch.from_numpy(np.array(ex.ques)).long().to(device)) 
                dev_pred = ques_classifier(dev_enc_ques)
                dev_pred_idx = torch.max(dev_pred, dim = 1)[1]

                dev_pred_res.append(dev_pred_idx.item())
                dev_truth_res.append(ex.label)
                #print(j, "label:", ex.label, "predicted", dev_pred_idx.item())

        dev_acc = get_accuracy(dev_truth_res, dev_pred_res)
        print("Dev Accuracy:", dev_acc) 
        if (dev_acc > best_dev_acc):
            print("New Best Model found. Saving state.")
            torch.save(model_enc.state_dict(), ' model_enc_after_class_best_acc_' + str(dev_acc) + '.model')
            torch.save(ques_classifier.state_dict(), ' classifier_best_acc_10_' + str(dev_acc) + '.model')
            best_dev_acc = dev_acc
        
PAD_SYMBOL = "<PAD>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"

def get_word_index_mapping_of_sentence(sentence):
    idx_pos_map = defaultdict(list)
    for i in range(len(sentence)):
        idx_pos_map[sentence[i]].append(i)
    return idx_pos_map

def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_final_states_reshaped)


class LSTMAttnDecoder(nn.Module):
 # We need to take input of max_length of input data for encoder sentences
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_sent_len, unk_idx, dropout=0): # OUTPUT DIM IS THE MAX LENGTH OF INPUT 
        super(LSTMAttnDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        #self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.encoder_hidden_output_ll = nn.Linear(hidden_dim*2, hidden_dim, bias = True)
        self.decoder_hidden_state_ll = nn.Linear(hidden_dim, hidden_dim, bias = True)
        self.coverage_ll = nn.Linear(max_sent_len, hidden_dim, bias = True)
        self.get_e_i_t = nn.Linear(hidden_dim, 1)
        self.get_a_t = nn.Softmax(dim=-1)
        self.first_p_vocab_ll = nn.Linear(hidden_dim*3, hidden_dim, bias = True)
        self.second_p_vocab_ll = nn.Linear(hidden_dim, vocab_size)
        self.get_p_vocab = nn.Softmax(dim=-1)
        self.context_ll = nn.Linear(hidden_dim*2, 1, bias=True)
        self.s_t_ll = nn.Linear(hidden_dim, 1, bias=True)
        self.x_t_ll = nn.Linear(embedding_dim, 1, bias=True)
        self.sigmoid_for_p_gen = torch.nn.Sigmoid()
        self.vocab_size = vocab_size
        self.max_sent_len = max_sent_len 
        self.init_weight()
    def init_weight(self):
        # ih: input - hidden, hh: hidden - hidden, l0: layer 0, l1: layer 1
        # weight_... : weights of input_size x hidden_size or hidden_size x hidden_size (depending on whether ih or hh is specified) are initialized
        # bias_... : bias of size hidden_size initialized
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=1)
        nn.init.constant_(self.lstm.bias_hh_l0, 0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0)
    
    def get_p_gen(self, context_vector, s_t, embedded):
        return self.sigmoid_for_p_gen(torch.add(torch.add(self.context_ll(context_vector), self.s_t_ll(s_t)),self.x_t_ll(embedded)))
       
    def forward(self, embedded, idx_pos_map, hidden, encoder_outputs, attention_weights, coverage_vec, inference = False):
        encoder_outputs = encoder_outputs.view(-1, self.hidden_dim * 2)
        coverage_vec = torch.add(coverage_vec, attention_weights) 
        output, hidden = self.lstm(embedded, hidden)
       
        s_t = hidden[0][0]

        e_i = self.get_e_i_t(torch.tanh(torch.add(self.encoder_hidden_output_ll(encoder_outputs), self.decoder_hidden_state_ll(s_t)))).t()
        
        #interim = torch.add(self.encoder_hidden_output_ll(encoder_outputs), self.decoder_hidden_state_ll(s_t))
        #e_i_t = self.get_e_i_t(torch.tanh(torch.add(interim, self.coverage_ll(coverage_vec))))
        a_t = self.get_a_t(e_i)
        context_vector = torch.mm(a_t,encoder_outputs)
        a1 = torch.cat((s_t.view(-1, self.hidden_dim),context_vector), dim=1)
        a2 = self.first_p_vocab_ll(a1)
        a3 = self.second_p_vocab_ll(a2)

        P_vocab = self.get_p_vocab(a3)
        p_gen = self.get_p_gen(context_vector, s_t, embedded)
        P_w = torch.zeros((1,(self.vocab_size  + self.max_sent_len)),requires_grad = True)

        unk_idx = 17614 # calculated and verified
        for word_idx in range(self.vocab_size):
            if word_idx != unk_idx:
                attn_sum = 0
                for i in idx_pos_map[word_idx]:
                    attn_sum += a_t[0][i]
                P_w[0][word_idx] = p_gen * P_vocab[0][word_idx] + (1 - p_gen) * attn_sum     
        
        for word_idx in range(self.vocab_size, self.vocab_size + self.max_sent_len):
            if sentence[word_idx - self.vocab_size] == unk_idx:
                P_w[0][word_idx] =   0
      
        return P_vocab, hidden, a_t, coverage_vec


def train_model_encdec(train_data, test_data, indexer, args, device):
    
    train_data.sort(key=lambda ex: len(ex.doc), reverse=True)
    test_data.sort(key=lambda ex: len(ex.doc), reverse=True)
   
    input_max_len = np.max(np.asarray([len(ex.doc) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, indexer, input_max_len, args.reverse_input)

    output_max_len = np.max(np.asarray([len(ex.best_x) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, indexer, output_max_len)

    print(input_max_len)
    print(len(train_data))
    model_enc = BiLSTMEncoder(300, 200, 0) 
    model_enc.to(device)
    unk_idx = indexer.index_of("UNK")
    
    model_dec = LSTMAttnDecoder(300, 200, len(indexer),input_max_len, unk_idx, dropout=0)
    model_dec.to(device)

    model_input_emb = EmbeddingLayer(300, len(indexer), 0.2)
    model_input_emb.to(device)
    model_output_emb = EmbeddingLayer(300, len(indexer), 0.2)
    model_output_emb.to(device)
    
    sos_ind = indexer.index_of(SOS_SYMBOL)
    eos_idx = indexer.index_of(EOS_SYMBOL)
  
    criterion = torch.nn.NLLLoss()
    params = list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    for epoch in range(10):
        model_enc.train()
        model_dec.train()
        model_input_emb.train()
        model_output_emb.train()
        
        perm0 =np.arange(len(train_data))
        random.shuffle(perm0)
        for idx in perm0:
            
            model_enc.zero_grad()
            model_dec.zero_grad()
            model_input_emb.zero_grad()
            model_output_emb.zero_grad()
            sent_loss = torch.autograd.Variable(torch.FloatTensor([0]))
             
            # Encoding input
            sentence = all_train_input_data[idx]
            print("input sequence", sentence)
            idx_pos_map = get_word_index_mapping_of_sentence(sentence)
            x_tensor = torch.from_numpy(sentence).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(train_data[idx].doc))).unsqueeze(0)
            enc_word, context = encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
            first_idx_tensor = torch.tensor(sos_ind).unsqueeze(0)
            sos_embed = model_output_emb.forward(first_idx_tensor).unsqueeze(0)
            hidden = context
            
            attention_weights =  torch.autograd.Variable(torch.FloatTensor([0]*len(train_data[idx].doc)))
            coverage_vec = torch.autograd.Variable(torch.FloatTensor([0]*len(train_data[idx].doc)))
            pred, hidden, a_t, coverage = model_dec.forward(sos_embed, idx_pos_map, hidden, enc_word, attention_weights, coverage_vec)
            
            for tgt_idx in train_data[idx].best_x:
                sent_loss += criterion(pred, torch.LongTensor([tgt_idx]))
                if tgt_idx == eos_idx:
                    break
                word_emb = model_output_emb.forward(torch.tensor(tgt_idx).unsqueeze(0)).unsqueeze(0)
                pred, hidden, a_t, coverage = model_dec.forward(word_emb, idx_pos_map, hidden,enc_word,a_t, coverage)
                print("pred size",pred.size())
                print(torch.max(pred.squeeze(0), dim = 1)[1])

            print(sent_loss)
            sent_loss.backward()
            optimizer.step()
            
    return Seq2SeqSemanticParser(model_enc,model_dec,model_input_emb,model_output_emb,indexer,args.reverse_input, output_max_len)

def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.doc[len(ex.doc) - 1 - i] if i < len(ex.doc) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.doc[i] if i < len(ex.doc) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.best_x[i] if i < len(ex.best_x) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

def len_input(exs):
    return np.array([[len(ex.doc)] for ex in exs])


class Seq2SeqSemanticParser(object):
    
    def __init__(self, model_enc,model_dec,model_input_emb,model_output_emb,indexer, reverse_input, output_max_len):
        self.input_indexer = indexer
        self.output_indexer = indexer
        self.model_enc = model_enc
        self.model_dec = model_dec
        self.model_input_emb = model_input_emb
        self.model_output_emb = model_output_emb
        self.reverse_input = reverse_input
        self.output_max_len = output_max_len 

    def decode(self, test_data):
        self.model_enc.eval()
        self.model_dec.eval()
        self.model_input_emb.eval()
        self.model_output_emb.eval()
        
        print(SOS_SYMBOL)
        print(EOS_SYMBOL)
       
        sos_idx = self.output_indexer.index_of(SOS_SYMBOL)
        eos_idx = self.output_indexer.index_of(EOS_SYMBOL)
        test_derivs =[]
        input_max_len = np.max(np.asarray([len(ex.doc) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, self.reverse_input)

        for idx in range(len(test_data)):
            ex = test_data[idx]
            
            sentence = all_test_input_data[idx]
            idx_pos_map = get_word_index_mapping_of_sentence(sentence)
            x_tensor = torch.from_numpy(sentence).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(test_data[idx].doc))).unsqueeze(0)
            enc_word, context = encode_input_for_decoder(x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            first_idx_tensor = torch.tensor(sos_idx).unsqueeze(0)
            sos_embed = model_output_emb.forward(first_idx_tensor).unsqueeze(0)
            hidden = context
            
            attention_weights =  torch.autograd.Variable(torch.FloatTensor([0]*len(test_data[idx].doc)))
            coverage_vec = torch.autograd.Variable(torch.FloatTensor([0]*len(test_data[idx].doc)))
            pred, hidden, a_t, coverage = self.model_dec.forward(sos_embed, idx_pos_map, hidden, enc_word, attention_weights, coverage_vec)
            
            # First output for start of sentence
            out_idx = torch.max(pred.squeeze(0), dim = 1)[1]
            length = 0
            output_seq = []
            while out_idx!=eos_idx and length < 3*self.output_max_len:
                output_seq.append(self.output_indexer.get_object(out_idx.item()))
                word_emb = self.model_output_emb.forward(torch.tensor(out_idx.item()).unsqueeze(0)).unsqueeze(0)
                pred, hidden, a_t, coverage = self.model_dec.forward(word_emb, idx_pos_map, hidden,enc_word,a_t, coverage)
                out_idx = torch.max(pred.squeeze(0), dim = 1)[1]
                length += 1
            print(output_seq)
            #test_derivs.append([Derivation(ex, 1.0, output_seq)])
        return test_derivs
    

