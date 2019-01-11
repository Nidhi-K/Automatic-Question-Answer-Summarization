# Question Answer Summarization
# CS388: Natural Language Processing 
# Authors: Nidhi Kadkol, Shivangi Mahto

import csv
import requests
import xml.etree.ElementTree as ET
import re
import numpy as np
import random
from utils import *
from math import factorial

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`\-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\-", " - ", string)
    # We may have introduced double spaces, so collapse these down
    string = re.sub(r"\s{2,}", " ", string)
    return string

class WordEmbeddings:
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding(self, word):
        word_idx = self.word_indexer.get_index(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[word_indexer.get_index("UNK")]

def read_word_embeddings(embeddings_file):
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.get_index(word)
            vectors.append(vector)
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    word_indexer.get_index("UNK")
    vectors.append(np.zeros(vectors[0].shape[0]))
    return WordEmbeddings(word_indexer, np.array(vectors))


class QueryExample:
    def __init__(self, indexed_q_one,  indexed_q_two, label):
        self.indexed_q_one = indexed_q_one
        self.indexed_q_two = indexed_q_two
        self.label = label

    def __repr__(self):
        return repr(self.indexed_q_one) + repr(self.indexed_q_two) + "; label=" + repr(self.label)

class QuestionSet:
    def __init__(self, ID, orig_ques):
        self.ID = ID
        self.Q = orig_ques
        self.para_Q_list = []
        self.len_para_list = 0

    def add_para_ques(self,para_ques):
        self.para_Q_list.append(para_ques)
        self.len_para_list +=1

    def __repr__(self):
        return repr(self.ID) + repr(self.Q) + repr(self.para_Q_list)

class Question:
    def __init__(self, label, ques):
        self.ques = ques
        self.label = label
    
    def __repr__(self):
        return "Question: " + repr(self.ques) + " Label: " + repr(self.label)

class AnswerExample:
    def __init__(self,rating, Answer):
        self.all_answer = [Answer]
        self.all_answer_ratings = [rating]

    def add_answer(self,rating,answer):
        self.all_answer.append(answer)
        self.all_answer_ratings.append(rating)
            
    def get_best_answer(self):
        index = np.argmax(np.array(self.all_answer_ratings))
        return self.all_answer[index]
    
    def __repr__(self):
        return repr(self.all_answer)

class Answerex2obj:
    def __init__(self, label, best_x, other_x):
        self.label = label
        self.best_x = best_x
        self.doc = best_x
        self.other_x = other_x
    
    def __repr__(self):
        return repr(self.label) + repr(self.best_x)
    
def read_and_index_sentiment_examples(infile, indexer, add_to_indexer=False, word_counter=None):
    f = open(infile)
    exs_0 = []
    exs_1 = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            if(len(fields)<6):
                continue
            label = 0 if '0' in fields[5] else 1
            sent_one = fields[3]
            sent_two = fields[4]

            tokenized_sent_one = filter(lambda x: x != '', clean_str(sent_one).rstrip().split(" "))
            tokenized_sent_two = filter(lambda x: x != '', clean_str(sent_two).rstrip().split(" "))

            indexed_q_one = [indexer.get_index(word) if indexer.contains(word) or add_to_indexer else indexer.get_index("UNK") 
            for word in tokenized_sent_one]
            indexed_q_two = [indexer.get_index(word) if indexer.contains(word) or add_to_indexer else indexer.get_index("UNK") 
            for word in tokenized_sent_two]
            
            if(label is 0):
                exs_0.append(QueryExample(indexed_q_one, indexed_q_two, label))
            else: 
                exs_1.append(QueryExample(indexed_q_one, indexed_q_two, label))
    f.close()
    return [exs_0, exs_1]

def read_and_index_answer_examples(text_file, ID_dict, indexer, add_to_indexer=True, word_counter=None):
    f = open(text_file) 
    Dict = {}
    count = 0
    for line in f:
        count += 1
        if len(line.strip()) > 0:
            fields = line.split(" ")
            if(len(fields)<3):
                continue
            
            Q_label = fields[0]
            Answer_rating = fields[1]
            Answer_sent = " ".join(fields[2:])
            tokenized_ans = filter(lambda x: x != '', clean_str(Answer_sent).rstrip().split(" "))
            indexed_ans = [indexer.get_index(word) if indexer.contains(word) or add_to_indexer else indexer.get_index("UNK")
            for word in tokenized_ans]
            
            if(Q_label not in Dict):
                Dict[Q_label] = AnswerExample(Answer_rating, indexed_ans)
            else:
                Dict[Q_label].add_answer(Answer_rating, indexed_ans)
            
    # ADD INDEX of Padding Symbol, Start of Sentence, and End of Sentence
    PAD_SYMBOL = "<PAD>"
    SOS_SYMBOL = "<SOS>"
    EOS_SYMBOL = "<EOS>"
    UNK_SYMBOL = "UNK"

    indexer.get_index(PAD_SYMBOL)
    indexer.get_index(SOS_SYMBOL)
    indexer.get_index(EOS_SYMBOL)
    indexer.get_index(UNK_SYMBOL)

    answer_dict = Dict
    answer_ex = []
    for key in answer_dict:
        label = ID_dict[key]
        best_x = answer_dict[key].get_best_answer()
        best_x.append(indexer.get_index(EOS_SYMBOL))
        other_x = answer_dict[key].all_answer.remove(best_x)
        answer_ex.append(Answerex2obj(label,best_x,other_x))
    return answer_ex

def permutation_generator(l):
  done_perms = set()
  while True:
    perm = tuple(random.sample(l, len(l)))
    if perm not in done_perms:
      done_perms.add(perm)
      yield list(perm)

def read_and_index_question_examples(url, indexer, add_to_indexer=False, word_counter=None):     
    xmlfile = 'para_quest_file.xml'

    # Write data to para_quest_file.xml - Already done
    """
    resp = requests.get(url) # Save the url content to the local file , to be used later by parser
    with open(xmlfile, 'wb') as f:
        f.write(resp.content) 
    """
    tree = ET.parse(xmlfile) # Create element tree object
    root = tree.getroot() # Get root element

    dev_ques_list = [] # Dev list to store all question examples 
    train_ques_list = [] # Train list to store all question examples 
    ID_dict= {}
    class_count = 0
    for elem in root:
        if elem[0].text is None or len(elem) <= 2: # If the original ques is empty or there is only 1 paraphrased question
            continue
        ID = elem.attrib['id']
        ID_dict[ID] = class_count

        tokenized_ques = filter(lambda x: x != '', clean_str(elem[0].text).rstrip().split(" "))
        indexed_ques = [indexer.get_index(word) if indexer.contains(word) or add_to_indexer \
                        else indexer.get_index("UNK") for word in tokenized_ques]
        
        cur_ques_list = []
        size_of_class = 0
    
        dev_idx = random.randint(0, len(elem)-1) # randint(a,b) generates a number between a and b inclusive.
        #dev_idx = 1
    
        if dev_idx == 0:
            dev_ques = Question(class_count, indexed_ques)
            # dev_ques_list.append(Question(class_count, indexed_ques))
        if dev_idx != 0:
            train_ques_list.append(Question(class_count, indexed_ques)) 
            perm = permutation_generator(indexed_ques)
            num_perms = 20 if len(set(indexed_ques)) >= 4 else factorial(len(set(indexed_ques)))
            for i in range(num_perms):
                cur_ques_list.append(Question(class_count, next(perm))) 
                size_of_class += 1

        for idx, subelem in enumerate(elem[1:]):
            tokenized_para_ques = filter(lambda x: x != '', clean_str(subelem.text).rstrip().split(" "))
            indexed_para_ques = [indexer.get_index(word) if indexer.contains(word) or add_to_indexer \
                                 else indexer.get_index("UNK") for word in tokenized_para_ques]
            if idx + 1 == dev_idx:
                dev_ques = Question(class_count, indexed_para_ques)
                #dev_ques_list.append(Question(class_count, indexed_para_ques))
            if idx + 1 != dev_idx:
                train_ques_list.append(Question(class_count, indexed_para_ques)) 
                perm = permutation_generator(indexed_para_ques)
                num_perms = 20 if len(set(indexed_para_ques)) >= 4 else factorial(len(set(indexed_para_ques)))
                for i in range(num_perms):
                    cur_ques_list.append(Question(class_count, next(perm))) 
                    size_of_class += 1
        
        assert(size_of_class == len(cur_ques_list))
        cur_ques_list = cur_ques_list[:40]
        if len(cur_ques_list) >= 40:
            train_ques_list += cur_ques_list
            dev_ques_list.append(dev_ques)
            class_count += 1
            #print(class_count, ":", len(cur_ques_list))
    
    return (ID_dict, train_ques_list, dev_ques_list, class_count)
