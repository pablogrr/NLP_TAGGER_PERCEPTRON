import scipy
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

#######
#To get corpus and word to positon, tag to position dictonary and their inverse
#######

def get_corpus_and_word_dict(df_train,df_test):
    corpus = set(df_train.words).union(df_test.words)
    corpus_list = sorted(list(corpus))
    word_to_pos = {corpus_list[i]:i for i in range(len(corpus))}
    pos_to_word = {v: k for k, v in word_to_pos.items()}
    return corpus, word_to_pos, pos_to_word

def get_tag_dict(df_train):
    tag_set = set(df_train.tags)
    tags_list = sorted(list(tag_set))
    tag_to_pos = {tags_list[i]:i for i in range(len(tags_list))}
    pos_to_tag = {v: k for k, v in tag_to_pos.items()}
    return tag_to_pos, pos_to_tag

#######
#To get all the sentences in different list of tokens,
#and all the tags in different lists of labels
#######

def get_X_Y(data):
    ids = set(data.sentence_id)
    X=[];Y=[]
    for i in ids: 
        sent = [w for w in data[data.sentence_id==i].words]
        tags = [t for t in data[data.sentence_id==i].tags]
        X.append(sent);Y.append(tags)
        
    return X,Y

#######
#Functions to evaluate the models
#######


def accuracy_tot(sequences, sequences_predictions):
    '''
    Evaluates the total accuracy
    '''
    correct = 0
    total   = 0
    N_seq   = len(sequences)
    
    for i in range(N_seq):
        y_true = np.array(sequences[i].y)
        y_pred = sequences_predictions[i].y
        correct  = correct + np.count_nonzero(y_true-y_pred==0)
        total    = total + len(y_true)
    
    return correct/total  

def accuracy_not_O(sequences, sequences_predictions, tag_to_pos):
    '''
    Evaluates accuracy without considering 'O'
    '''
    correct_not_O = 0
    total_not_O   = 0
    N_seq         = len(sequences)
    
    for i in range(N_seq):
        mask = np.array(sequences[i].y) != tag_to_pos['O']
        y_cleaned_true = np.array(sequences[i].y)[mask]
        y_cleaned_pred = sequences_predictions[i].y[mask]
        correct_not_O  = correct_not_O + np.count_nonzero(y_cleaned_true-y_cleaned_pred==0)
        total_not_O    = total_not_O + len(y_cleaned_true)
    
    return correct_not_O/total_not_O  

def get_f1_score(sequences, sequences_predictions, pos_to_tag, ave='weighted'):
    '''
    Evaluates f1 score
    '''
    N_seq        = len(sequences)
    
    chained_true = [sequences[i].y for i in range(N_seq)]
    merged_true  = list(itertools.chain(*chained_true))
    merged_true_tag = [pos_to_tag[i] for i in merged_true]

    chained_pred = [sequences_predictions[i].y.tolist() for i in range(N_seq)]
    merged_pred  = list(itertools.chain(*chained_pred))
    merged_pred_tag = [pos_to_tag[i] for i in merged_pred]
    
    f1score = f1_score(merged_true, merged_pred, average=ave)
    cm = confusion_matrix(merged_true_tag, merged_pred_tag, labels=list(pos_to_tag.values()))
    
    return f1score, cm

#based on https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

#######
#To decode an arbirtary list based on viterbi decoding
#######

def string_list_decoder(string_list, perceptron):
    decoded=[]
    for p in string_list:
        new_seq = skseq.sequences.sequence.Sequence(x=p.split(), y=[int(0) for w in p.split()])
        decoded.append(perceptron.viterbi_decode(new_seq)[0].to_words(train_seq,
                                       only_tag_translation=True))
    return decoded
    

