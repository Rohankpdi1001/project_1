import os
import math

def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r', encoding ="'utf-8") as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    with open(filepath, "r", encoding='utf-8') as output:
        for word in output:
            word = word.strip()
            if word in vocab:
                if word not in bow:
                    bow[word] = 1
                else:
                    bow[word] += 1
            else:
                if None not in bow.keys():
                    bow[None] = 1
                else:
                    bow[None] += 1
    return bow


def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    numLabel1 = 0
    numLabel2 = 0
    for i in range(len(training_data)):
        if training_data[i]['label'] == '2016':
            numLabel1+=1
        elif training_data[i]['label'] == '2020':
            numLabel2+=1
        
        prob_label_1 = math.log((numLabel1 + smooth)/((numLabel1 + numLabel2) + 2))
        prob_label_2 = math.log((numLabel2 + smooth)/((numLabel1 + numLabel2) + 2))

    logprob = {label_list[0] : prob_label_1, label_list[1] : prob_label_2}

    return logprob

def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    count_words = {}
    word_count = 0
    none_count = 0
    for i in range(len(vocab)):
        temp = vocab[i]
        for j in range(len(training_data)):
            if training_data[j]['label'] == label:
                bow_label = training_data[j]['bow']
                if temp in bow_label and temp not in count_words:
                    count_words[temp] = bow_label[temp]
                elif temp in bow_label and temp in count_words:
                    count_words[temp] += bow_label[temp]
                elif temp not in bow_label and temp not in count_words:
                    count_words[temp] = 0

    for number in training_data:
        if number['label'] == label:
            temp = number['bow']
            word_count += sum(temp.values())
            if None in temp:
                none_count += temp[None]

    for i in range(len(vocab)):
        word_prob[vocab[i]] = math.log(((count_words[vocab[i]] + smooth)/(word_count + smooth * (len(vocab) + 1))))

    word_prob[None] = math.log(((none_count + smooth)/(word_count + smooth * (len(vocab) + 1))))

    return word_prob

    

def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    log_prior = prior(training_data, label_list)
    log_pword_2020 = p_word_given_label(vocab,training_data,'2020')
    log_pword_2016 = p_word_given_label(vocab,training_data,'2016')

    retval['vocabulary'] = vocab
    retval['log prior'] = log_prior
    retval['log p(w|y=2016)'] = log_pword_2016
    retval['log p(w|y=2020)'] = log_pword_2020
    return retval


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}

    retval['predicted y'] = 0
    retval['log p(y=2016|x)'] = model['log prior']['2016']
    retval['log p(y=2020|x)'] = model['log prior']['2020']

    temp_bow = create_bow(model['vocabulary'],filepath)

    for i in temp_bow:
        retval['log p(y=2016|x)'] += model['log p(w|y=2016)'][i] * temp_bow[i]
        retval['log p(y=2020|x)'] += model['log p(w|y=2020)'][i] * temp_bow[i]

    if retval['log p(y=2016|x)'] > retval['log p(y=2020|x)']:
        retval['predicted y'] = '2016'
    else:
        retval['predicted y'] = '2020'

    return retval

