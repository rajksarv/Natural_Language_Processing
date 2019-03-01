########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log
import math
import numpy as np

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

class Distributions():
    def __init__(self, corpus):
            self.counts_unigrams = defaultdict(float)
            self.counts_bigrams = defaultdict(float)
            self.count_tags = defaultdict(float)
            self.transition_probabilities = defaultdict(float)
            self.total = 0.0
            self.emission_probabilities = defaultdict(float)
            self.start_counts = defaultdict(float)
            self.starting_probabilities = defaultdict(float)
            self.train(corpus) 
            self.length = len(self.counts_unigrams)
            
            
    def getlist(self,corpus):
        lis = []
        for sen in corpus:
            new = []
            for word in sen:
                new.append((word.word,word.tag))
            lis.append(new)
        return lis
    
    
    def train(self,corpus):
        tagget = itemgetter(1)
        wordget = itemgetter(0)
        tuples = self.getlist(corpus)
        for sentence in tuples:
            self.start_counts[sentence[0][1]] += 1
        tags = []
        for i in tuples:
            tags.append(list(map(tagget,i)))
        tuples1 = []
        for i in tuples:
            for j in i:
                tuples1.append(j)
        transition = []
        all_tags = []
        for sent in tags:
            transition.append(list(zip(sent[:],sent[1:])))
            for elem in sent:
                all_tags.append(elem)
        fin_trans = []
        for sets in transition:
            for couple in sets:
                fin_trans.append(couple)

        self.unique_tags = list(set(all_tags))
        for tag in self.unique_tags:
            self.starting_probabilities[tag] = self.start_counts[tag]/len(tuples)

        self.words = list(set(list(map(wordget,tuples1))))

        for tag in list(set(all_tags)):
            self.counts_unigrams[tag] = all_tags.count(tag)
        
      
        for pair in list(set(fin_trans)):
            self.count_tags[pair] = fin_trans.count(pair)
            
       
        for tag1 in self.unique_tags:
            for tag2 in self.unique_tags:
                if (tag1,tag2) in self.transition_probabilities.keys():
                    continue
                else:
                    self.transition_probabilities[(tag1,tag2)] = self.transition_prob(tag1,tag2)
           
         
        
        for wordset in ((tuples1)):
            
            self.counts_bigrams[wordset] += 1
        for wordset in list(set(tuples1)):
            self.emission_probabilities[wordset] = self.prob(wordset)
           
        
        
    def transition_prob(self,previous_word, word):
        try:
            prob = (self.count_tags[(previous_word,word)] + 1 )/(self.counts_unigrams[previous_word]+(len(self.counts_unigrams.keys())))
        except:
            prob = (1)/(self.counts_unigrams[previous_word]+(len(self.counts_unigrams.keys())))
        return prob
        
    def prob(self,wordset):
        tag = wordset[1]
        return self.counts_bigrams[wordset]/self.counts_unigrams[tag]


# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        ### Initialize the rest of your data structures here ###

    def minfreq(self,corpus):
    #find all the rare words
        freqDict = defaultdict(int)
        for sentence in corpus:
            for word in sentence:
               freqDict[word] += 1

        for sentence in corpus:
            for i in range(0, len(sentence)):
                word = sentence[i]
                if word not in self.emission_matrix.words:
                    sentence[i] = UNK

        #bookend the sentences with start and end tokens
        
    
        return corpus

    def preprocess(self,corpus):
    #find all the rare words
        freqDict = defaultdict(int)
        for sentence in corpus:
            for word in sentence:
               freqDict[word.word] += 1
        #endfor
        #endfor

        #replace rare words with unk
        for sentence in corpus:
            for i in range(0, len(sentence)):
                word = sentence[i].word
                if freqDict[word] < self.minFreq:
                    sentence[i].word = UNK

        return corpus


    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile)
        datatokenised = self.preprocess(data)
   
        self.emission_matrix = Distributions(datatokenised)


    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        
        data = self.minfreq(data)
        f=open(outFile, 'w+')
        for sen in data:
            #print(sen)
            viterbi_tags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+viterbi_tags[i]+" "
            #print(senString)
            print(senString.rstrip(), end="\n", file=f)
          

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        for word in range(len(words)):
            elem = words[word]
            if elem not in self.emission_matrix.words:
                words[word] = UNK
        ncols = len(words)
        nrows = len(self.emission_matrix.unique_tags)
        tags = self.emission_matrix.unique_tags
        trellis = np.zeros(shape=(nrows,ncols),dtype=object)
        for row in range((nrows)):
            if (tags[row]) in self.emission_matrix.starting_probabilities.keys() and (words[0],tags[row]) in self.emission_matrix.emission_probabilities.keys():
                trellis[row][0] = list([(-np.log(self.emission_matrix.starting_probabilities[(tags[row])])-np.log(self.emission_matrix.emission_probabilities[(words[0],tags[row])])),tags[row],(row,0)])
            else: 
                trellis[row][0] = list([9999999,tags[row],(row,0)])
        for word in range(1,ncols):
            for row in range((nrows)):
                trellis[row][word] = list((9999999,0,0))
                for row_prime in range(nrows):
                    if ((trellis[row_prime][word-1])[0]) == 9999999:
                        continue
                    temp = ((trellis[row_prime][word-1])[0])-np.log(self.emission_matrix.transition_probabilities[(tags[row_prime],tags[row])])
                    if temp < (trellis[row][word])[0]:
                        trellis[row][word] = list([temp,tags[row_prime],(row_prime,word-1)])
                if (words[word],tags[row]) in self.emission_matrix.emission_probabilities.keys():       
                    (trellis[row][word])[0] -= np.log(self.emission_matrix.emission_probabilities[(words[word],tags[row])])
                else:
                    (trellis[row][word])[0] = 9999999
        self.trellis = trellis
        tagset = []
        maximum_value = min(trellis[:,-1])
        dest = maximum_value[-1]
        maxtag = tags[np.argmin((trellis[:,-1]))]
        tagset.append(maximum_value[1])
        for j in range(1,ncols):
            tagset.append(trellis[dest][1])
            dest = trellis[dest][-1]
        tagset.pop(-1)
        tagset.insert(0,maxtag)    
        tagset.reverse()
        return tagset
            
               
                        
               
            
if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
