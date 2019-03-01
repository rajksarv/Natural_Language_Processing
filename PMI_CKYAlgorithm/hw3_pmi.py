########################################
## CS447 Natural Language Processing  ##
##           Homework 3               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Use pointwise mutual information to compare words in the movie corpora
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
import math

# ----------------------------------------
#  Data input 
# ----------------------------------------
def readFileToCorpus(f):
    """ Read a text file into a corpus (list of sentences (which in turn are lists of words)). Reads in the text file f which contains one sentence per line. """
    if os.path.isfile(f):
        with open(f, "r") as file: 
            i = 0  
            corpus = []  # this will become a list of sentences
            print("Reading file", f, "...")
            for line in file:
                i += 1
                sentence = line.split() 
                corpus.append(sentence) 
                # if i % 1000 == 0:
                #    sys.stderr.write("Reading sentence " + str(i) + "\n") # just a status message: str(i) turns the integer i into a string, so that we can concatenate it
        return corpus
    else:
        print("Error: corpus file", f, "does not exist")  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit()  # exit the script

class pmiDict(dict): 
    """ A subclass of the standard Python dictionary. Overloads the addition operator so that we can efficiently take the "union" of dictonaries A and B a la A + B. """ 
    def __init__(self): 
        super()

    def __iadd__(self, other):
        """ Overloads the += operator. Given two pmiDicts A and B, A += B is equivalent to A = A.__iadd__(B) """ 
        for k, v in other.items(): 
            if k in self: 
                self[k] += v 
            else: 
                self[k] = v
        return self 

class PMI:
    """ A data structure that parses a corpus and generates the Pointwise Mutual Information between any two words in the corpus. The PMI weighs the association
    between two words with the ratio of the likelihood of the words occurring together in a sentence vs. the likelihood of both words occurring independently.

    The formula used is: I(w_i, w_j) = log_2(P(w_i, w_j) / P(w_i) * P(w_j)), where P(w_i) is the probability that a sentence S drawn uniformly at random from the corpus 
    contains at least one occurrence of word w_i and P(w_i, w_j) is the probability that a sentence S drawn uniformly at random from the corpus contains occurrences of 
    both words w_i and w_j. This equation can be simplified to: I(w_i, w_j) = log_2(P(w_i, w_j)) - log_2(P(w_i)) - log_2(P(w_j))
    Note that multiple occurrences of the same word within a single sentence still only counts as one observed event. 
    """ 
    def __init__(self, corpus):
        """ Initializes the data structures necessary to parse the corpus and calculate the PMI values

        A dictionary A = {w_i : k}, where w_i is the ith word and k is an integer, is used to count the values needed to count P(w_i) 
        A nested dictionary B = {w_i : {w_j : k}}, where w_i is the ith word, w_j is the jth word, and k is an integer, is used to represent the Term-Term Matrix of the corpus 
        """ 
        self.sentenceCount = len(corpus)
        self.wordCount = pmiDict()
        self.pairCount = pmiDict() 
        self.parseCorpus(corpus) 

    def parseCorpus(self, corpus): 
        """ Parses the input corpus and fills out the values of wordCount and pairCount """ 
        for sentence in corpus: 
            wordCount = pmiDict()
            pairCount = pmiDict() 
            for word in sentence: # First pass over the sentence to perform an O(n) operation that makes a vector full of ones for the wordCount vector
                wordCount[word] = min(1, wordCount.get(word, 0) + 1)
                pairCount[word] = pmiDict() 
            for word in sentence: # Second pass over the sentence to perform an O(n^2) operation that basically makes a matrix full of ones for the pairCount matrix 
                for key in pairCount.keys(): 
                    if word == key: 
                        continue 
                    pairCount[key][word] = 1
            self.wordCount += wordCount 
            self.pairCount += pairCount 

    def getPMI(self, w_1, w_2):
        """ Calculates the Pointwise Mutual Information for w_1 and w_2 based on co-occurrence frequencies in the corpus

        The formula used is: I(w_1, w_2) = log_2(P(w_1, w_2) / P(w_1) * P(w_2)) = log_2(P(w_1, w_2)) - log_2(P(w_1)) - log_2(P(w_2))
        """ 
        prob_w_1 = self.wordCount.get(w_1, 0) / self.sentenceCount
        prob_w_2 = self.wordCount.get(w_2, 0) / self.sentenceCount
        joint_prob = self.pairCount.get(w_1, pmiDict()).get(w_2, 0) / self.sentenceCount

        if prob_w_1 == 0 or prob_w_2 == 0 or joint_prob == 0: 
            return 0 
        prob_w_1 = math.log(prob_w_1, 2)
        prob_w_2 = math.log(prob_w_2, 2)
        joint_prob = math.log(joint_prob, 2)
        return joint_prob - prob_w_1 - prob_w_2

    def getVocabulary(self, k):
        """ Returns the list of observed words that appear in at least k sentences """ 
        return [word for (word, count) in self.wordCount.items() if count >= k]

    def getPairsWithMaximumPMI(self, words, N):
        """ Given a list of words and a number N, return a list of N pairs of words that have the highest PMI 
        (without repeated pairs, and without duplicate pairs (wi, wj) and (wj, wi)). 
        Each entry in the list should be a triple (pmiValue, w1, w2), where pmiValue is the PMI of the pair of words (w1, w2)
        """ 
        triples = [] 
        for i in range(len(words) - 1):
            for j in range(i, len(words)): 
                if words[i] == words[j]:
                    continue
                pair = self.pair(words[i], words[j])
                triples.append((self.getPMI(pair[0], pair[1]), pair[0], pair[1]))
        sort = sorted(triples, key = lambda x: x[0], reverse = True)
        return sort[:N]

    def writePairsToFile(self, numPairs, wordPairs, filename): 
        """ Writes the first numPairs entries in the list of wordPairs to a file, along with each pair's PMI 
    
        Input: 
            numPairs [int]: the number of pairs that we want to write to the file 
            wordPairs [list((int, string, string))]: a list of word pairs with their PMI. These triples are generated by the getPairsWithMaximumPMI method
            filename [string]: the file we want to write to 
        Output: 
            None
        """ 
        with open(filename, "w+") as f: 
            for (pmiValue, w_i, w_j) in wordPairs[:numPairs]: 
                f.write("{} {} {}\n".format(pmiValue, w_i, w_j))

    def pair(self, w_1, w_2):
        """ Given two words w_1 and w_2, returns the pair of words in sorted order. This function is order invariant: pair(w_1, w_2) == pair(w_2, w_1) """ 
        return (min(w_1, w_2), max(w_1, w_2))

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    corpus = readFileToCorpus('movies.txt')
    pmi = PMI(corpus)
    lv_pmi = pmi.getPMI("luke", "vader")
    print("  PMI of \"luke\" and \"vader\": ", lv_pmi)
    numPairs = 100
    k_vals = [200, 100, 50, 10, 5, 2] # Fastest to slowest
    for k in k_vals: 
        commonWords = pmi.getVocabulary(k)    # words must appear in least k sentences
        wordPairsWithGreatestPMI = pmi.getPairsWithMaximumPMI(commonWords, numPairs)
        pmi.writePairsToFile(numPairs, wordPairsWithGreatestPMI, "pairs_minFreq=" + str(k) + ".txt")