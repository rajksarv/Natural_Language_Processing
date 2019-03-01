########################################
## CS447 Natural Language Processing  ##
##           Homework 1               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
import os.path
import sys
import random
from operator import itemgetter
from collections import defaultdict
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus to help avoid sess the corpus to help avoid sparsity
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement five kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      d) a bigram model smoothed using absolute discounting (SmoothedBigramModelAD)
      e) a bigram model smoothed using kneser-ney smoothing (SmoothedBigramModelKN)
      """)
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
        def __init__(self, corpus):
                self.counts = defaultdict(float)
                self.total = 0.0
                self.train(corpus)
                #self.getSentenceProbability(["<s>", "I", "UNK", "</s>"])
                #self.getCorpusPerplexity(testcorpus)
                
        def generateSentence(self):
            sentence_gen = []
            word = start
            while(word!=end):
                sentence_gen.append(word)
                word = self.draw(word)
                    #print(word)
            sentence_gen.append(end)
            return sentence_gen
        
        # Add observed counts from corpus to the distribution
        def train(self, corpus):
            print("Training Beginning!")
            for sen in corpus:
                for word in sen:
                    if word == start:
                        continue
                    self.counts[word] += 1.0
                    self.total += 1.0
            print("Training Complete!")

        # Returns the probability of word in the distribution
        def prob(self, word):
            return self.counts[word]/self.total
        #enddef

        # Generate a single random word according to the distribution
        def draw(self,word1):
            rand = random.random()
            for word in self.counts.keys():
                rand -= self.prob(word)
                if rand <= 0.0:
                    return word
        
        def getSentenceProbability(self, sentence):
            total = 0.0
            for word in sentence:
                try:
                    prob = math.log(self.prob(word))
                except:
                    continue
                total += prob
            return math.exp(total)
 
        def getCorpusPerplexity(self, corpus):
            perp = 0.0
            count = 0.0
            for sentence in corpus:
                count+=len(sentence)
                perp -= math.log(self.getSentenceProbability(sentence))
            count = count - len(corpus)
            return math.exp(perp/(count))
        
        def generateSentencesToFile(self, numberOfSentences, filename):
            filePointer = open(filename, 'w+')
            for i in range(0,numberOfSentences):
                sen = self.generateSentence()
                prob = self.getSentenceProbability(sen)

                stringGenerated = str(prob) + " " + " ".join(sen) 
                print(stringGenerated, end="\n", file=filePointer)

    #endddef
#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
        def __init__(self, corpus):
            self.counts = defaultdict(float)
            self.total = 0.0
            self.train(corpus)
        
    # Add observed counts from corpus to the distribution
        def train(self, corpus):
            for sen in corpus:
                for word in sen:
                    if word == start:
                        continue
                    self.counts[word] += 1.0
                    self.total += 1.0

        # Returns the probability of word in the distribution
        def prob(self, word):
            return (self.counts[word] + 1)/(self.total + len(self.counts))
        #enddef

        # Generate a single random word according to the distribution
        def draw(self,word1):
            rand = random.random()
            for word in self.counts.keys():
                rand -= self.prob(word)
                if rand <= 0.0:
                    return word
                
        def generateSentence(self):
            sentence_gen = []
            word = start
            while(word!=end):
                sentence_gen.append(word)
                word = self.draw(word)
                    #print(word)
            sentence_gen.append(end)
            return sentence_gen
                
        def getSentenceProbability(self, sentence):
            total = 0.0
            for word in sentence:
                if(word!=start):
                    try:
                        prob = math.log(self.prob(word))
                    except:
                        continue
                    total += prob
            return math.exp(total)
        
        
        def getCorpusPerplexity(self, corpus):
            perp = 0.0
            count = 0.0
            for sentence in corpus:
                count+=len(sentence)
                perp -= math.log(self.getSentenceProbability(sentence))
            count = count - len(corpus)
            return math.exp(perp/(count))
        
        def generateSentencesToFile(self, numberOfSentences, filename):
            filePointer = open(filename, 'w+')
            for i in range(0,numberOfSentences):
                sen = self.generateSentence()
                prob = self.getSentenceProbability(sen)
                stringGenerated = str(prob) + " " + " ".join(sen) 
                print(stringGenerated, end="\n", file=filePointer)   
    #endddef
#endclass

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
        def __init__(self, corpus):
            self.counts_unigrams = defaultdict(float)
            self.counts_bigrams = defaultdict(float)
            self.total = 0.0
            self.train(corpus)
        
    # Add observed counts from corpus to the distribution
        def train(self, corpus):
            for sen in corpus:
                for word in sen:
                    self.counts_unigrams[word] += 1.0
                    self.total += 1.0
            
            for sen in corpus:
                previous_word = start
                for word in sen:
                    if word == start:
                        continue
                    self.counts_bigrams[(previous_word,word)] += 1.0
                    previous_word = word
        
        # Returns the probability of word in the distribution
        def prob(self,previous_word, word):
            return (self.counts_bigrams[(previous_word,word)] )/self.counts_unigrams[previous_word]
        #enddef

        # Generate a single random word according to the distribution
        def draw(self,word):
            rand = random.random()
            total_words = list(self.counts_unigrams.keys())
            for word1 in total_words:
                try:
                    rand -= self.prob(word,word1)
                except KeyError:
                    rand -= 0
                if rand <= 0.0:
                    return word1
                
        
        def generateSentence(self):
            sentence_gen = []
            word = start
            while(word!=end):
                sentence_gen.append(word)
                word = self.draw(word)
                    #print(word)
            sentence_gen.append(end)
            return sentence_gen
        
        def getSentenceProbability(self, sen):
            total = 0.0
            sentence = list(zip(sen, sen[1:]))
            for pair in sentence:
                try:
                    prob = math.log(self.prob(pair[0],pair[1]))
                    total += prob
                except:
                    prob = float('-inf')   
                    total += prob
            return math.exp(total)
        
        def getCorpusPerplexity(self, corpus):
            perp = 0.0
            count = 0.0
            for sentence in corpus:
                count+=len(sentence)
                try:
                    perp -= math.log(self.getSentenceProbability(sentence))
                except:
                    perp-=float('-inf')
            count = count - len(corpus)
            return math.exp(perp/(count))
        
        def generateSentencesToFile(self, numberOfSentences, filename):
            filePointer = open(filename, 'w+')
            for i in range(0,numberOfSentences):
                sen = self.generateSentence()
                prob = self.getSentenceProbability(sen)
                stringGenerated = str(prob) + " " + " ".join(sen) 
                print(stringGenerated, end="\n", file=filePointer)
    #endddef
#endclass

# Smoothed bigram language model (use absolute discounting for smoothing)
class SmoothedBigramModelAD(LanguageModel):
         def __init__(self, corpus):
            self.counts_unigrams = defaultdict(float)
            self.counts_bigrams = defaultdict(float)
            self.unique_count = defaultdict(float)
            self.total = 0.0
            self.D = self.train(corpus)
            self.SBAD = SmoothedUnigramModel(corpus)
        
    # Add observed counts from corpus to the distribution
         def train(self, corpus):
            for sen in corpus:
                for word in sen:
                    self.counts_unigrams[word] += 1.0
                    self.total += 1.0
            
            for sen in corpus:
                previous_word = start
                for word in sen:
                    if word == start:
                        continue
                    self.counts_bigrams[(previous_word,word)] += 1.0
                    previous_word = word
                    
            iterg = itemgetter(0)
            bigrams = list(self.counts_bigrams.keys())
            counts = list(map(iterg,bigrams))
            for i in self.counts_unigrams.keys():
                self.unique_count[i] = counts.count(i)
            n1 = 0
            n2 = 0
            for i in ((self.counts_bigrams.values())):
                if(i==1):
                    n1+=1
                elif(i==2):
                    n2+=1
            return n1/(n1+(2*n2))
        
        # Returns the probability of word in the distribution
         def prob(self,previous_word, word):
            '''
            S_w = 0
            for (word1,word2) in self.counts_bigrams.keys():
                #print(word1,word2,previous_word)
                if(word1 == previous_word):
                    S_w+=1
            '''
            S_w = self.unique_count[previous_word]
            P_L = self.SBAD.prob(word)
            #print(P_L)
            C_w = self.counts_unigrams[previous_word]
            C_ww = self.counts_bigrams[(previous_word,word)]
            D = self.D
            #print((max((C_ww - D),0)/C_w), P_L,((D/C_w) * S_w * P_L))
            #print(S_w)
            return (max((C_ww - D),0)/C_w) + ((D/C_w) * S_w * P_L)
        #enddef

        # Generate a single random word according to the distribution
         def draw(self,word):
            rand = random.random()
            total_words = list(self.counts_unigrams.keys())
            for word1 in total_words:
                try:
                    rand -= self.prob(word,word1)
                except KeyError:
                    rand -= 0
                if rand <= 0.0:
                    return word1
                
         def generateSentence(self):
            sentence_gen = []
            word = start
            while(word!=end):
                sentence_gen.append(word)
                word = self.draw(word)
                    #print(word)
            sentence_gen.append(end)
            return sentence_gen
        
         def getSentenceProbability(self, sen):
            total = 0.0
            sentence = list(zip(sen, sen[1:]))
            for pair in sentence:
                prob = math.log(self.prob(pair[0],pair[1]))
                total += prob
            return math.exp(total)
        
         def getCorpusPerplexity(self, corpus):
            perp = 0.0
            count = 0.0
            for sentence in corpus:
                count+=len(sentence)
                perp -= math.log(self.getSentenceProbability(sentence))
            count = count - len(corpus)
            return math.exp(perp/(count))
        
    
         def generateSentencesToFile(self, numberOfSentences, filename):
            filePointer = open(filename, 'w+')
            for i in range(0,numberOfSentences):
                sen = self.generateSentence()
                prob = self.getSentenceProbability(sen)
                stringGenerated = str(prob) + " " + " ".join(sen) 
                print(stringGenerated, end="\n", file=filePointer)    #endddef
#endclass

# Smoothed bigram language model (use absolute discounting and kneser-ney for smoothing)
class SmoothedBigramModelKN(LanguageModel):
         def __init__(self, corpus):
            self.counts_unigrams = defaultdict(float)
            self.counts_bigrams = defaultdict(float)
            self.unique_count_prev = defaultdict(float)
            self.unique_count_next = defaultdict(float)
            self.total = 0.0
            self.D = self.train(corpus)
            #self.SBAD = SmoothedUnigramModel(corpus)
        
    # Add observed counts from corpus to the distribution
         def train(self, corpus):
            for sen in corpus:
                for word in sen:
                    self.counts_unigrams[word] += 1.0
                    self.total += 1.0
            
            for sen in corpus:
                previous_word = start
                for word in sen:
                    if word == start:
                        continue
                    self.counts_bigrams[(previous_word,word)] += 1.0
                    previous_word = word
                    
            iterg = itemgetter(0)
            itern = itemgetter(1)
            bigrams = list(self.counts_bigrams.keys())
            counts = list(map(iterg,bigrams))
            countsn = list(map(itern,bigrams))
            for i in self.counts_unigrams.keys():
                self.unique_count_prev[i] = counts.count(i)
                self.unique_count_next[i] = countsn.count(i)
            n1 = 0
            n2 = 0
            for i in ((self.counts_bigrams.values())):
                if(i==1):
                    n1+=1
                elif(i==2):
                    n2+=1
            return n1/(n1+(2*n2))
        
        # Returns the probability of word in the distribution
         def prob(self,previous_word, word):
                '''
                S_w = 0
                for (word1,word2) in self.counts_bigrams.keys():
                    
                    if(word1 == previous_word):
                        S_w+=1
                '''
                S_w = self.unique_count_prev[previous_word]
                P_C_NR = self.unique_count_next[word]
                P_C_DR = 0
                for i in self.counts_bigrams.values():
                    if(i>0):
                        P_C_DR+=1
                P_C = P_C_NR/P_C_DR
                C_w = self.counts_unigrams[previous_word]
                C_ww = self.counts_bigrams[(previous_word,word)]
                D = self.D
                return (max((C_ww - D),0)/C_w) + ((D/C_w) * S_w * P_C)
        #enddef

        # Generate a single random word according to the distribution
         def draw(self,word):
            rand = random.random()
            total_words = list(self.counts_unigrams.keys())
            for word1 in total_words:
                try:
                    rand -= self.prob(word,word1)
                except KeyError:
                    rand -= 0
                if rand <= 0.0:
                    return word1
                
         def generateSentence(self):
            sentence_gen = []
            word = start
            while(word!=end):
                sentence_gen.append(word)
                word = self.draw(word)
                    #print(word)
            sentence_gen.append(end)
            return sentence_gen
        
         def getSentenceProbability(self, sen):
            total = 0.0
            sentence = list(zip(sen, sen[1:]))
            for pair in sentence:
                prob = math.log(self.prob(pair[0],pair[1]))
                total += prob
            return math.exp(total)
        
         def getCorpusPerplexity(self, corpus):
            perp = 0.0
            count = 0.0
            for sentence in corpus:
                count+=len(sentence)
                perp -= math.log(self.getSentenceProbability(sentence))
            count = count - len(corpus)
            return math.exp(perp/(count))
        
         def generateSentencesToFile(self, numberOfSentences, filename):
            filePointer = open(filename, 'w+')
            for i in range(0,numberOfSentences):
                sen = self.generateSentence()
                prob = self.getSentenceProbability(sen)
                stringGenerated = str(prob) + " " + " ".join(sen) 
                print(stringGenerated, end="\n", file=filePointer)    #endddef
#endclass



# Sample class for a unsmoothed unigram probability distribution
# Note: 
#       Feel free to use/re-use/modify this class as necessary for your 
#       own code (e.g. converting to log probabilities after training). 
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        return self.counts[word]/self.total
    #enddef

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
#endclass

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    
    vocab = set()
    # Please write the code to create the vocab over here before the function preprocessTest
    print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")


    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    # Run sample unigram dist code
    unigramDist = UnigramDist(trainCorpus)
    print("Sample UnigramDist output:")
    print("Probability of \"vader\": ", unigramDist.prob("vader"))
    print("Probability of \""+UNK+"\": ", unigramDist.prob(UNK))
    print("\"Random\" draw: ", unigramDist.draw())


