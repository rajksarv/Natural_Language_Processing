########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter
from hw2_hmm import TaggedWord
import numpy as np
# A class for evaluating POS-tagged data
class Eval:
    ################################
    #intput:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    def __init__(self, goldFile, testFile):
        self.gold = self.readLabeledData(goldFile)
        self.goldtags = []
        self.size = 0
        for i in self.gold:
            self.size += len(i)
            for j in i:
                self.goldtags.append(j)
        self.test = self.readLabeledData(testFile)
        gold_sentence = [[word.tag for word in sent] for sent in self.gold]
        base_sentence = [[word.tag for word in sent] for sent in self.test]
        self.gold_sentence = gold_sentence
        self.test_sentence = base_sentence
        self.confusion_matrix_input = self.conjoin(self.test_sentence,self.gold_sentence)

    def readLabeledData(self,inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence)
        return sens
    
    def conjoin(self,gold,test):
        conjoined = []
        for i in range(len(gold)):
            for j in range(len(gold[i])):
                conjoined.append(list(((gold[i][j],test[i][j]))))
        return conjoined
            
    
    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        accuracy=0
        for word in range(len(self.gold_sentence)):
            for pair in zip(self.gold_sentence[word],self.test_sentence[word]):
                if pair[0]==pair[1]:
                    accuracy+=1
        return (accuracy/self.size)

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getSentenceAccuracy(self):
        sent_accuracy = 0
        for base in zip(self.gold_sentence,self.test_sentence):
            if base[0]==base[1]:
                sent_accuracy+=1
        print(sent_accuracy/len(self.gold_sentence))
        return(sent_accuracy/len(self.gold_sentence))

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        tags = list(set(self.goldtags))
        confusion_matrix = np.zeros(shape=(len(tags)+1,len(tags)+1),dtype=object)
        for i in range(len(tags)):
            if i!=0:
                confusion_matrix[i,0] = tags[i]
                confusion_matrix[0,i] = tags[i]
                confusion_matrix[-1,0] = tags[0]
                confusion_matrix[0,-1] = tags[0]
        for correct in confusion_matrix[:,0]:
            for predicted in confusion_matrix[0,:]:
                if correct!=0 and predicted !=0:
                    confusion_matrix[tags.index(correct)+1,tags.index(predicted)+1] = self.confusion_matrix_input.count([correct,predicted])
        np.savetxt(outFile,confusion_matrix,fmt = '%s',delimiter=',')
        
    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    def getPrecision(self, tagTi):
        self.confus_input = self.conjoin(self.gold_sentence,self.test_sentence)
        correct = self.confus_input.count([tagTi,tagTi])
        total = 0
        for i in self.confus_input:
            if i[1]==tagTi:
                total+=1
        return correct/total

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        correct = self.confus_input.count([tagTj,tagTj])
        total = 0
        for i in self.confus_input:
            if i[0]==tagTj:
                total+=1
        return correct/total


if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print("Call hw2_eval_hmm.py with two arguments: gold.txt and out.txt")
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        print("Token accuracy: ", eval.getTokenAccuracy())
        print("Sentence accuracy: ", eval.getSentenceAccuracy())
        # Calculate recall and precision
        print("Recall on tag NNP: ", eval.getPrecision('NNP'))
        print("Precision for tag NNP: ", eval.getRecall('NNP'))
        # Write a confusion matrix
        eval.writeConfusionMatrix("conf_matrix.txt")
