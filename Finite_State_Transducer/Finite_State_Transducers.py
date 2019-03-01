from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
CONS = set("bcdfghjklmnprstvwxz")
E = set("E")

# Implement your solution here
def buildFST():
    print("Your task is to implement a better FST in the buildFST() function, using the methods described here")
    print("You may define additional methods in this module (hw1_fst.py) as desired")
    f = FST("q0") # q0 is the initial (non-accepting) state
    f.addState("q1") # a non-accepting state
    f.addState("q2")
    f.addState("q3")
    f.addState("q4")
    f.addState("q5")
    f.addState("q6")
    f.addState("qp")
    f.addState("q7")
    f.addState("q8")
    f.addState("q9")
    f.addState("q10")
    f.addState("q11")
    f.addState("q12")
    f.addState("qn")
    f.addState("qt")
    f.addState("qr")
    f.addState("qe")
    f.addState("qde")
    f.addState("qee")
    f.addState("qn_ex")
    f.addState("qr_ex")
    f.addState("qi")
    f.addState("qy")
    f.addState("q_e")
    f.addState("q_ing") 
    f.addState("q_EOW", True) 
    #
    # The transitions (you need to add more):
    # ---------------------------------------
    # transduce every element in this set to itself: 
    f.addSetTransition("q0", AZ, "q0")
    f.addSetTransition("q0",CONSU,"q2")
    f.addSetTransition("q9",VOWS,"q11")
    f.addSetTransition("q11",NPTR,"q10")
    f.addSetTransition("q0",VOWS,"q9")
    f.addSetTransition("q2",CONS,"qy")
    f.addSetTransition("q9",CONS-NPTR,"q10")
    f.addEpsilonTransition("qy","q_ing")
    f.addEpsilonTransition("q10","q_ing")
    f.addSetTransition("q0",AZ-CONSU-I,"q4")
    f.addSetEpsilonTransition("q2",E,"q3")
    f.addSetTransition("q2",AZ-E,"q5")
    f.addSetTransition("q4",AZ,"q5")
    f.addSetTransition("q5",AZ-E-CONS,"q_ing")
    f.addEpsilonTransition("q3","q_ing")
    f.addTransition("q_ing", "", "ing", "q_EOW")
    f.addSetTransition("q0",AZ,"q6")
    f.addSetEpsilonTransition("q6",I,"qi")
    f.addSetEpsilonTransition("qi",E,"q_e")
    f.addTransition("q_e","","y","qy")
    f.addEpsilonTransition("qy","q_ing") 
    f.addSetTransition("q0",E,"qde")
    f.addSetTransition("qde",E,"qee")
    f.addEpsilonTransition("qee","q_ing")
    f.addSetTransition("q0",CONS,"q7")
    f.addSetTransition("q7",VOWSE,"q8")
    f.addSetTransition("q7",E,"qe")
    f.addSetTransition("q7",E,"q12")
    f.addSetEpsilonTransition("q12",T,"qt")
    f.addSetEpsilonTransition("q8",P,"qp")
    f.addTransition("qp","","pp","q_ing")
    f.addSetEpsilonTransition("q8",N,"qn")
    f.addSetEpsilonTransition("q8",T,"qt")
    f.addSetEpsilonTransition("q8",R,"qr")
    f.addTransition("qn","","nn","q_ing")
    f.addTransition("qt","","tt","q_ing")
    f.addTransition("qr","","rr","q_ing")
    f.addSetTransition("qe",N,"qn_ex")
    f.addSetTransition("qe",R,"qr_ex")
    f.addEpsilonTransition("qn_ex","q_ing") 
    f.addEpsilonTransition("qr_ex","q_ing")

    return f
    

if __name__ == "__main__":
    # Pass in the input file as an argument
    if len(sys.argv) < 2:
        print("This script must be given the name of a file containing verbs as an argument")
        quit()
    else:
        file = sys.argv[1]
    #endif

    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile(file)
