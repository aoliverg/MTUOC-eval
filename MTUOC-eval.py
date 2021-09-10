#!/usr/bin/python3
#    MTUOC-eval
#    Copyright (C) 2020  Antoni Oliver
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import os
import argparse
import codecs

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.nist_score import sentence_nist

from nltk.metrics import edit_distance


import numpy

import subprocess


import importlib

#IMPORTS FOR YAML
import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def ter_corpus(referencesTOK,hypothesisTOK):
    terlist=[]
    shyp=codecs.open("hyp.txt","w",encoding="utf-8")
    srefs=codecs.open("refs.txt","w",encoding="utf-8")
    for posicio in range(0,len(hypothesisTOK)):
        mt=" ".join(hypothesisTOK[posicio])
        cadena=mt+" "+"(SEG-"+str(posicio)+")"
        shyp.write(cadena+"\n")
        for reference in referencesTOK[posicio]:
            pe=" ".join(reference)
            cadena=pe+" "+"(SEG-"+str(posicio)+")"
            srefs.write(cadena+"\n")
    shyp.close()
    srefs.close()

    p = subprocess.Popen("java -jar tercom-0.10.0.jar -r refs.txt -h hyp.txt -N -o sum -n hter", stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    ter=None
      
    entradater=codecs.open("hter.sum","r",encoding="utf-8")
    for linia in entradater:
        linia=linia.strip()
        camps=linia.split("|")
        if camps[0].startswith("TOTAL"):
            ter=float(camps[-1].replace(",",".").strip())/100
            
    return(ter)

def wer_score(ref, hyp):
    """ 
    code from: https://github.com/gcunhase/NLPMetrics
    Calculation of WER with Levenshtein distance.
    Time/space complexity: O(nm)
    Source: https://martin-thoma.com/word-error-rate-calculation/
    :param ref: reference text (separated into words)
    :param hyp: hypotheses text (separated into words)
    :return: WER score
    Modified to return the value divide by the number of words of the reference
    $ WER = \frac{S+D+I}{N} = \frac{S+D+I}{S+D+C} $
    S: number of substitutions, D: number of deletions, I: number of insertions, C: number of the corrects, N: number of words in the reference ($N=S+D+C$)
    """

    # Initialization
    d = numpy.zeros([len(ref) + 1, len(hyp) + 1], dtype=numpy.uint8)
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # print(d)

    # Computation
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    # print(d)
    return d[len(ref)][len(hyp)]/len(ref)

def wer_corpus(references,hypothesis):
    cont=0
    weracu=0
    for posicio in range(0,len(hypothesis)):
        werpos=[]
        cont+=1
        for reference in references[posicio]:
            wer=wer_score(reference,hypothesis[posicio])
            werpos.append(wer)
        wer=min(werpos)
        weracu+=wer
    wer=weracu/cont
    return(wer)
    


    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MTUOC script for automatic machine translation evaluation. Available measures: BLEU, NIST, WER')
    parser.add_argument('--tokenizer', action="store", dest="tokenizer", help='The tokenizer to be used.',required=True)
    parser.add_argument('--refs', action="store", dest="refs", help='The file with the reference segments. Multiple references per segment can be used. They should be separated by a tabulator',required=True)
    parser.add_argument('--hyp', action="store", dest="hyp", help='The target language.',required=True)
    parser.add_argument('--BLEU', action='store_true', default=False, dest='mbleu',help='Calculate BLEU.')
    parser.add_argument('--NIST', action='store_true', default=False, dest='mnist',help='Calculate NIST.')
    parser.add_argument('--WER', action='store_true', default=False, dest='mwer',help='Calculate WER.')
    parser.add_argument('--ED', action='store_true', default=False, dest='med',help='Calculate Edit Distant percent.')
    parser.add_argument('--TER', action='store_true', default=False, dest='mter',help='Calculate TER (using termcom).')
    
    
    
    args = parser.parse_args()
    
    stream = open('config.yaml', 'r',encoding="utf-8")
    config=yaml.load(stream,Loader=Loader)
    filepathin=config['Filepath']['path_in']
    tokenizerlist=config['Tokenizers']['list']
    defaultokenizer=config['Tokenizers']['default']
    doBLEU=False
    doNIST=False
    doWER=False
    doED=False
    doTER=False
    if args.mbleu: doBLEU=True
    if args.mnist: doNIST=True
    if args.mwer: doWER=True
    if args.med: doED=True
    if args.mter: doTER=True

    if not args.mbleu and not args.mnist and not args.mwer and not args.med and not args.mter:
        doBLEU=True
        doNIST=True
        doWER=True
        doED=True
        doTER=True

    #############
    #ROUND VALUES:
    rbleu=int(config['Round']['rbleu'])
    rnist=int(config['Round']['rnist'])
    rwer=int(config['Round']['rwer'])
    red=int(config['Round']['red'])
    rter=int(config['Round']['rter'])
    #############

    rfile=codecs.open(args.refs,"r",encoding="utf-8")
    hfile=codecs.open(args.hyp,"r",encoding="utf-8")
    
    sys.path.append(os.getcwd())
    
    tokenizer=importlib.import_module(args.tokenizer.replace(".py",""))
        
    references=[]
    references_tok=[]

    for linia in rfile:
        linia=linia.rstrip()
        #pot haver més d'una referència separada per tabuladors
        rd=[]
        rd_tok=[]
        linia=linia.rstrip()
        #pot haver més d'una referència separada per tabuladors
        for segment in linia.split("\t"):
            tokens=tokenizer.tokenize(segment).split(" ")
            rd.append(segment)
            rd_tok.append(tokens)
        references.append(rd)
        references_tok.append(rd_tok)
        
    hypothesis=[]
    hypothesis_tok=[]

    for linia in hfile:
        linia=linia.rstrip()
        tokens=tokenizer.tokenize(linia).split(" ")
        hypothesis.append(linia)
        hypothesis_tok.append(tokens)

 

    
    
    if doBLEU:
        try:
            BLEU=corpus_bleu(references_tok,hypothesis_tok)
            print("BLEU:     ",round(BLEU,rbleu))
        except:
            print("ERROR: unable to calculate BLEU.")
    if doNIST:
        try:
            NIST=corpus_nist(references_tok,hypothesis_tok)
            print("NIST:     ",round(NIST,rnist))
        except:
            print("ERROR: unable to calculate NIST:")
    if doWER:
        try:
            WER=wer_corpus(references_tok,hypothesis_tok)
            print("WER:      ",round(WER,rwer))
        except:
            print("ERROR: unable to calculate WER.")
        
    if doED:
        try:
            edtotal=0
            chartotal=0
            for i in range(0,len(hypothesis)):
                editmin=100000000
                chartotal+=len(hypothesis[i])
                for h in references[i]:
                    ed=edit_distance(hypothesis[i],h)
                    if ed<editmin:
                        editmin=ed
                edtotal+=editmin
            
            EditDistance=100*(edtotal/chartotal)
            print("%EdDist:  ",round(EditDistance,red))
        except:
            print("ERROR: unable to calculate Ed")
        
    if doTER:
        try:
            TERcorpus=ter_corpus(references_tok,hypothesis_tok)
            print("TER:      ",round(TERcorpus,rter))
        except:
            print("ERROR: unable to calculate TER",sys.exc_info())

