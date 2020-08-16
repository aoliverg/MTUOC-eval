#    MTUOC-eval-GUI
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
import argparse
import codecs
import time

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.nist_score import sentence_nist

from nltk.metrics import edit_distance

import numpy

import xml.etree.ElementTree as ET

import os

from pathlib import Path

from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter import messagebox
from tkinter import ttk
import tkinter.scrolledtext as scrolledtext

#IMPORTS FOR YAML
import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
from nltk.translate.nist_score import sentence_nist

from nltk.metrics import edit_distance


import numpy

import subprocess


import importlib


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


def open_references():
    references_file = askopenfilename(initialdir = filepathin, filetypes = (("All files", "*"),("txt files","*.txt")))
    F_frame_E_Ref.delete(0,END)
    F_frame_E_Ref.insert(0,references_file)
    return
    
def open_hypothesis():
    hypothesis_file = askopenfilename(initialdir = filepathin, filetypes = (("All files", "*"),("txt files","*.txt")))
    F_frame_E_Hyp.delete(0,END)
    F_frame_E_Hyp.insert(0,hypothesis_file)
    return

def copy_results():
    main_window.clipboard_append(results_frame_text.get("1.0",END))
    
def calculate():
    
    doBLEU=True
    doNIST=True
    doWER=True
    doED=True
    doTER=True
    
    rfilename=F_frame_E_Ref.get()
    hfilename=F_frame_E_Hyp.get()
    
    rfile=codecs.open(rfilename,"r",encoding="utf-8")
    hfile=codecs.open(hfilename,"r",encoding="utf-8")
    
    sys.path.append(os.getcwd())
    selectedtokenizer=combo_tokenizersF.get()
    tokenizer=importlib.import_module(selectedtokenizer)
    
        
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
            cadena="BLEU:    "+str(round(BLEU,rbleu))
            print(cadena)
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")
        except:
            print("ERROR: unable to calculate BLEU.",sys.exc_info())
            cadena="BLEU:  Unable to calculate BLEU"
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")
    if doNIST:
        try:
            NIST=corpus_nist(references_tok,hypothesis_tok)
            cadena="NIST:    "+str(round(NIST,rnist))
            print(cadena)
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")
        except:
            print("ERROR: unable to calculate NIST:",sys.exc_info())
            cadena="NIST:  Unable to calculate NIST"
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")
    if doWER:
        try:
            WER=wer_corpus(references_tok,hypothesis_tok)
            cadena="WER:     "+str(round(WER,rwer))
            print(cadena)
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")
        except:
            print("ERROR: unable to calculate WER.",sys.exc_info())
            cadena="WER:  Unable to calculate WER"
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")
        
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
            cadena="%EdDist: "+str(round(EditDistance,red))
            print(cadena)
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")
        except:
            print("ERROR: unable to calculate Ed",sys.exc_info())
            cadena="%EdDIst: Unable to calculate Ed"
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")
        
    if doTER:
        try:
            TERcorpus=ter_corpus(references_tok,hypothesis_tok)
            cadena="TER:     "+str(round(TERcorpus,rter))
            print(cadena)
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")
        except:
            print("ERROR: unable to calculate TER",sys.exc_info())
            cadena="TER:  Unable to calcualte TER"
            results_frame_text.insert(INSERT, cadena)
            results_frame_text.insert(INSERT, "\n")

stream = open('config.yaml', 'r',encoding="utf-8")
config=yaml.load(stream,Loader=Loader)
filepathin=config['Filepath']['path_in']
tokenizerlist=config['Tokenizers']['list']
defaultokenizer=config['Tokenizers']['default']

#############
#ROUND VALUES:
rbleu=int(config['Round']['rbleu'])
rnist=int(config['Round']['rnist'])
rwer=int(config['Round']['rwer'])
red=int(config['Round']['red'])
rter=int(config['Round']['rter'])
#############

main_window = Tk()
main_window.title("MTUOC Eval v. 1.0")
main_window.clipboard_clear()
notebook = ttk.Notebook(main_window)

#FILES
F_frame = Frame(notebook)
F_frame_B_Ref=Button(F_frame, text = str("Select References"), command=open_references,width=15)
F_frame_B_Ref.grid(row=0,column=0)
F_frame_E_Ref = Entry(F_frame,  width=50)
F_frame_E_Ref.grid(row=0,column=1)
F_frame_B_Hyp=Button(F_frame, text = str("Select Hypothesis"), command=open_hypothesis,width=15)
F_frame_B_Hyp.grid(row=1,column=0)
F_frame_E_Hyp = Entry(F_frame,  width=50)
F_frame_E_Hyp.grid(row=1,column=1)
F_frame_L_tokenizers = Label(F_frame,text="T.L. tokenizer:").grid(sticky="W",row=3,column=0)

combo_tokenizersF = ttk.Combobox(F_frame,state="readonly",values=tokenizerlist)
combo_tokenizersF.grid(sticky="W",row=3,column=1)
position=tokenizerlist.index(defaultokenizer)
combo_tokenizersF.current(position)

F_frame__B_Go=Button(F_frame, text = str("Calculate"), command=calculate,width=15)
F_frame__B_Go.grid(sticky="E",row=4,column=1)

results_frame = Frame(notebook)
results_frame_text=scrolledtext.ScrolledText(results_frame,height=7)
results_frame_text.grid(row=0,column=0)
results_frame_B=Button(results_frame, text = str("Copy to clipboard"), command=copy_results,width=20)
results_frame_B.grid(row=1,column=0)



notebook.add(F_frame, text="Files", padding=30)
notebook.add(results_frame, text="Results", padding=30)

notebook.pack()
notebook.pack_propagate(0) #Don't allow the widgets inside to determine the frame's width / height
notebook.pack(fill=BOTH, expand=1)
main_window.mainloop()

