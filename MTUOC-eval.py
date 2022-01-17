#    MTUOC-eval
#    Copyright (C) 2022  Antoni Oliver
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
from tkinter.filedialog import asksaveasfilename
from tkinter.filedialog import askdirectory
from tkinter import messagebox
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

import xlsxwriter

import difflib

def differences(a,b):
    d = difflib.Differ()
    diff = d.compare(a, b)
    cont=0
    result=[]
    for d in diff:
        d.strip()
        accio=d[0]
        lletra=d[-1]
        if accio=="":
            cadena=lletra
            result.append(cadena)
        elif accio=="+":
            cadena=lletra+"\u0332"
            result.append(cadena)
        elif accio=="-":
            cadena=lletra+"\u0336"
            result.append(cadena)
        else:
            result.append(lletra)
    amod="".join(result)
    return(amod)

def differences(a,b):
    d = difflib.Differ()
    diff = d.compare(a, b)
    cont=0
    result=[]
    for d in diff:
        d.strip()
        accio=d[0]
        lletra=d[-1]
        if accio=="":
            cadena=lletra
            result.append(cadena)
        elif accio=="+":
            cadena=lletra+"\u0332"
            result.append(cadena)
        elif accio=="-":
            cadena=lletra+"\u0336"
            result.append(cadena)
        else:
            result.append(lletra)
    amod="".join(result)
    return(amod)

def differencesExcel(a,b,blue,red,bold):
    
    d = difflib.Differ()
    diff = d.compare(a, b)
    cont=0
    string_parts=[]
    for d in diff:
        d.strip()
        accio=d[0]
        lletra=d[-1]
        if accio=="":
            string_parts.append(lletra)
        elif accio=="+":
            string_parts.append(blue)
            string_parts.append(lletra)
            
        elif accio=="-":
            string_parts.append(red)
            string_parts.append(lletra)
            
        else:
            string_parts.append(lletra)
    
       
    return(string_parts)

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

def open_source():
    source_file = askopenfilename(initialdir = filepathin, filetypes = (("All files", "*"),("txt files","*.txt")))
    F_frame_E_Source.delete(0,END)
    F_frame_E_Source.insert(0,source_file)
    return
    
def open_detailed():
    detailed_file = asksaveasfilename(initialdir = filepathin, filetypes = (("All files", "*"),("txt files","*.txt"),("xlsx files","*.xlsx")))
    F_frame_E_Detailed.delete(0,END)
    F_frame_E_Detailed.insert(0,detailed_file)
    return

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
    main_window.clipboard_clear()  
    main_window.clipboard_append(results_frame_text.get("1.0",END))
    
def calculate():
    results_frame_text.delete('1.0', END)

    
    rfilename=F_frame_E_Ref.get()
    hfilename=F_frame_E_Hyp.get()
    
    rfile=codecs.open(rfilename,"r",encoding="utf-8")
    hfile=codecs.open(hfilename,"r",encoding="utf-8")
    
    sys.path.append(os.getcwd())
    selectedtokenizer=combo_tokenizersF.get()
    if not selectedtokenizer.endswith(".py"): selectedtokenizer=selectedtokenizer+".py"
    spec = importlib.util.spec_from_file_location('', selectedtokenizer)
    tokenizermod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizermod)
    tokenizer=tokenizermod.Tokenizer() 
    
    references=[]
    references_tok=[]    
    contref=0
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
        contref+=1
        
    hypothesis=[]
    hypothesis_tok=[]
    
    conthyp=0
    for linia in hfile:
        linia=linia.rstrip()
        tokens=tokenizer.tokenize(linia).split(" ")
        hypothesis.append(linia)
        hypothesis_tok.append(tokens)
        conthyp+=1
        
    if not contref==conthyp:
        messagebox.showerror("Error", "Reference and hypothesis files should have the same number of lines.")
    
    
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
            cadena="%EdDist: "+str(round(EditDistance,reddist))
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
    print("-------------------------------------------")        
    rfile.close()
    hfile.close()
    
    if 'selected' in F_frame_detailed.state():
    
        sourcesegments=[]
        try:
            sourcef=codecs.open(F_frame_E_Source.get(),"r",encoding="utf-8")
            for linia in sourcef:
                linia=linia.rstrip()
                sourcesegments.append(linia)
        except:
            for i in range(0,contref):
                sourcesegments.append("")
    
        excelfile=F_frame_E_Detailed.get()
        textfile=F_frame_E_Detailed.get()
        if not excelfile.endswith(".xlsx"): excelfile=excelfile.replace(".txt","")+".xlsx"
        if not textfile.endswith(".txt"): textfile=textfile.replace(".xlsx","")+".txt"
        
        workbook = xlsxwriter.Workbook(excelfile)
        sheetAll = workbook.add_worksheet("All")
        sheetDetailed = workbook.add_worksheet("Detailed")
        sheetDetailed.set_column(1, 4, 30)
        bold = workbook.add_format({'bold': True})
        red = workbook.add_format({'color': 'red'})
        red.set_font_strikeout()
        blue = workbook.add_format({'color': 'blue'})
        text_wrap = workbook.add_format({'text_wrap': 1, 'valign': 'top'})
        
        sortida=codecs.open(textfile,"w",encoding="utf-8")
        
        sheetAll.write(0,0,"SEGMENTS")
        sheetAll.write(0,1,contref)
        sheetAll.write(1,0,"BLEU")
        if doBLEU:
            sheetAll.write(1,1,round(BLEU,rbleu))
        sheetAll.write(2,0,"NIST")
        if doNIST:
            sheetAll.write(2,1,round(NIST,rnist))
        sheetAll.write(3,0,"WER")
        if doWER:
            sheetAll.write(3,1,round(WER,rwer))
        sheetAll.write(4,0,"%EdDist")
        if doED:
            sheetAll.write(4,1,round(EditDistance,reddist))
        sheetAll.write(5,0,"TER")
        if doTER:
            sheetAll.write(5,1,round(TERcorpus,rter))
        
        cadenasortida=[]
        cadenasortida.append("IDENT.")
        cadenasortida.append("Source.")
        cadenasortida.append("Reference")
        cadenasortida.append("Hyphotesis")
        cadenasortida.append("DIFF.")
        
        sheetDetailed.write(0, 0, "IDENT.", bold)
        sheetDetailed.write(0, 1, "Source", bold)
        sheetDetailed.write(0, 2, "Reference", bold)
        sheetDetailed.write(0, 3, "Hyphotesis", bold)
        sheetDetailed.write(0, 4, "DIFF.", bold)
        column=5
        if doBLEU: 
            sheetDetailed.write(0, column, "BLEU", bold)
            cadenasortida.append("BLEU")
            columnBLEU=column
            column+=1
        if doNIST: 
            sheetDetailed.write(0, column, "NIST", bold)
            cadenasortida.append("NIST")
            columnNIST=column
            column+=1
        if doWER: 
            sheetDetailed.write(0, column, "WER", bold)
            cadenasortida.append("WER")
            columnWER=column
            column+=1
        if doTER: 
            sheetDetailed.write(0, column, "TER", bold)
            cadenasortida.append("TER")
            columnTER=column
            column+=1
        if doED: 
            sheetDetailed.write(0, column, "EditDistance", bold)
            cadenasortida.append("EditDistance")
            columnED=column
            column+=1
            
        sortida.write("\t".join(cadenasortida)+"\n")
        for i in range(0,len(hypothesis)):
            cadenasortida=[]
            sheetDetailed.write(i+1, 0, i+1)
            cadenasortida.append(str(i+1))
            sheetDetailed.write(i+1, 1, sourcesegments[i], text_wrap)
            cadenasortida.append(sourcesegments[i].replace("\t"," "))
            sheetDetailed.write(i+1, 3, hypothesis[i], text_wrap)
                       
            rtok=[references_tok[i]]
            htok=[hypothesis_tok[i]]
            selectedreference=references[i][0]
            #NOTE: if more than one reference, the one used in the excel file is the first one
            
            sheetDetailed.write(i+1, 2, selectedreference, text_wrap)
            cadenasortida.append(selectedreference.replace("\t"," ")) 
            cadenasortida.append(sourcesegments[i].replace("\t"," ")) 
            dE=differencesExcel(selectedreference,hypothesis[i],red,blue,bold)
            dEtext=differences(selectedreference.replace("\t"," "),hypothesis[i].replace("\t"," "))
            cadenasortida.append(dEtext)
            sheetDetailed.write_rich_string(i+1, 4, *dE, text_wrap)
            
            if doBLEU:
                try:
                    BLEU=corpus_bleu(rtok,htok)
                    sheetDetailed.write(i+1, columnBLEU, round(BLEU,rbleu))
                    cadenasortida.append(str(round(BLEU,rbleu)))
                except:
                    cadenasortida.append("")
                    print("ERROR: unable to calculate detailed BLEU.")
            if doNIST:
                try:
                    NIST=corpus_nist(rtok,htok)
                    sheetDetailed.write(i+1, columnNIST, round(NIST,rnist))
                    cadenasortida.append(str(round(NIST,rnist)))
                except:
                    cadenasortida.append("")
                    print("ERROR: unable to calculate detailed NIST:")
            if doWER:
                try:
                    WER=wer_corpus(rtok,htok)
                    sheetDetailed.write(i+1, columnWER, round(WER,rwer))
                    cadenasortida.append(str(round(WER,rwer)))
                except:
                    cadenasortida.append("")
                    print("ERROR: unable to calculate detailed WER.")
                
            if doED:
                try:
                    edtotal=0
                    chartotal=0
                    for i2 in range(0,len(htok)):
                        editmin=100000000
                        chartotal+=len(htok[i2])
                        for h in rtok[i2]:
                            ed=edit_distance(htok[i2],h)
                            if ed<editmin:
                                editmin=ed
                        edtotal+=editmin
                    
                    EditDistance=100*(edtotal/chartotal)
                    sheetDetailed.write(i+1, columnED, round(EditDistance,reddist))
                    cadenasortida.append(str(round(EditDistance,reddist)))
                except:
                    cadenasortida.append("")
                    print("ERROR: unable to calculate detailed Ed")
                
            if doTER:
                try:
                    TERcorpus=ter_corpus(rtok,htok)
                    sheetDetailed.write(i+1, columnTER, round(TERcorpus,rter))
                    cadenasortida.append(str(round(TERcorpus,rter)))                  
                except:
                    cadenasortida.append("")
                    print("ERROR: unable to calculate detailed TER",sys.exc_info())      
            
            sortida.write("\t".join(cadenasortida)+"\n")
            
        workbook.close()
    
    

try:
    stream = open('config.yaml', 'r',encoding="utf-8")
    config=yaml.load(stream,Loader=Loader)
    filepathin=config['Filepath']['path_in']
    tokenizerlist=config['Tokenizers']['list']
    defaultokenizer=config['Tokenizers']['default']   
    #############
    #ROUND VALUES:
    doBLEU=config['Measure']['bleu']
    doNIST=config['Measure']['nist']
    doWER=config['Measure']['wer']
    doED=config['Measure']['ed']
    doTER=config['Measure']['ter']
    rbleu=int(config['Round']['rbleu'])
    rnist=int(config['Round']['rnist'])
    rwer=int(config['Round']['rwer'])
    reddist=int(config['Round']['red'])
    rter=int(config['Round']['rter'])
    #############
except:
    filepathin="."
    tokenizerlist=["MTUOC_tokenizer_arg", "MTUOC_tokenizer_ast", "MTUOC_tokenizer_cat", "MTUOC_tokenizer_deu", "MTUOC_tokenizer_eng", "MTUOC_tokenizer_fra", "MTUOC_tokenizer_gal", "MTUOC_tokenizer_gen", "MTUOC_tokenizer_ita", "MTUOC_tokenizer_por", "MTUOC_tokenizer_rus", "MTUOC_tokenizer_spa", "MTUOC_tokenizer_zho_jieba", "MTUOC_tokenizer_zho_pseudo"]
    defaultokenizer="MTUOC_tokenizer_gen"  
    #############
    #ROUND VALUES:
    rbleu=3
    rnist=3
    rwer=3
    reddist=3
    rter=3


main_window = Tk()
main_window.title("MTUOC Eval v. 2.0")
s = ttk.Style()
themes=s.theme_names()
if "winnative" in themes:
    s.theme_use("winnative")
else:
    s.theme_use("default")
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

F_frame_detailed=ttk.Checkbutton(F_frame, text="Detailed results")
F_frame_detailed.grid(sticky="W",row=4,column=0)
F_frame_detailed.state(['!alternate'])

F_frame_B_Detailed=Button(F_frame, text = str("Select Detailed"), command=open_detailed,width=15)
F_frame_B_Detailed.grid(row=5,column=0)
F_frame_E_Detailed = Entry(F_frame,  width=50)
F_frame_E_Detailed.grid(row=5,column=1)

F_frame_B_Source=Button(F_frame, text = str("Select Source"), command=open_source,width=15)
F_frame_B_Source.grid(row=6,column=0)
F_frame_E_Source = Entry(F_frame,  width=50)
F_frame_E_Source.grid(row=6,column=1)

F_frame__B_Go=Button(F_frame, text = str("Calculate"), command=calculate,width=15)
F_frame__B_Go.grid(sticky="E",row=7,column=1)

results_frame = Frame(notebook)
results_frame_text=scrolledtext.ScrolledText(results_frame,width = 50, height = 10 )
results_frame_text.grid(row=0,column=0)
results_frame_B=Button(results_frame, text = str("Copy to clipboard"), command=copy_results,width=20)
results_frame_B.grid(row=1,column=0)



notebook.add(F_frame, text="Files", padding=30)
notebook.add(results_frame, text="Results", padding=30)

notebook.pack()
notebook.pack_propagate(0) #Don't allow the widgets inside to determine the frame's width / height
notebook.pack(fill=BOTH, expand=1)



main_window.mainloop()

