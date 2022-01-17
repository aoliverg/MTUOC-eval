#!/usr/bin/python3
#    MTUOC-eval-lite
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

import xlsxwriter

#IMPORTS FOR YAML
import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
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
    parser.add_argument('--outfile', action='store', default=False, dest='outfile',help='The Excel and tabbed text file with the segment values.')
    
    parser.add_argument('--source', action='store', default=False, dest='source',help='The file with the source segments (only needed to create the output file).')
    
    try:
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
    
    args = parser.parse_args()

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

    
    
    rfile=codecs.open(args.refs,"r",encoding="utf-8")
    hfile=codecs.open(args.hyp,"r",encoding="utf-8")
    
    sys.path.append(os.getcwd())
    
    #tokenizer=importlib.import_module(args.tokenizer.replace(".py",""))
    if not args.tokenizer.endswith(".py"): args.tokenizer=args.tokenizer+".py"
    spec = importlib.util.spec_from_file_location('', args.tokenizer)
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
        print("ERROR: number of references and hypotheses not equal.")
        sys.exit()
        
    
        
 
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
            print("%EdDist:  ",round(EditDistance,reddist))
        except:
            print("ERROR: unable to calculate Ed")
        
    if doTER:
        try:
            TERcorpus=ter_corpus(references_tok,hypothesis_tok)
            print("TER:      ",round(TERcorpus,rter))
        except:
            print("ERROR: unable to calculate TER",sys.exc_info())

    if args.outfile:
    
        sourcesegments=[]
        if args.source:
            sourcef=codecs.open(args.source,"r",encoding="utf-8")
            for linia in sourcef:
                linia=linia.rstrip()
                sourcesegments.append(linia)
        else:
            for i in range(0,contref):
                sourcesegments.append("")
    
        excelfile=args.outfile
        textfile=args.outfile
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