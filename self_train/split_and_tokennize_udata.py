# n: synonym or acronym, [C1,as,C2]
# h: hypernym, [C1,contain,C2]
# f: fact tuple, [C1|{C1:A1}, predicate, C2|{C2:A2}]
# c: condition tuple, [C1|{C1:A1}, predicate, C2|{C2:A2}]
# stmt: statement, "f ... f c ... c"

import os, sys
sys.path.append('../')
import argparse

from config import *

parser = argparse.ArgumentParser(description='PyTorch multi_input multi_output model')
# Model parameters.
parser.add_argument('--input', type=str, default=workdir+'/pubmed/in.txt')
parser.add_argument('--output', type=str, default=workdir+'/pubmed/out.txt')
args = parser.parse_args()

###############################
# tokenize the unlabeled data #
###############################
def split_tokenize_unlabeled_data(fileoutput,filedata,docid_filelabel,num_parts):

    nStmt = 0

    fw = open(fileoutput,'w')
    fr = open(filedata,'r')
    num_sentences = sum(1 for line in fr)
    print num_sentences
    part_size = num_sentences/num_parts
    index = 0
    fr = open(filedata,'r')
    docidCurr,sid = '',1
    doc_set = set()
    for line in fr: 
        arr = line.strip().split('\t')
        docid,text = arr[0],arr[1]
        if docid in docid_filelabel: continue
        if docid == docidCurr:
            sid += 1
        else:
            sid = 1
            docidCurr = docid
        doc_set.add(docid)

        seqword,seqpostag,seqanno = [],[],[]

        elems = text.split(' ')
        n = len(elems)
        for i in range(n):
            elem = elems[i]
            if elem.startswith('$C'):
                _arr = elem.split(':')
                phrase = _arr[1]
                arrphrase = phrase.split('_')
                arrpostag = _arr[2].split('_')
                _n = len(arrphrase)
                for j in range(_n):
                    seqword.append(arrphrase[j])
                    seqpostag.append(arrpostag[j])
                    if j == 0:
                        seqanno.append('B-C')
                    else:
                        seqanno.append('I-C')
            elif elem.startswith('$A'):
                _arr = elem.split(':')
                arrphrase = _arr[1].split('_')
                arrpostag = _arr[2].split('_')
                _n = len(arrphrase)
                for j in range(_n):
                    seqword.append(arrphrase[j])
                    seqpostag.append(arrpostag[j])
                    if j == 0:
                        seqanno.append('B-A')
                    else:
                        seqanno.append('I-A')
            elif elem.startswith('$P'):
                _arr = elem.split(':')
                arrphrase = _arr[1].split('_')
                arrpostag = _arr[2].split('_')
                _n = len(arrphrase)
                for j in range(_n):
                    seqword.append(arrphrase[j])
                    seqpostag.append(arrpostag[j])
                    if j == 0:
                        seqanno.append('B-P')
                    else:
                        seqanno.append('I-P')
            else:
                _arr = elem.split(':')
                seqword.append(_arr[0])
                seqpostag.append(_arr[1])
                seqanno.append('O')
        # print len(seqword),len(seqpostag),len(seqanno)
        assert len(seqword)==len(seqpostag)==len(seqanno)
        n = len(seqword)

        if index % part_size == 0:
            if index != 0:
                fpw.write('#'+str(part_size)+'\n')
                fpw.close()
            fpw = open(fileoutput.split('.tsv')[0]+'_part'+str(index/part_size)+'.tsv','w')

        fw.write('===== '+docid+' stmt'+str(sid)+' =====\n')
        fpw.write('===== '+docid+' stmt'+str(sid)+' =====\n')
        s = ''
        for i in range(n): s += '\t'+seqword[i]
        fw.write('WORD'+s+'\n')
        fpw.write('WORD'+s+'\n')
        s = ''
        for i in range(n): s += '\t'+seqpostag[i]
        fw.write('POSTAG'+s+'\n')
        fpw.write('POSTAG'+s+'\n')
        s = ''
        for i in range(n): s += '\t'+seqanno[i]
        fw.write('CAP'+s+'\n')
        fpw.write('CAP'+s+'\n')


        index += 1
        nStmt += 1

    print len(doc_set)
    fr.close()

    fpw.write('#'+str(index%part_size+1)+'\n')
    fpw.close()
    fw.write('#'+str(nStmt)+'\n')
    fw.close()

if __name__ == '__main__':

    filedata = WORKDIR+'/preprocessing/anno-coref-pubmed.t_lymphocyte.txt'
    # filedata = WORKDIR+'/preprocessing/anno-coref-demo.new.txt'

    docid_filelabel = []
    for filename in os.listdir(WORKDIR+"/label/train"):
        docid = filename.split('-')[-1].split('.txt')[0]
        docid_filelabel.append(docid)
    for filename in os.listdir(WORKDIR+"/label/eval"):
        docid = filename.split('-')[-1].split('.txt')[0]
        docid_filelabel.append(docid)

    print docid_filelabel
    
    filestmt = './udata/stmts-demo-unlabeled-pubmed.tsv'

    split_tokenize_unlabeled_data(args.output,args.input,docid_filelabel,1)

    # filepkl = 'stmts-demo.pkl.gz'

#    genPKL(filepkl,filestmt,filedata)

    # diriob = 'iob/'

    # genIOB(diriob,filestmt,filedata)


