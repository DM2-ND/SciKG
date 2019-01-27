# n: synonym or acronym, [C1,as,C2]
# h: hypernym, [C1,contain,C2]
# f: fact tuple, [C1|{C1:A1}, predicate, C2|{C2:A2}]
# c: condition tuple, [C1|{C1:A1}, predicate, C2|{C2:A2}]
# stmt: statement, "f ... f c ... c"

import os
import argparse

workdir = './'
# workdir = '/afs/crc.nd.edu/group/dmsquare/vol1/data/PaperTitleAbstract/data/'

parser = argparse.ArgumentParser(description='statement extraction from labeled data')

# Model parameters.
parser.add_argument('--train', action='store_true',
                    help='extraction for trainning')
parser.add_argument('--eval', action='store_true',
                    help='extraction for valid')
parser.add_argument('--IOB', action='store_true',
                    help='extraction for valid')
parser.add_argument('--filestmt', type=str, default='./stmts-train.tsv')

args = parser.parse_args()

###########################################################
# find fact/condition tuples and (conditional) statements #
###########################################################
def findStmt(fileoutput,filedata,docid_filelabel):

    # load concepts, attributes, predicates, and stmts

    docid2struc = {}
    fact_tuple_num = 0
    condition_tuple_num = 0
    for [docid,filelabel] in docid_filelabel:

        nid2tuple = {}
        hid2tuple = {}
        fid2tuple = {}
        cid2tuple = {}
        sid2stmts = {}
        print docid
        fr = open(filelabel,'r')
        for line in fr:
            text = line.strip()
            print text
            if text == '': continue
            head = text[0]
            if head == '#':
                continue
            elif head == 'n':
                pos = text.find('[')
                arr = text[pos+1:-1].split(',')
                assert(len(arr) == 3 and arr[1] == 'as')
                _id = text[:pos-1]
                assert(not _id in nid2tuple)
                nid2tuple[_id] = [['C',arr[0]],arr[1],['C',arr[2]]]
            elif head == 'h':
                pos = text.find('[')
                arr = text[pos+1:-1].split(',')
                assert(len(arr) == 3 and arr[1] == 'contain')
                _id = text[:pos-1]
                assert(not _id in hid2tuple)        
                hid2tuple[_id] = [['C',arr[0]],arr[1],['C',arr[2]]]
            elif head == 'f':
                pos = text.find('[')
                arr = text[pos+1:-1].split(',')
                assert(len(arr) == 3)
                _tuple = [[],'',[]]
                for i in [0,2]:
                    if ':' in arr[i]:
                        _arr = arr[i][1:-1].split(':')
                        assert(len(_arr) == 2)
                        _tuple[i] = ['A',_arr[0],_arr[1]]
                    else:
                        if arr[i] == 'NIL':
                            _tuple[i] = ['N',arr[i]]
                        else:
                            _tuple[i] = ['C',arr[i]]
                _tuple[1] = arr[1]
                _id = text[:pos-1]
                try:
                    assert(not _id in fid2tuple)  
                except:
                    print text
                    sys.exit(1)
                fid2tuple[_id] = _tuple
            elif head == 'c':
                pos = text.find('[')
                arr = text[pos+1:-1].split(',')
                assert(len(arr) == 3)
                _tuple = [[],'',[]]        
                for i in [0,2]:
                    if ':' in arr[i]:
                        _arr = arr[i][1:-1].split(':')
                        assert(len(_arr) == 2)
                        _tuple[i] = ['A',_arr[0],_arr[1]]
                    else:
                        if arr[i] == 'NIL':
                            _tuple[i] = ['N',arr[i]]
                        else:
                            _tuple[i] = ['C',arr[i]]
                _tuple[1] = arr[1]
                _id = text[:pos-1]
                assert(not _id in cid2tuple)
                cid2tuple[_id] = _tuple
            elif head == 's':
                if text[:4] == 'stmt':
                    arr = text.split(' ')
                    stmt = [[],[],'NIL']
                    assert(arr[1] == '=')
                    for i in range(2,len(arr)):
                        _id = arr[i]
                        if _id[0] == 'f':
                            assert(_id in fid2tuple)
                            stmt[0].append(_id)
                        elif _id[0] == 'c':
                            assert(_id in cid2tuple)
                            stmt[1].append(_id)
                        elif _id[0] == '(' and _id[-1] == ')':
                            stmt[2] = _id[1:-1]
                        else:
                            assert(False)
                    sid = int(arr[0][4:])
                    if not sid in sid2stmts:
                        sid2stmts[sid] = []
                    sid2stmts[sid].append(stmt)
                elif text[:4] == 's???':
                    continue
                else:
                    assert(False)
            else:
                assert(False)
        fr.close()

        fact_tuple_num += len(fid2tuple)
        condition_tuple_num += len(cid2tuple)
        docid2struc[docid] = [nid2tuple,hid2tuple,fid2tuple,cid2tuple,sid2stmts]

    '''
    for [docid,[nid2tuple,hid2tuple,fid2tuple,cid2tuple,sid2stmt]] in sorted(docid2struc.items(),key=lambda x:x[0]):
        print('===== '+docid+' - tuples =====')
        for [_id,_tuple] in sorted(nid2tuple.items(),key=lambda x:int(x[0][1:])):
            print(_id,_tuple)
        for [_id,_tuple] in sorted(hid2tuple.items(),key=lambda x:int(x[0][1:])):
            print(_id,_tuple)
        for [_id,_tuple] in sorted(fid2tuple.items(),key=lambda x:int(x[0][1:])):
            print(_id,_tuple)
        for [_id,_tuple] in sorted(cid2tuple.items(),key=lambda x:int(x[0][1:])):
            print(_id,_tuple)
        print('===== '+docid+' - stmts =====')
        for [sid,stmts] in sorted(sid2stmts.items(),key=lambda x:x[0]):
            for stmt in stmts:
                print(sid,stmt)
    '''

    nStmt = 0

    fw = open(fileoutput,'w')
    fr = open(filedata,'r')
    docidCurr,sid = '',1
    for line in fr: 
        arr = line.strip().split('\t')
        docid,text = arr[0],arr[1]
        if not docid in docid2struc: continue
        print docid
        if docid == docidCurr:
            sid += 1
        else:
            sid = 1
            docidCurr = docid
        nid2tuple,hid2tuple,fid2tuple,cid2tuple,sid2stmts = docid2struc[docid]

        if not sid in sid2stmts: 
            fw.write('===== '+docid+' stmt'+str(sid)+' =====\n')

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
            #print seqword,seqpostag,seqanno
            n = len(seqword)

            s = ''
            for i in range(n): s += '\t'+seqword[i]
            fw.write('WORD'+s+'\n')
            s = ''
            for i in range(n): s += '\t'+seqpostag[i]
            fw.write('POSTAG'+s+'\n')
            s = ''
            for i in range(n): s += '\t'+seqanno[i]
            fw.write('CAP'+s+'\n')
            s = ''
            for i in range(n): s += '\t'+'O'
            fw.write('f'+s+'\n')
            s = ''
            for i in range(n): s += '\t'+'O'
            fw.write('c'+s+'\n')

            nStmt += 1

            continue

        stmts = sid2stmts[sid]        

        for stmt in stmts:
            print sid

            if stmt[2] == 'NIL':
                fw.write('===== '+docid+' stmt'+str(sid)+' =====\n')       
            else:
                fw.write('===== '+docid+' stmt'+str(sid)+' ('+stmt[2]+') =====\n')

            # indexing

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
            #print seqword,seqpostag,seqanno

            phrase2symbols = {}
            id_tuple = []
            for fid in stmt[0]: id_tuple.append([fid,fid2tuple[fid]])
            for cid in stmt[1]: id_tuple.append([cid,cid2tuple[cid]])
            for [hid,_tuple] in hid2tuple.items(): id_tuple.append([hid,_tuple])
            for [_id,_tuple] in id_tuple:
                print _tuple
                if _tuple[0][0] == 'C':
                    phrase, off = _tuple[0][1].split('#')
                    phrase = phrase.replace('@',',')
                    print phrase, seqword[int(off)].lower()
                    assert seqword[int(off)].lower() == phrase.split('_')[0] or (phrase.split('_')[0]=='NIL')
                    if not phrase in phrase2symbols:
                        phrase2symbols[phrase] = []
                    phrase2symbols[phrase].append([_id,'1C',off])
                elif _tuple[0][0] == 'A':
                    phrase, off = _tuple[0][1].split('#')
                    phrase = phrase.replace('@',',')
                    print phrase, seqword[int(off)].lower()
                    assert seqword[int(off)].lower() == phrase.split('_')[0] or (phrase.split('_')[0]=='NIL')
                    if not phrase == 'NIL':
                        if not phrase in phrase2symbols:
                            phrase2symbols[phrase] = []
                        phrase2symbols[phrase].append([_id,'1C',off])
                    phrase, off = _tuple[0][2].split('#')
                    phrase = phrase.replace('@',',')
                    print phrase, seqword[int(off)].lower()
                    assert seqword[int(off)].lower() == phrase.split('_')[0]
                    if not phrase in phrase2symbols:
                        phrase2symbols[phrase] = []
                    phrase2symbols[phrase].append([_id,'1A',off])
                phrase, off = _tuple[1].split('#')
                phrase = phrase.replace('@',',')
                print phrase, seqword[int(off)].lower()
                assert (seqword[int(off)].lower() == phrase.split('_')[0]) or (phrase.split('_')[0]=='NIL')
                if not phrase == 'contain':                    
                    if not phrase in phrase2symbols:
                        phrase2symbols[phrase] = []
                    phrase2symbols[phrase].append([_id,'2P',off])
                if _tuple[2][0] == 'C':
                    phrase, off = _tuple[2][1].split('#')
                    phrase = phrase.replace('@',',')
                    print phrase, seqword[int(off)].lower()
                    assert seqword[int(off)].lower() == phrase.split('_')[0] or (phrase.split('_')[0]=='NIL')
                    if not phrase in phrase2symbols:
                        phrase2symbols[phrase] = []
                    phrase2symbols[phrase].append([_id,'3C',off])
                elif _tuple[2][0] == 'A':
                    phrase, off = _tuple[2][1].split('#')
                    phrase = phrase.replace('@',',')
                    print phrase, seqword[int(off)].lower()
                    assert seqword[int(off)].lower() == phrase.split('_')[0] or (phrase.split('_')[0]=='NIL')
                    if not phrase == 'NIL':
                        if not phrase in phrase2symbols:
                            phrase2symbols[phrase] = []
                        phrase2symbols[phrase].append([_id,'3C',off])
                    phrase, off = _tuple[2][2].split('#')
                    phrase = phrase.replace('@',',')
                    print phrase, seqword[int(off)].lower()
                    assert seqword[int(off)].lower() == phrase.split('_')[0]
                    if not phrase in phrase2symbols:
                        phrase2symbols[phrase] = []
                    phrase2symbols[phrase].append([_id,'3A',off])

            index,nindex = [{}],1
            for [phrase,symbols] in phrase2symbols.items():
                words = phrase.split('_')
                n = len(words)
                if n > nindex:
                    for i in range(nindex,n):
                        index.append({})
                    nindex = n
                temp = index[n-1]
                if n > 1:
                    for i in range(0,n-1):
                        word = words[i]
                        if not word in temp:
                            temp[word] = {}
                        temp = temp[word]
                    word = words[n-1]
                else:
                    word = words[0]
                temp[word] = symbols 


            n = len(seqword)

            tid2seqoutput = {}
            for [tid,_] in id_tuple: tid2seqoutput[tid] = ['O' for i in range(n)]

            _seqword = [word.lower() for word in seqword]
            tid2nlabel = {}
            i = 0
            while i < n:
                iffound = False
                for j in range(min(nindex,n-i),0,-1):
                    temp = index[j-1]
                    k = 0
                    while k < j and i+k < n:
                        tempword = _seqword[i+k]
                        #print 'tempword:', tempword
                        #print temp
                        if not tempword in temp: break
                        temp = temp[tempword]
                        k += 1
                    if k == j:
                        symbols = temp
                        for [tid,label,off] in symbols:
                            if not tid in tid2nlabel:
                                tid2nlabel[tid] = 0
                            tid2nlabel[tid] += 1
                        for _i in range(i,i+k):
                            if _i == i:
                                for [tid,label,off] in symbols:
                                    if i == int(off):
                                        tid2seqoutput[tid][_i] = 'B-'+tid[0]+label
                            else:
                                for [tid,label,off] in symbols:
                                    if i == int(off):
                                        tid2seqoutput[tid][_i] = 'I-'+tid[0]+label
                        i += k
                        iffound = True
                        break
                if iffound: continue
                i += 1
            # print tid2seqoutput
            # output
            nStmt += 1

            s = ''
            for i in range(n): s += '\t'+seqword[i]
            fw.write('WORD'+s+'\n')
            s = ''
            for i in range(n): s += '\t'+seqpostag[i]
            fw.write('POSTAG'+s+'\n')
            s = ''
            for i in range(n): s += '\t'+seqanno[i]
            fw.write('CAP'+s+'\n')
            for [tid,seqoutput] in sorted(tid2seqoutput.items(),key=lambda x:-ord(x[0][0])*999+int(x[0][1:])):
                s = ''
                for i in range(n):
                    s += '\t'+seqoutput[i]
                if tid[0] == 'h' and (not tid in tid2nlabel or tid2nlabel[tid] < 2): continue
                fw.write(tid+s+'\n')

    fr.close()

    fw.write('#'+str(nStmt)+'\t'+str(fact_tuple_num)+'\t'+str(condition_tuple_num)+'\n')

    fw.close()



def genPKL(filepkl,filestmt,filedata):
    return


def genIOB(diriob,filestmt,filedata):
   
    docid2data = {}
    fr = open(filestmt,'r')
    for line in fr:
        if line.startswith('#'): break
        if line.startswith('='):
            arr = line.strip().split(' ')
            docid = arr[1]
            if not docid in docid2data:
                docid2data[docid] = []
        else:
            arr = line.strip().split('\t')
            n = len(arr)
            if arr[0] == 'WORD':
                seqword = []
                for i in range(1,n):
                    seqword.append(arr[i])
            elif arr[0] == 'POSTAG':
                seqpostag = [] 
                for i in range(1,n):
                    seqpostag.append(arr[i])
            elif arr[0] == 'CAP':
                seqcap = []
                for i in range(1,n):
                    seqcap.append(arr[i])
            else:
                seqoutput = []
                for i in range(1,n):
                    seqoutput.append(arr[i])
                docid2data[docid].append([seqword,seqpostag,seqcap,seqoutput])
    fr.close()

    fw1 = open(diriob+'stmts-demo-x0X-train.iob','w')
    fw2 = open(diriob+'stmts-demo-x0X-testa.iob','w')
    fw3 = open(diriob+'stmts-demo-x0X-testb.iob','w')
    fw1.write('-DOCSTART- -X- O O\n')
    fw2.write('-DOCSTART- -X- O O\n')
    fw3.write('-DOCSTART- -X- -X- O\n')    
    for [docid,data] in sorted(docid2data.items(),key=lambda x:x[0]):
        for seqword,seqpostag,seqcap,seqoutput in data:
            n = len(seqword)
            fw1.write('\n')
            fw2.write('\n')
            fw3.write('\n')            
            for i in range(n):
                s = seqword[i]+' '+seqpostag[i]+' '+seqcap[i]+' '+seqoutput[i]+'\n'
                fw1.write(s)
                fw2.write(s)
                fw3.write(s)                
    fw3.close()
    fw2.close()
    fw1.close()

    fw1 = open(diriob+'stmts-demo-0X-train.iob','w')
    fw2 = open(diriob+'stmts-demo-0X-testa.iob','w')
    fw3 = open(diriob+'stmts-demo-0X-testb.iob','w')
    fw1.write('-DOCSTART- -X- O O\n')
    fw2.write('-DOCSTART- -X- O O\n')
    fw3.write('-DOCSTART- -X- -X- O\n')    
    for [docid,data] in sorted(docid2data.items(),key=lambda x:x[0]):
        for seqword,seqpostag,seqcap,seqoutput in data:
            n = len(seqword)
            fw1.write('\n')
            fw2.write('\n')
            fw3.write('\n')            
            for i in range(n):
                if seqoutput[i].startswith('B-') or seqoutput[i].startswith('I-'):
                    s = seqword[i]+' '+seqpostag[i]+' '+seqcap[i]+' '+seqoutput[i][:2]+seqoutput[i][3:]+'\n'
                else:
                    s = seqword[i]+' '+seqpostag[i]+' '+seqcap[i]+' '+seqoutput[i]+'\n'
                fw1.write(s)
                fw2.write(s)
                fw3.write(s)                
    fw3.close()
    fw2.close()
    fw1.close()

    fw1 = open(diriob+'stmts-demo-X-train.iob','w')
    fw2 = open(diriob+'stmts-demo-X-testa.iob','w')
    fw3 = open(diriob+'stmts-demo-X-testb.iob','w')
    fw1.write('-DOCSTART- -X- O O\n')
    fw2.write('-DOCSTART- -X- O O\n')
    fw3.write('-DOCSTART- -X- -X- O\n')    
    for [docid,data] in sorted(docid2data.items(),key=lambda x:x[0]):
        for seqword,seqpostag,seqcap,seqoutput in data:
            n = len(seqword)
            fw1.write('\n')
            fw2.write('\n')
            fw3.write('\n')            
            for i in range(n):
                if seqoutput[i].startswith('B-') or seqoutput[i].startswith('I-'):
                    s = seqword[i]+' '+seqpostag[i]+' '+seqcap[i]+' '+seqoutput[i][:2]+seqoutput[i][4:]+'\n'
                else:
                    s = seqword[i]+' '+seqpostag[i]+' '+seqcap[i]+' '+seqoutput[i]+'\n'
                fw1.write(s)
                fw2.write(s)
                fw3.write(s)                
    fw3.close()
    fw2.close()
    fw1.close()


if __name__ == '__main__':

    filedata = './anno-coref-demo.txt'

    if args.train:
        docid_filelabel = []
        for filename in os.listdir("./label/train"):
            docid = filename.split('-')[-1].split('.txt')[0]
            print docid
            docid_filelabel.append([docid, "./label/train/"+filename])
        
        filestmt = '_stmts-train.tsv'

        findStmt(filestmt,filedata,docid_filelabel)

    if args.eval:
        docid_filelabel = []
        for filename in os.listdir("./label/eval"):
            docid = filename.split('-')[-1].split('.txt')[0]
            print docid
            docid_filelabel.append([docid, "./label/eval/"+filename])
        
        filestmt = '_stmts-eval.tsv'

        findStmt(filestmt,filedata,docid_filelabel)
