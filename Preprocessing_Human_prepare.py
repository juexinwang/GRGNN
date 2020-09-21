#Generate goldstandard files and expression
# inputfile='data/Human_B3expression_100.txt'
inputfile='data/expression_trans_all_use.txt'
inputfileBench='data/trrust_rawdata.human.tsv'
outputfile='data/Human_expression.txt'
outputfileBench='data/Human_goldstandard.txt'
outputfileBench1='data/Human_goldstandard1.txt'
outputfileBench2='data/Human_goldstandard2.txt'
outputfileBench3='data/Human_goldstandard3.txt'
outputfileBench23='data/Human_goldstandard23.txt'
outputfileBench13='data/Human_goldstandard13.txt'
outputfileBench12='data/Human_goldstandard12.txt'
# total mapped is 745
tfnum = 745
cvNum = 249

linecount = 0
tfDictName = {}
outList = []
outList1 = []
outList2 = []
outList3 = []
count = 0
with open(inputfileBench) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        words = line.split()
        tfDictName[words[0]]=''
        outList.append(words[0]+'\t'+words[1]+'\t1\n')
        # cross validation
        if count%cvNum ==0:
            outList1.append(words[0]+'\t'+words[1]+'\t1\n')
        elif count%cvNum ==1:
            outList2.append(words[0]+'\t'+words[1]+'\t1\n')
        else:
            outList3.append(words[0]+'\t'+words[1]+'\t1\n')
        linecount += 1    
    f.close()

print('Total TF number is '+str(len(tfDictName)))

# generate output Benchmark
with open(outputfileBench,'w') as fw:
    fw.writelines(outList)

# cross validation
with open(outputfileBench1,'w') as fw:
    fw.writelines(outList1)

with open(outputfileBench2,'w') as fw:
    fw.writelines(outList2)

with open(outputfileBench3,'w') as fw:
    fw.writelines(outList3)

outList23 = outList2+outList3
with open(outputfileBench23,'w') as fw:
    fw.writelines(outList23)

outList13 = outList1+outList3
with open(outputfileBench13,'w') as fw:
    fw.writelines(outList13)

outList12 = outList1+outList2
with open(outputfileBench12,'w') as fw:
    fw.writelines(outList12)


linecount = -1
tfDict = {}
notfDict = {}
with open(inputfile) as f:
    lines = f.readlines()
    for line in lines:        
        line = line.strip()
        words = line.split()
        if linecount == -1:
            count = 0
            tfcount = 0
            notfcount = tfnum
            for word in words:
                if word in tfDictName:
                    tfDict[count] = tfcount
                    tfcount+=1
                else:
                    notfDict[count] = notfcount
                    notfcount+=1
                count+=1
            # print(tfcount)
            # print(tfDict)
        linecount += 1    
    f.close()

outList = []
with open(inputfile) as f:
    lines = f.readlines()
    for line in lines:        
        line = line.strip()
        words = line.split()
        count = 0
        tfstr = ''
        notfstr = ''
        for word in words:
            if count in tfDict:
                tfstr = tfstr + word + '\t'
            else:
                notfstr = notfstr + word + '\t'
            count+=1
        outList.append(tfstr+notfstr.strip()+'\n')   
    f.close()

with open(outputfile,'w') as fw:
    fw.writelines(outList)