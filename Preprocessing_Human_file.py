#Generate goldstandard files and expression
inputfile='data/Human_B3expression_100.txt'
inputfileBench='data/trrust_rawdata.human.tsv'
outputfile='data/Human_expression.txt'
outputfileBench='data/Human_goldstandard.txt'

linecount = 0
tfDictName = {}
outList = []
with open(inputfileBench) as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        words = line.split()
        tfDictName[words[0]]=''
        outList.append(words[0]+'\t'+words[1]+'\t1\n')
        linecount += 1    
    f.close()

print('Total TF number is '+str(len(tfDictName)))

# generate output Benchmark
with open(outputfileBench,'w') as fw:
    fw.writelines(outList)


# total mapped is 747
tfnum = 747
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