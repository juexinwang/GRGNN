count = 0
with open('/home/wangjue/biodata/DREAM5_network_inference_challenge/Network3/gold standard/Network3_GoldStandard.tsv') as f:
    lines = f.readlines()
    for line in lines:
        words = line.split()
        end1 = int(words[0][1:])
        end2 = int(words[1][1:])
        if end1 < end2:
            count = count +1
        else:
            print(line)
    f.close()
print(count)
