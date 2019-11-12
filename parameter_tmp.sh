# for k in {1..10}
# do
#    python Main_inductive_ensembl.py --traindata-name dream4 --testdata-name dream3 --use-attribute
# done

for i in 0.1 0.5 1.0
do
for j in False True
do
for k in {1..3}
do
   python Main_inductive_ensembl.py --traindata-name dream3 --testdata-name dream4 --use-attribute --neighbors-ratio $i --nonezerolabel-flag $j
done
done
done

for i in 0.1 0.5 1.0
do
for j in False True
do
for k in {1..3}
do
   python Main_inductive_ensembl.py --traindata-name dream3 --testdata-name dream4 --use-attribute --neighbors-ratio $i --nonezerolabel-flag $j --use-embedding
done
done
done


# for i in {1}
# do
# for k in {1..10}
# do
#     python Main_inductive_ensembl.py --traindata-name dream4 --testdata-name dream3 --embedding-dim $i --use-attribute --use-embedding
# done
# done


# for i in {1,2}
# do
# for k in {1..5}
# do
#     python Main_inductive_ensembl.py --traindata-name dream3 --testdata-name dream4 --embedding-dim $i --use-attribute --use-embedding
# done
# done
