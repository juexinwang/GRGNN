# https://www.bioconductor.org/packages/release/bioc/manuals/minet/man/minet.pdf
# Use minet to infer minet,aracne,clr

# Usage:
# Rscript Preprocessing_minet.r 3
# Rscript Preprocessing_minet.r 4
# test if there is one argument: if not, return an error
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 
geneNum=4511
if(args[1]=='3'){
    geneNum=4511
}else if (args[1]=='4'){
    geneNum=5950
}

library(minet)
exp_file=paste("/home/wangjue/biodata/DREAM5_network_inference_challenge/Network",args[1],"/input_data/expression_data.tsv",sep='')
m =read.csv(exp_file,sep='\t')
true_file=paste("/home/wangjue/myprojects/public/GRGNN/data/True",args[1],".csv",sep='')
r =read.csv(true_file,sep='\t',row.names=1)
r =data.matrix(r)

mr <- minet( m, method="mrnet", estimator="spearman" )
ar <- minet( m, method="aracne", estimator="spearman" )
clr<- minet( m, method="clr", estimator="spearman" )
# # Validation
# mr.tbl <- validate(mr,r)
# ar.tbl <- validate(ar,r)
# clr.tbl<- validate(clr,r)
# # Plot PR-Curves
# max(fscores(mr.tbl))
# dev <- show.pr(mr.tbl, col="green", type="b")
# dev <- show.pr(ar.tbl, device=dev, col="blue", type="b")
# show.pr(clr.tbl, device=dev, col="red",type="b")
# auc.pr(clr.tbl)


mrlist=order(mr, decreasing=TRUE)[1:1000000]
amr=ceiling(mrlist/geneNum)
bmr=mrlist%%geneNum

arlist=order(ar, decreasing=TRUE)[1:1000000]
aar=ceiling(arlist/geneNum)
bar=arlist%%geneNum

clrlist=order(clr, decreasing=TRUE)[1:1000000]
aclr=ceiling(clrlist/geneNum)
bclr=clrlist%%geneNum

mrl<-c()
mrl_true<-c()
arl<-c()
arl_true<-c()
clrl<-c()
clrl_true<-c()
i=1
while(i<=1000000) {
    mrl<-c(mrl,mr[amr[i],bmr[i]])
    mrl_true<-c(mrl_true,r[amr[i],bmr[i]])
    arl<-c(arl,ar[aar[i],bar[i]])
    arl_true<-c(arl_true,r[aar[i],bar[i]])
    clrl<-c(clrl,clr[aclr[i],bclr[i]])
    clrl_true<-c(clrl_true,r[aclr[i],bclr[i]])
    i=i+1
}

write.table(mrl, file=paste("data/mr",args[1],".csv",sep=''), row.names = F, col.names = F)
write.table(mrl_true, file=paste("data/mr_true",args[1],".csv",sep=''), row.names = F, col.names = F)
write.table(arl, file=paste("data/ar",args[1],".csv",sep=''), row.names = F, col.names = F)
write.table(arl_true, file=paste("data/ar_true",args[1],".csv",sep=''), row.names = F, col.names = F)
write.table(clrl, file=paste("data/clr",args[1],".csv",sep=''), row.names = F, col.names = F)
write.table(clrl_true, file=paste("data/clr_true",args[1],".csv",sep=''), row.names = F, col.names = F)