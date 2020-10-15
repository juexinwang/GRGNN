# https://www.bioconductor.org/packages/release/bioc/manuals/minet/man/minet.pdf

library(minet)
m =read.csv("/home/wangjue/biodata/DREAM5_network_inference_challenge/Network3/input_data/expression_data.tsv",sep='\t')
r =read.csv("/home/wangjue/myprojects/public/GRGNN/data/minet3.csv",sep='\t',row.names=1)
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
amr=ceiling(mrlist/4511)
bmr=mrlist%%4511

arlist=order(ar, decreasing=TRUE)[1:1000000]
aar=ceiling(arlist/4511)
bar=arlist%%4511

clrlist=order(clr, decreasing=TRUE)[1:1000000]
aclr=ceiling(clrlist/4511)
bclr=clrlist%%4511

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

write.table(mrl, file="data/mr3.csv", row.names = F, col.names = F)
write.table(mrl_true, file="data/mr_true3.csv", row.names = F, col.names = F)
write.table(arl, file="data/ar3.csv", row.names = F, col.names = F)
write.table(arl_true, file="data/ar_true3.csv", row.names = F, col.names = F)
write.table(clrl, file="data/clr3.csv", row.names = F, col.names = F)
write.table(clrl_true, file="data/clr3_true3.csv", row.names = F, col.names = F)