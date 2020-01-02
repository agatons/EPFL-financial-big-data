library('parallel')
library(data.table)

expand.grid.unique <- function(x, y, include.equals=FALSE){
  x <- unique(x)
  y <- unique(y)

  g <- function(i){
    z <- setdiff(y, x[seq_len(i-include.equals)])
    if(length(z)) cbind(x[i], z, deparse.level=0)
  }
  do.call(rbind, lapply(seq_along(x), g))
}

maxLikelihood <- function(c,n){
  return(ifelse(n>1,log(n/c)+(n-1)*log((n*n-n)/(n*n-c)),0))
}
maxLikelihoodList <- function(cs,ns){
  Lc=mclapply(names(cs),function(x) ifelse(ns[[x]]>1,log(ns[[x]]/cs[[x]])+(ns[[x]]-1)*log((ns[[x]]*ns[[x]]-ns[[x]])/(ns[[x]]*ns[[x]]-cs[[x]])),0))
  names(Lc)=names(cs)
  return(Lc)
}

findMaxImprovingPair <- function(C,cs,ns,i_s){ ##i_s is the set of all elements belonging to cluster s
  N=length(i_s)
  Lc.new=vector("list",N)
  Lc.old=maxLikelihoodList(cs,ns)
  names.cs=names(cs)
  max.row=vector("list",N)
  max.impr=-10000000
  pair.max.improv=c()
  for(i in head(names.cs,-1)){
    names.cs.j=names.cs[(which(names.cs==i)+1):length(names.cs)]
    for(j in names.cs.j){
      ns.new=ns[[i]]+ns[[j]]
      i_s.new=c(i_s[[i]],i_s[[j]])
      cs.new=sum(C[i_s.new,i_s.new])
      maxLikeliHood.new=maxLikelihood(cs.new,ns.new)
      improvement=maxLikeliHood.new-Lc.old[[i]]-Lc.old[[j]]
      #      print(paste(i,j,improvement,max.impr))
      if(length(improvement)==0){
        browser()
      }
      if(improvement>max.impr){
        max.impr=improvement
        pair.max.improv=c(i,j)
        Lc.max.impr=maxLikeliHood.new
        #        print(paste("------->> ",i,j,improvement,Lc.max.impr))
      }
    }
  }
  print(pair.max.improv)
  #  print(max.impr)
  #  print(Lc.max.impr)
  #  print("*************")
  return(list(pair=pair.max.improv,Lc.new=Lc.max.impr,Lc.old=lapply(pair.max.improv,function(x) Lc.old[[x]])))
}


aggregateClusters <- function(C,onlyLogLikImprovingMerges=FALSE){

  N=nrow(C)
  cs=as.list(rep(1,N))					# Sum of C by cluster
  s_i=as.list(1:N)              # Cluster of a given element. Start with one cluster per element s_i=1.
  ns=as.list(rep(1,N))					# Number of elements in each cluster
  i_s=as.list(1:N)              # Elements of a given cluster
  Lc=as.list(rep(0,N))                   #

  names.lists=as.character(1:N)
  names(Lc)=names(cs)=names(ns)=names(s_i)=names(i_s)=names.lists

  allPairs=expand.grid.unique(1:N,1:N)  # a matrix with 2 columns, listing all possible pairs
  allPairs=as.data.table(allPairs)
  setnames(allPairs,c("i","j"))
  allPairs[,improvement:={ns.new=ns[[i]]+ns[[j]]
  i_s.new=c(i_s[[i]],i_s[[j]])  # merge elements
  cs.new=sum(C[i_s.new,i_s.new])
  maxLikeliHood.new=maxLikelihood(cs.new,ns.new)
  maxLikeliHood.new-Lc[[i]]-Lc[[j]]
  },by="i,j"]

  clusters=list()
  for(it in 1:(N-1)){   # hierarchical merging
    bestPairIndex=allPairs[,which.max(improvement)]
    bestPair=as.character(allPairs[bestPairIndex,.(i,j)])
 #   print(bestPair)
    ic=bestPair[1]
    jc=bestPair[2]
    i=as.integer(ic)
    j=as.integer(jc)

    Lc.old=Lc[[ic]]+Lc[[jc]]
    Lc.old.max=max(Lc[[ic]],Lc[[jc]])
    Lc.new=allPairs[,max(improvement)]+Lc.old

 #   browser()
    #    print(paste("  ",Lc.new,"<>",sum(Lc.old),"=",Lc.old[1],"+",Lc.old[2],sep=""))
    if(Lc.new < Lc.old.max){
#      print("Lc.new<=max(Lc.old), exiting")
      break
    }
    if(Lc.new < Lc.old){
      if(onlyLogLikImprovingMerges){
#        print("no more log-likelihood improving merges found")
        break
      }else{
#        print(" HALF CLUSTER  Lc.new>max(Lc.old)")
      }
    }
    Lc[[ic]]=Lc.new
    Lc[[jc]]=NULL

    s_i[unlist(s_i)==j]=i

    i_s[[ic]]=c(i_s[[ic]],i_s[[jc]])
    i_s[[jc]]=NULL

    ns[[ic]]=ns[[ic]]+ns[[jc]]
    ns[[jc]]=NULL

    cs[[ic]]=sum(C[i_s[[ic]],i_s[[jc]]])  #sums C over the elements of cluster1
    cs[[jc]]=NULL

    allPairs=allPairs[j!=as.integer(jc)]
    allPairs=allPairs[i!=as.integer(jc)]

    allPairs[i==as.integer(ic),improvement:={
      ic=as.character(i)
      jc=as.character(j)
      ns.new=ns[[ic]]+ns[[jc]]
      i_s.new=c(i_s[[ic]],i_s[[jc]])  #  merge elements
      cs.new=sum(C[i_s.new,i_s.new])
      maxLikeliHood.new=maxLikelihood(cs.new,ns.new)
      maxLikeliHood.new-Lc[[ic]]-Lc[[jc]]
    },by="i,j"]

    #   print(cs.vec)
    #   print(ns.vec)
    #   print(sqrt((cs.vec-ns.vec)/(ns.vec^2-ns.vec)))
    clusters[[i]]=list(Lc=maxLikelihoodList(cs,ns),pair.merged=bestPair,s_i=s_i,i_s=i_s,cs=cs,ns=ns)
  }
  lastClusters=last(clusters)

  return(lastClusters)
}

