library("GenomicDistributions")
library(tidyverse)

query = rtracklayer::import('/home/yyasumizu/nanoporemeth/results/230228/ref_1000bp_2000regions_0.4diff.bed')

# # First, calculate the distribution:
# x = calcChromBinsRef(query, "hg38")

gp = calcPartitionsRef(query, "hg38")
write.csv(gp, file = '/home/yyasumizu/nanoporemeth/results/230228/gp.csv')
plotPartitions(gp)
ggsave('/home/yyasumizu/nanoporemeth/results/230228/gp.pdf', width=4, height=4)

ep = calcExpectedPartitionsRef(query, "hg38")
plotExpectedPartitions(ep)
ggsave('/home/yyasumizu/nanoporemeth/results/230228/ep.pdf', width=4, height=4)
write.csv(ep, file = '/home/yyasumizu/nanoporemeth/results/230228/ep.csv')
