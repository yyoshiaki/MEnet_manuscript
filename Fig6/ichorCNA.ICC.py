import os
import sys
import subprocess
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

dir_base = '/home/yyasumizu/nanoporemeth/data/hWGBS_ICC/hICC_bam'
list_files = glob.glob(dir_base+'/hICC_*.chr1-22XY.sorted.bam')
list_sample = [x.split('/')[-1].split('.chr1-22XY.sorted.bam')[0] for x in list_files]

for sample in tqdm(list_sample):
    print("# ", sample)
    cmd = f'''readCounter --window 1000000 --quality 20 \
--chromosome "chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX,chrY" \
{dir_base}/{sample}.chr1-22XY.sorted.bam > {dir_base}/{sample}.chr1-22XY.sorted.readCounter.1Mb.wig'''
    print(cmd)
    subprocess.run(cmd, shell=True)

    cmd = f'''cat {dir_base}/{sample}.chr1-22XY.sorted.readCounter.1Mb.wig | sed "s/chrom=chr/chrom=/g" > {dir_base}/{sample}.1-22XY.sorted.readCounter.1Mb.wig'''
    print(cmd)
    subprocess.run(cmd, shell=True)

    os.makedirs(f'{dir_base}/ichorCNA/{sample}', exist_ok=True)

    cmd = f'''
    Rscript /home/yyasumizu/Programs/ichorCNA/scripts/runIchorCNA.R --id {sample} \
  --WIG {dir_base}/{sample}.1-22XY.sorted.readCounter.1Mb.wig --ploidy "c(2,3)" --normal "c(0.5,0.6,0.7,0.8,0.9)" --maxCN 5 \
  --gcWig /home/yyasumizu/Programs/ichorCNA/inst/extdata/gc_hg38_1000kb.wig \
  --mapWig /home/yyasumizu/Programs/ichorCNA/inst/extdata/map_hg38_1000kb.wig \
  --centromere /home/yyasumizu/Programs/ichorCNA/inst/extdata/GRCh38.GCA_000001405.2_centromere_acen.txt \
  --normalPanel /home/yyasumizu/Programs/ichorCNA/inst/extdata/HD_ULP_PoN_hg38_1Mb_median_normAutosome_median.rds \
  --includeHOMD False --chrs "c(1:22, \\"X\\")" --chrTrain "c(1:22)" \
  --estimateNormal True --estimatePloidy True --estimateScPrevalence True \
  --scStates "c(1,3)" --txnE 0.9999 --txnStrength 10000 --outDir {dir_base}/ichorCNA/
    '''
    print(cmd)
    subprocess.run(cmd, shell=True)
