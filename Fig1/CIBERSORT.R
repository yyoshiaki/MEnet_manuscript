setwd('/home/yyasumizu/nanoporemeth/other_tools/Cibersort')
source('CIBERSORT.R')

perm = 0
QN = TRUE

test_time <- system.time(
results_test <- CIBERSORT('/home/yyasumizu/nanoporemeth/results/230531_compare_algorithms_test_indsamples_min3/ref_nnls.tsv',
    '/home/yyasumizu/nanoporemeth/results/230531_compare_algorithms_test_indsamples_min3/x_test_nnls.tsv',perm, QN)
)
write.csv(rey_test_mix.cpu()sults_test, file = '/home/yyasumizu/nanoporemeth/results/230531_compare_algorithms_test_indsamples_min3/pred_test_CIBERSORT.csv')

print(test_time)

test_time_mix <- system.time(
results_mix <- CIBERSORT('/home/yyasumizu/nanoporemeth/results/230531_compare_algorithms_test_indsamples_min3/ref_nnls.tsv',
    '/home/yyasumizu/nanoporemeth/results/230531_compare_algorithms_test_indsamples_min3/x_test_mix_nnls.tsv',perm, QN)
)

write.csv(results_mix, file = '/home/yyasumizu/nanoporemeth/results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_CIBERSORT.csv')


print(test_time_mix)
