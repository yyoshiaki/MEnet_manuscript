library(EpiDISH)
# data(centEpiFibIC.m)
# data(DummyBeta.m)

# out.l <- epidish(beta.m = DummyBeta.m, ref.m = centEpiFibIC.m, method = "RPC") 

setwd('/home/yyasumizu/nanoporemeth')

x_ref <- read.csv('./results/230531_compare_algorithms_test_indsamples_min3/ref_nnls.csv', row.names=1)
x_test <- read.csv('./results/230531_compare_algorithms_test_indsamples_min3/x_test_nnls.csv', row.names=1)
x_test_mix <- read.csv('./results/230531_compare_algorithms_test_indsamples_min3/x_test_mix_nnls.csv', row.names=1)

print('test RPC')
system.time(
out.l <- epidish(beta.m = as.matrix(x_test), ref.m = as.matrix(x_ref), method = "RPC")
)
write.csv(as.data.frame(out.l$estF), file = './results/230531_compare_algorithms_test_indsamples_min3/pred_test_EpiDISH.RPC.csv')

print('test CBS')
system.time(
out.l <- epidish(beta.m = as.matrix(x_test), ref.m = as.matrix(x_ref), method = "CBS")
)
write.csv(as.data.frame(out.l$estF), file = './results/230531_compare_algorithms_test_indsamples_min3/pred_test_EpiDISH.CBS.csv')


print('test CP')
system.time(
out.l <- epidish(beta.m = as.matrix(x_test), ref.m = as.matrix(x_ref), method = "CP")
)
write.csv(as.data.frame(out.l$estF), file = './results/230531_compare_algorithms_test_indsamples_min3/pred_test_EpiDISH.CP.csv')

print('test mix RPC')
system.time(
out.l <- epidish(beta.m = as.matrix(x_test_mix), ref.m = as.matrix(x_ref), method = "RPC")
)
write.csv(as.data.frame(out.l$estF), file = './results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.RPC.csv')

print('test mix CBS')
system.time(
out.l <- epidish(beta.m = as.matrix(x_test_mix), ref.m = as.matrix(x_ref), method = "CBS")
)
write.csv(as.data.frame(out.l$estF), file = './results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.CBS.csv')


print('test mix PC')
system.time(
out.l <- epidish(beta.m = as.matrix(x_test_mix), ref.m = as.matrix(x_ref), method = "CP")
)
write.csv(as.data.frame(out.l$estF), file = './results/230531_compare_algorithms_test_indsamples_min3/pred_test_mix_EpiDISH.CP.csv')
