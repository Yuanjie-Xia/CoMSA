install.packages("ScottKnottESD")
library(ScottKnottESD)
library(readr)
X264_model <- read_csv("54_lrzip_model.csv")
X264_model = X264_model[,2:14]
X264_model = X264_model[,1:9]
X264_model<-subset(X264_model,select = (-12))
sk = sk_esd(X264_model)
plot(sk)
sk$ord
