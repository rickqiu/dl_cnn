options(scipen=999)
require(tensorflow)
require(keras)

df <- readRDS("input/test.rds")

normalize <- function(x) (x/sqrt(sum(x^2)))

band1 <- sapply(df[,2], function(x) unlist(x))
band2 <- sapply(df[,3], function(x) unlist(x))

li_train <- list()
 
for (i in 1:ncol(band1)){
    x <- normalize (matrix(band1[,i], nrow = 75))
    y <- normalize (matrix(band2[,i], nrow = 75))
    z <- array(c( x , y ) , dim = c( 75, 75, 2))
    li_train[[i]] <- z
}

X <- sapply(li_train, identity, simplify = "array")
X <- aperm(X, c(4,1,2,3))

pred <- predict_proba(model, X, batch_size = 32, verbose = 0)
 
colnames(pred) <- c("p0", "p1")
pred <- data.frame(pred)
submission <- data.frame(df$id, pred$p1)
colnames(submission) <- c("id", "is_iceberg")

write.csv(submission, file = "output/submission.csv", na = "", row.names = FALSE, quote = FALSE)