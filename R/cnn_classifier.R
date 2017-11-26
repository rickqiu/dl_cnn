options(scipen=999)
require(tensorflow)
require(keras)

# Data importation
df <- readRDS(file="input/train.rds")

#df$inc_angle <- as.numeric(df$inc_angle)
#df$inc_angle[is.na(df$inc_angle)] <- 0

# Feature scaling
normalize <- function(x) (x/sqrt(sum(x^2)))

band1 <- sapply(df[,2], function(x) unlist(x))
band2 <- sapply(df[,3], function(x) unlist(x))

li_train <- list()
 
for (i in 1:ncol(band1)){
    x <- normalize (matrix(band1[,i], nrow = 75))
    y <- normalize (matrix(band2[,i], nrow = 75))
    z <- array(c( x , y ) , dim = c(75,75,2))
    li_train[[i]] <- z
}

X <- sapply(li_train, identity, simplify = "array")
X <- aperm(X, c(4,1,2,3))

# Hot encoding
y <- df$is_iceberg

v0 <- y
v1 <- y

v0[v0 == 1] <- 2
v0[v0 == 0] <- 1
v0[v0 == 2] <- 0

y <- matrix(c(v0,v1),nrow=length(v0))

colnames(y) <- c("class0", "class1")

# Keras - Tensorflow backend (LeNet7)
n_classes <- 2
batch_size = 32
epochs = 100
kernel_size = 3 
pool_size = 2
conv_depth = 32
hidden_size = 128
drop_prob_1 = 0.24
drop_prob_2 = 0.5
l2_lambda = 0.00015
    
model <- keras_model_sequential()

model %>%

    layer_conv_2d(filters = conv_depth, kernel_size = c(kernel_size,kernel_size), 
                  strides = c(1,1), padding = 'same', 
                  kernel_initializer='he_uniform',
                  data_format = "channels_last", dtype = "float32", input_shape = dim(X[1,,,])) %>%
    layer_activity_regularization(l1=0.0,l2=l2_lambda) %>%
    layer_activation('relu') %>%

    layer_conv_2d(filters = conv_depth, kernel_size = c(kernel_size,kernel_size), 
                  strides = c(1,1), padding='same',
                  kernel_initializer='he_uniform') %>%
    layer_activity_regularization(l1=0.0,l2=l2_lambda) %>%
    layer_activation('relu') %>%
    layer_max_pooling_2d(pool_size = c(pool_size,pool_size)) %>%
    layer_dropout(drop_prob_1) %>% 
    
    layer_conv_2d(filters = conv_depth, kernel_size = c(kernel_size, kernel_size),
                  kernel_initializer='he_uniform') %>%
    layer_activity_regularization(l1=0.0,l2=l2_lambda) %>%
    layer_activation('relu') %>%
   
    layer_conv_2d(filters = conv_depth, kernel_size = c(kernel_size,kernel_size),
                  kernel_initializer='he_uniform') %>%
    layer_activity_regularization(l1=0.0,l2=l2_lambda) %>%
    layer_activation('relu') %>%
    layer_max_pooling_2d(pool_size = c(pool_size,pool_size)) %>%
    layer_dropout(drop_prob_1) %>%
    
    layer_flatten() %>%
    layer_dense(hidden_size,  kernel_initializer='he_uniform') %>%
    layer_activity_regularization(l1=0.0,l2=l2_lambda) %>%
    layer_activation('relu') %>%
    layer_dropout(drop_prob_2) %>%
    
    layer_dense(n_classes,  kernel_initializer='glorot_uniform') %>%
    layer_activity_regularization(l1=0.0,l2=l2_lambda) %>%
    layer_activation('softmax')

model %>% compile(optimizer = optimizer_rmsprop(), loss = 'categorical_crossentropy',metrics = c('accuracy'))

history <- model %>% fit(x = X, y = y, bach_size = batch_size, epochs = epochs, validation_split = .2)
history