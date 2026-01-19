rm(list = ls())
cat("\014")
graphics.off()
set.seed(65288)
library(tidyverse)
library(caret)
library(corrplot)
library(e1071)
library(car)
library(factoextra)
library(gridExtra)
library(clue) 
library(ggsci)
library(dplyr)
library(ggplot2)
library(lmtest)
df <- read.csv("Store_CA.csv")
head(df, 10)
str(df)
sapply(df[, sapply(df, is.numeric)], sd)
desc(df)
#Missing Values
missing_values <- colSums(is.na(df))
print("Missing Values per Column:")
print(missing_values)

#Target variable distribution
ggplot(df, aes(x = MonthlySalesRevenue)) +
  geom_histogram(fill = "steelblue", color = "white", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Monthly Sales Revenue")



## Grid histogram
df_hist_data <- df %>%
  select(-starts_with("StoreLocation"), -starts_with("StoreCategory"), -starts_with("Monthly")) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

ggplot(df_hist_data, aes(x = Value)) +
  geom_histogram(fill = "steelblue", color = "white", bins = 30) +
  facet_wrap(~Variable, scales = "free") + 
  theme_minimal() +
  theme(strip.text = element_text(face = "bold")) + 
  labs(title = "",
       y = "Count",
       x = "Value")

###### EDA-------------

#Outlier Boxpluts
df_boxplot <- df %>%
  select(-starts_with("StoreLocation"), -starts_with("StoreCategory")) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value")

# 2. Grid Boxplot
ggplot(df_boxplot, aes(x = Variable, y = Value)) +
  geom_boxplot(fill = "orange", outlier.colour = "red", outlier.shape = 16) +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  labs(title = "", y = "Values", x = "") +
  theme(axis.text.x = element_blank()) 


df_numeric_only <- df %>% 
  select(-starts_with("StoreLocation"), 
         -starts_with("StoreCategory"))

cor_matrix_clean <- cor(df_numeric_only)

corrplot(cor_matrix_clean,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.cex = 0.8,    
         addCoef.col = "black", 
         number.cex = 0.7,
         title = "Correlation Matrix (Continuous Variables Only)",
         mar = c(0,0,2,0))




# filter  categorical vars
numeric_vars_check <- df %>% 
  select(-starts_with("StoreLocation"), 
         -starts_with("StoreCategory"))

calculate_stats <- function(x) {
  return(c(
    Skewness = skewness(x, na.rm = TRUE),
    Kurtosis = kurtosis(x, na.rm = TRUE)
  ))
}

normality_table <- data.frame(t(sapply(numeric_vars_check, calculate_stats)))
normality_table <- round(normality_table, 3)
print("Skewness and Kurtosis Values:")
print(normality_table)


### Data Transformation
#One hot encoding for categorical variables
df$StoreLocation <- as.factor(df$StoreLocation)
df$StoreCategory <- as.factor(df$StoreCategory)

#Linear and RF one hot encoding n-1
dummies_linear <- dummyVars(" ~ .", data = df, fullRank = TRUE)
df_linear <- data.frame(predict(dummies_linear, newdata = df))
df_rf = df_linear
# kmeans n encoding
dummies_rf <- dummyVars(" ~ .", data = df, fullRank = FALSE)
df_kmeans <- data.frame(predict(dummies_rf, newdata = df))


#linear train-test split
trainIndex_lin <- createDataPartition(df_linear$MonthlySalesRevenue, p = .8, list = FALSE)
train_linear   <- df_linear[trainIndex_lin, ]
test_linear    <- df_linear[-trainIndex_lin, ]
train_linear_scaled <- train_linear
test_linear_scaled  <- test_linear
#exclude y and dummy vars
exclude_cols <- c(
  "MonthlySalesRevenue",
  "StoreLocation.Palo.Alto",     
  "StoreLocation.San.Francisco",   
  "StoreLocation.Sacramento",     
  "StoreCategory.Grocery",       
  "StoreCategory.Electronics"     

)

# columns to sclae
vars_to_scale <- setdiff(names(train_linear), exclude_cols)

# creating z standart rule on train set
scaler_rule <- preProcess(train_linear[, vars_to_scale], method = c("center", "scale"))

#apply to all data
train_linear_scaled[, vars_to_scale] <- predict(scaler_rule, train_linear[, vars_to_scale])
test_linear_scaled[, vars_to_scale]  <- predict(scaler_rule, test_linear[, vars_to_scale])

#min-max scale for kmeans
trainIndex_km <- createDataPartition(df_kmeans$MonthlySalesRevenue, p = .8, list = FALSE)
kmeans_train <- df_kmeans[trainIndex_km, ]
kmeans_test  <- df_kmeans[-trainIndex_km, ]
kmeans_train_scaled <- kmeans_train
kmeans_test_scaled  <- kmeans_test

# exclude y
exclude_cols_km <- c("MonthlySalesRevenue") 
vars_to_scale_km <- setdiff(names(kmeans_train), exclude_cols_km)
scaler_rule_km <- preProcess(kmeans_train[, vars_to_scale_km], method = "range")
#apply rule
kmeans_train_scaled[, vars_to_scale_km] <- predict(scaler_rule_km, kmeans_train[, vars_to_scale_km])
kmeans_test_scaled[, vars_to_scale_km]  <- predict(scaler_rule_km, kmeans_test[, vars_to_scale_km])


#VIF CONTROL FOR LINEAR REG
vif_model <- lm(MonthlySalesRevenue ~ ., data = train_linear_scaled)
vif_values <- vif(vif_model)
print(sort(vif_values, decreasing = TRUE))
#dropping promotionscount
train_linear_scaled <- subset(train_linear_scaled, select = -c(PromotionsCount))
test_linear_scaled <- subset(test_linear_scaled, select = -c(PromotionsCount))
kmeans_train_scaled <- subset(kmeans_train_scaled, select = -c(PromotionsCount))
kmeans_test_scaled <- subset(kmeans_test_scaled, select = -c(PromotionsCount))



#vif2 after drop
vif_model <- lm(MonthlySalesRevenue ~ ., data = train_linear_scaled)
vif_values <- vif(vif_model)
print(sort(vif_values, decreasing = TRUE))


### K MEANS CLUSTERING MODELING
train_data_for_clustering <- subset(kmeans_train_scaled, select = -c(MonthlySalesRevenue))
#Elbow
elbow_graph <- fviz_nbclust(train_data_for_clustering, kmeans, method = "wss") +
  labs(title = "Elbow Method (Train Set)")
#Silhouette
silhoutte_graph <- fviz_nbclust(train_data_for_clustering, kmeans, method = "silhouette") +
  labs(title = "Silhouette Method (Train Set)")
silhoutte_graph$layers[[length(silhoutte_graph$layers)]] <- NULL
grid.arrange(elbow_graph, silhoutte_graph, ncol = 2)

#######K MEANS MODELING
train_data_final <- subset(kmeans_train_scaled, select = -c(MonthlySalesRevenue))

#modeling
final_kmeans_model <- kmeans(train_data_final, centers = 3, nstart = 100)

#clusters to train set
kmeans_train_scaled$Cluster <- as.factor(final_kmeans_model$cluster)

#predict test
assign_kmeans_cluster <- function(km, newdata) {
  centers <- as.matrix(km$centers)
  need_cols <- colnames(centers)
  
  # filter needed columns
  Xdf <- as.data.frame(newdata)
  missing_cols <- setdiff(need_cols, names(Xdf))
  if (length(missing_cols) > 0) {
    stop("problems", paste(missing_cols, collapse = ", "))
  }
  Xdf <- Xdf[, need_cols, drop = FALSE]
  
  #transform to numerc
  Xdf_num <- data.frame(lapply(Xdf, function(v) {
    if (is.factor(v)) v <- as.character(v)
    if (is.character(v)) v <- trimws(v)
    as.numeric(v)
  }))
  
  #if na or broken data
  if (anyNA(Xdf_num)) {
    bad_cols <- names(Xdf_num)[colSums(is.na(Xdf_num)) > 0]
    stop(
      "problems: ",
      paste(bad_cols, collapse = ", "),
      "\nproblems."
    )
  }
  
  X <- as.matrix(Xdf_num)
  
  apply(X, 1, function(x) {
    d <- rowSums((centers - matrix(x, nrow(centers), ncol(centers), byrow = TRUE))^2)
    which.min(d)
  })
}

test_data_final <- subset(kmeans_test_scaled, select = -c(MonthlySalesRevenue))
test_tahminleri <- assign_kmeans_cluster(final_kmeans_model, test_data_final)
kmeans_test_scaled$Cluster <- factor(test_tahminleri,
                                     levels = sort(unique(final_kmeans_model$cluster)))


print("Train")
print(head(kmeans_train_scaled[, c("Cluster", "MonthlySalesRevenue")]))

print("Test")
print(head(kmeans_test_scaled[, c("Cluster", "MonthlySalesRevenue")]))

print("number of data in clusters")
print(table(kmeans_train_scaled$Cluster))


fviz_cluster(final_kmeans_model, data = train_data_final,
             palette = "jco",         
             geom = "point",          
             ellipse.type = "convex", 
             ggtheme = theme_bw(),
             main = "K-Means Clusters")


# cluster wo scaling
kmeans_train$Cluster <- as.factor(final_kmeans_model$cluster)

cluster_profil_tablosu <- kmeans_train %>%
  group_by(Cluster) %>%
  summarise_if(is.numeric, mean, na.rm = TRUE)

print(cluster_profil_tablosu)

#####
#####
#second attempt for k means w/o any dummy
original_train_data <- df_kmeans[trainIndex_km, ]
original_test_data  <- df_kmeans[-trainIndex_km, ]

exclude_cols_final <- c("MonthlySalesRevenue", 
                        "StoreCategory.Clothing", "StoreCategory.Electronics", "StoreCategory.Grocery",
                        "StoreLocation.Los.Angeles", "StoreLocation.San.Francisco", 
                        "StoreLocation.Sacramento", "StoreLocation.Palo.Alto")

train_numeric_only <- kmeans_train_scaled[, !names(kmeans_train_scaled) %in% exclude_cols_final]
test_numeric_only  <- kmeans_test_scaled[, !names(kmeans_test_scaled) %in% exclude_cols_final]

#k=3 modelling
final_numeric_model <- kmeans(train_numeric_only, centers = 3, nstart = 100)

# cluster labels
original_train_data$Cluster <- as.factor(final_numeric_model$cluster)
kmeans_train_scaled$Cluster <- as.factor(final_numeric_model$cluster)

#predict test
test_pred <- assign_kmeans_cluster(final_numeric_model, test_numeric_only)


kmeans_test_scaled$Cluster <- factor(test_pred,
                                     levels = sort(unique(final_numeric_model$cluster)))

final_numeric_summary <- original_train_data %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE)) %>%
  select(-starts_with("StoreLocation"), -starts_with("StoreCategory"))

print(final_numeric_summary)
train_numeric_only$Cluster <- NULL
fviz_cluster(final_numeric_model, data = train_numeric_only,
             palette = "jco",         
             geom = "point",          
             ellipse.type = "convex", 
             ggtheme = theme_bw(),
             main = "K-Means: Performance Based")


###linear regression -------------------
train_linear_scaled$Cluster <- as.factor(final_numeric_model$cluster)

#clusters to test set
test_cluster_tahminleri <- assign_kmeans_cluster(final_numeric_model, test_numeric_only)
test_linear_scaled$Cluster <- factor(test_cluster_tahminleri,
                                     levels = levels(train_linear_scaled$Cluster))

final_linear_model <- lm(MonthlySalesRevenue ~ ., data = train_linear_scaled)
predictions <- predict(final_linear_model, newdata = test_linear_scaled)

#performance metrics
model_performance <- postResample(predictions, test_linear_scaled$MonthlySalesRevenue)

print("model performance")
print(model_performance)
summary(final_linear_model)

#qq plot and breush pagan
par(mfrow = c(2, 2)) 
plot(final_linear_model)
bp_test <- bptest(final_linear_model)
print(bp_test)



#####RANDOM FOREST
# add cluster to train set
train_rf_final <- df_rf[trainIndex_lin, ]
train_rf_final$Cluster <- as.factor(final_numeric_model$cluster)
train_rf_final$PromotionsCount <- NULL

#add cluster to test set
test_rf_final <- df_rf[-trainIndex_lin, ]
test_rf_final$Cluster <- as.factor(test_cluster_tahminleri)
test_rf_final$PromotionsCount <- NULL
#training
set.seed(65288)
rf_model <- train(MonthlySalesRevenue ~ ., 
                  data = train_rf_final, 
                  method = "rf", 
                  importance = TRUE, 
                  trControl = trainControl(method = "cv", number = 5))

print(rf_model)


#predict on test
rf_predictions <- predict(rf_model, newdata = test_rf_final)

#metrics of rf
rf_performance <- postResample(rf_predictions, test_rf_final$MonthlySalesRevenue)
print(rf_performance)

varImp(rf_model)
importance <- varImp(rf_model)
importance_df <- data.frame(
  Variable = row.names(importance$importance),
  Overall = importance$importance$Overall
)

ggplot(importance_df, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Random Forest: Variable Importance",
    subtitle = "Sales Performance Drivers",
    x = "Variables",
    y = "Importance (Overall Score)"
  ) +
  geom_text(aes(label = round(Overall, 2)), hjust = -0.1, size = 3.5) 