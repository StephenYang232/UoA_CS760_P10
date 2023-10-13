# 加载必要的库
library(dplyr)
library(data.table)

# 读取相似度矩阵和数据集
m1_similarity1 <- read.table("similarity_matrix.csv", header = FALSE)
df_pivot1 <- read.csv("data_frame.csv", header = TRUE)

m2_similarity <- read.table("similarity_matrix2.csv", header = FALSE)
df_pivot2 <- read.csv("data_frame2.csv", header = TRUE)

find_top_similar_users <- function(user_index) {
  user_similarity <- m1_similarity[user_index, ]
  sorted_similarity <- sort(user_similarity, decreasing = TRUE)
  top_similar_users <- names(sorted_similarity[1:8])
  return(top_similar_users)
}
get_recommendations <- function(user_index) {
  similar_users <- find_top_similar_users(user_index)
  recommendations <- character(8)
  
  for (i in 1:8) {
    user_b_index <- similar_users[i]
    user_b_recommendations <- df_pivot[user_b_index, ]
    
    # 找到用户b推荐倾向最高的产品
    max_recommendation <- names(user_b_recommendations[user_b_recommendations == max(user_b_recommendations)])
    
    if (length(max_recommendation) > 0) {
      # 避免向同一用户重复推荐相同的产品
      max_recommendation <- setdiff(max_recommendation, recommendations)
      
      if (length(max_recommendation) > 0) {
        recommendations[i] <- max_recommendation[1]
      } else {
        # 如果所有推荐产品都已经在recommendations中，随机选择一个产品
        available_products <- setdiff(names(df_pivot), recommendations)
        recommendations[i] <- sample(available_products, 1)
      }
    } else {
      # 如果用户b没有推荐倾向最高的产品，随机选择一个产品
      available_products <- setdiff(names(df_pivot), recommendations)
      recommendations[i] <- sample(available_products, 1)
    }
  }
  
  return(recommendations)
}

calculate_precision <- function(recommendations_df1, recommendations_df2) {
  # 获取用户列表（假定用户列表是在 "User" 列中）
  users <- recommendations_df1$User
  
  # 初始化一个向量来存储每个用户的相同产品数量
  precision <- numeric(length(users))
  
  for (i in 1:length(users)) {
    user <- users[i]
    recommendations1 <- unlist(recommendations_df1[recommendations_df1$User == user, -1])
    recommendations2 <- unlist(recommendations_df2[recommendations_df2$User == user, -1])
    
    # 计算相同产品数量
    common_recommendations <- intersect(recommendations1, recommendations2)
    precision[i] <- length(common_recommendations)
  }
  
  return(precision)
}


calculate_average_new_recommendations <- function(user_product_df, recommendations_df) {
  # 初始化一个向量来存储每行用户的新产品推荐数
  new_recommendations_count <- integer(nrow(recommendations_df))
  
  for (i in 1:nrow(recommendations_df)) {
    user <- recommendations_df[i, 1]
    recommended_products <- recommendations_df[i, -1]  # 去掉用户名列
    user_products <- user_product_df[user, ]
    
    # 计算用户获得的新产品推荐数
    new_recommendations <- sum(recommended_products %in% names(user_products)[user_products == 0])
    
    new_recommendations_count[i] <- new_recommendations
  }
  
  # 计算平均的新产品推荐数
  average_new_recommendations <- mean(new_recommendations_count)
  return(average_new_recommendations)
}


users1 <- rownames(df_pivot1)
recommendations1 <- t(sapply(users1, get_recommendations))
recommendations_df1 <- data.frame(User = users1, recommendations1)

users2 <- rownames(df_pivot2)
recommendations2 <- t(sapply(users2, get_recommendations))
recommendations_df2 <- data.frame(User = users2, recommendations2)


# 使用函数计算平均产品推荐相同数
average_precision <- mean(calculate_precision(recommendations_df1, recommendations_df2))/8

# 打印结果
cat("平均产品推荐相同数（精确度）：", average_precision, "\n")

# 传入用户-产品数据框和推荐列表数据框
average_new_products1 <- calculate_average_new_recommendations(df_pivot1,recommendations_df1)

# 输出平均的新产品数量
cat("平均的新产品数量:", average_new_products1, "\n")

