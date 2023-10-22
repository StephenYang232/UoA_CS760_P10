
library(dplyr)
library(data.table)

# Read the similarity matrix and dataset
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
    
    # Find the products that have the highest propensity to be recommended by users
    max_recommendation <- names(user_b_recommendations[user_b_recommendations == max(user_b_recommendations)])
    
    if (length(max_recommendation) > 0) {
      # Avoid recommending the same product to the same user over and over again
      max_recommendation <- setdiff(max_recommendation, recommendations)
      
      if (length(max_recommendation) > 0) {
        recommendations[i] <- max_recommendation[1]
      } else {
        # If all the recommended products are already in the recommendations, randomly select a product
        available_products <- setdiff(names(df_pivot), recommendations)
        recommendations[i] <- sample(available_products, 1)
      }
    } else {
      # If user b does not have the highest propensity to recommend a product, randomly select a product
      available_products <- setdiff(names(df_pivot), recommendations)
      recommendations[i] <- sample(available_products, 1)
    }
  }
  
  return(recommendations)
}

calculate_precision <- function(recommendations_df1, recommendations_df2) {
  # Get the list of users (assuming the list of users is in the "User" column)
  users <- recommendations_df1$User
  
  # Initialise a vector to store the same number of products per user
  precision <- numeric(length(users))
  
  for (i in 1:length(users)) {
    user <- users[i]
    recommendations1 <- unlist(recommendations_df1[recommendations_df1$User == user, -1])
    recommendations2 <- unlist(recommendations_df2[recommendations_df2$User == user, -1])
    
    # Calculate the number of identical products
    common_recommendations <- intersect(recommendations1, recommendations2)
    precision[i] <- length(common_recommendations)
  }
  
  return(precision)
}


calculate_average_new_recommendations <- function(user_product_df, recommendations_df) {
  # Initialise a vector to store the number of new product recommendations per row of users
  new_recommendations_count <- integer(nrow(recommendations_df))
  
  for (i in 1:nrow(recommendations_df)) {
    user <- recommendations_df[i, 1]
    recommended_products <- recommendations_df[i, -1]  # Remove user name list
    user_products <- user_product_df[user, ]
    
    # Calculate the number of new product referrals a user receives
    new_recommendations <- sum(recommended_products %in% names(user_products)[user_products == 0])
    
    new_recommendations_count[i] <- new_recommendations
  }
  
  # Calculate the average number of new product recommendations
  average_new_recommendations <- mean(new_recommendations_count)
  return(average_new_recommendations)
}


users1 <- rownames(df_pivot1)
recommendations1 <- t(sapply(users1, get_recommendations))
recommendations_df1 <- data.frame(User = users1, recommendations1)

users2 <- rownames(df_pivot2)
recommendations2 <- t(sapply(users2, get_recommendations))
recommendations_df2 <- data.frame(User = users2, recommendations2)


# Calculate the average number of product recommendations using the function
average_precision <- mean(calculate_precision(recommendations_df1, recommendations_df2))/8

# Print results
cat("Average number of identical product recommendations (accuracy):", average_precision, "\n")

# Incoming user-product data frame and recommendation list data frame
average_new_products1 <- calculate_average_new_recommendations(df_pivot1,recommendations_df1)

# Output average number of new products
cat("Average number of new products:", average_new_products1, "\n")

