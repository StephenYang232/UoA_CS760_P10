library(dplyr)
library(tidyr)
library(readr)
library(jsonlite)
library(xml2)
library(arrow)

reviews <- stream_in(file("yelp_academic_dataset_review.json"))

businesses <- stream_in(file("yelp_academic_dataset_business.json"))

reviews2 <- reviews |>
  # Take mean stars for duplicated bussinesses reviewed by each user,
  # Assume reviews with stars > 3 are positive
  group_by(user_id, business_id) |>
  summarise(sentiment = as.numeric(mean(stars) > 3)) |>
  # Select users with at least 2 positive revieww and 2 negative reviews
  filter((sum(sentiment == 1) >= 2) & (sum(sentiment == 0) >= 2)) |>
  # Join with business information
  ungroup() |>
  inner_join(businesses, by = "business_id") |>
  # Remove unnecessary asin column
  select(-business_id)

# reviews2 |> write_parquet("Yelp.parquet", compression = "BROTLI")
# reviews2 <- read_parquet("Yelp.parquet") |> as_tibble()

set.seed(760)
reviews3 <- reviews2[reviews2$user_id %in% sample(unique(reviews2$user_id), 20000, replace = FALSE), ]

# Generate prompts for HuggingFace
prompts <- reviews3 |>
  group_by(user_id) |>
  summarise(
    instruction = "Given the user's preference and unpreference, identify whether the user will like the target business by answering \"Yes.\" or \"No.\".",
    input = sprintf(
      '<x>User preference: "%s"\nUser Unpreference: "%s"\nWhether the user will like the target business "%s"?</x>',
      paste(name[sentiment == 1 & row_number() > 1], collapse = '", "'),
      paste(name[sentiment == 0 & row_number() > 1], collapse = '", "'),
      name[1]
    ) |> read_html() |> xml_text(), # nolint: pipe_continuation_linter.
    output = ifelse(sentiment[1] == 1, "Yes.", "No.")
  ) |>
  # Remove prompts that are too long for LLM models
  filter(nchar(input) < 1000)

positive <- which(prompts$output == "Yes.")
negative <- which(prompts$output == "No.")
min_length <- min(length(positive), length(negative))
indexes <- rbind(positive[1:min_length], negative[1:min_length]) |>
  as.vector() |>
  c(positive[-(1:min_length)], negative[-(1:min_length)])
prompts <- prompts[indexes, ]

stream_out(prompts[1:1024, -1], file("train.jsonl"))
stream_out(prompts[1025:2048, -1], file("valid.jsonl"))
stream_out(prompts[2049:3072, -1], file("test.jsonl"))

# Generate prompts for GPT
prompts2 <- prompts |>
  group_by(user_id) |>
  reframe(
    role = c("system", "user", "assistant"),
    content = c(instruction, input, output)
  ) |>
  nest(.by = user_id, .key = "messages") |>
  select(-user_id)

prompts2 <- prompts2[indexes, ]

stream_out(prompts2[1:64, ], file("yelp_gpt_train_64.jsonl"))
stream_out(prompts2[1:256, ], file("yelp_gpt_train_256.jsonl"))
stream_out(prompts2[1:1024, ], file("yelp_gpt_train_1024.jsonl"))
stream_out(prompts2[1025:2048, ], file("yelp_gpt_valid.jsonl"))
stream_out(prompts2[2049:3072, ], file("yelp_gpt_test.jsonl"))
