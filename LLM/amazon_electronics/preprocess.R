library(dplyr)
library(tidyr)
library(readr)
library(jsonlite)
library(xml2)
library(arrow)

# con_out <- file("Electronics_5.jsonl", "wb")
# stream_in(gzfile("Electronics_5.json.gz", "rb"), handler = function(df) {
#   df |>
#     select(reviewerID, asin, overall) |>
#     drop_na() |>
#     stream_out(con_out, pagesize = 5000, verbose = FALSE)
# }, pagesize = 10000)
# close(con_out)
reviews <- stream_in(file("Electronics_5.jsonl"))

# con_out <- file("meta_Electronics.jsonl", "wb")
# stream_in(gzfile("meta_Electronics.json.gz", "rb"), handler = function(df) {
#   df |>
#     select(title, asin) |>
#     drop_na() |>
#     stream_out(con_out, pagesize = 5000, verbose = FALSE)
# }, pagesize = 10000)
# close(con_out)
metadata <- stream_in(file("meta_Electronics.jsonl"))

reviews2 <- reviews |>
  # Remove reviews with asins not appeared in the metadata
  filter(asin %in% metadata$asin) |>
  # Take mean overall for duplicated electronics reviewed by each user,
  # Assume reviews with overall > 3 are positive
  group_by(reviewerID, asin) |>
  summarise(sentiment = as.numeric(mean(overall) > 3)) |>
  # Select users with three more reviews with at least one positive review and one negative review
  group_by(reviewerID) |>
  filter((sum(sentiment == 1) >= 2) & (sum(sentiment == 0) >= 2)) |>
  # Join with book information
  ungroup() |>
  left_join(distinct(metadata, asin, .keep_all = TRUE), by = "asin") |>
  # Remove unnecessary asin column
  select(-asin)

# reviews2 |> write_parquet("AmazonElectronics.parquet", compression = "BROTLI")
# reviews2 <- read_parquet("AmazonElectronics.parquet") |> as_tibble()

set.seed(760)
reviews3 <- reviews2[reviews2$reviewerID %in% sample(unique(reviews2$reviewerID), 20000, replace = FALSE), ]

# Generate prompts for HuggingFace
prompts <- reviews3 |>
  group_by(reviewerID) |>
  summarise(
    instruction = "Given the user's preference and unpreference, identify whether the user will like the target electronics by answering \"Yes.\" or \"No.\".",
    input = sprintf(
      '<x>User preference: "%s"\nUser Unpreference: "%s"\nWhether the user will like the target electronics "%s"?</x>',
      paste(title[sentiment == 1 & row_number() > 1], collapse = '", "'),
      paste(title[sentiment == 0 & row_number() > 1], collapse = '", "'),
      title[1]
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
  group_by(reviewerID) |>
  reframe(
    role = c("system", "user", "assistant"),
    content = c(instruction, input, output)
  ) |>
  nest(.by = reviewerID, .key = "messages") |>
  select(-reviewerID)

prompts2 <- prompts2[indexes, ]

stream_out(prompts2[1:64, ], file("amazon_electronics_gpt_train_64.jsonl"))
stream_out(prompts2[1:128, ], file("amazon_electronics_gpt_train_128.jsonl"))
stream_out(prompts2[1:1024, ], file("amazon_electronics_gpt_train_1024.jsonl"))
stream_out(prompts2[1025:2048, ], file("amazon_electronics_gpt_valid.jsonl"))
stream_out(prompts2[2049:3072, ], file("amazon_electronics_gpt_test.jsonl"))
