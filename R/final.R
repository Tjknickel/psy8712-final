# Set the working directory using getActiveDocumentContext instead of setwd followed by the path name directory or by using R Studio settings directly as this is more reproducible for running the script on a different machine
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Library calls for necessary packages to run the script. 
library(tidyverse)
library(rvest)
library(httr)
library(tm)
library(textclean)
library(textstem)
library(caret)
library(stm)
library(jsonlite)

# Post an API request to Ollama to access the Ollama endpoint to use the nomic-embed-text LLM embeddings model
get_embeddings <- function(text) {
  response <- POST(
    url = "http://localhost:11434/api/embed",
    body = list(
      model = "nomic-embed-text", 
      input = text),
    encode = "json"
  )
  if (status_code(response) != 200) {
    warning("Request failed with status: ", status_code(response))
    return(rep(NA_real_, 768))
  }
  return(httr::content(response)$embeddings[[1]])
}
# Load the datafile and create reviews_tbl, a dataframe with two columns, overall_rating and all_text  
reviews_tbl <- read_csv("../data/glassdoor_reviews.csv") %>%
  mutate(all_text = paste(headline, pros, cons, sep = " ")) %>%
  select(overall_rating, all_text)
# Create a corpus 
corpus <- VCorpus(VectorSource(reviews_tbl$all_text))

corpus_prep <- corpus %>%
  tm_map(content_transformer(function(x) str_remove_all(x, "(?<=\\b[A-Z])\\.(?=[A-Z]\\b)"))) %>%
  tm_map(content_transformer(replace_contraction)) %>%
  tm_map(content_transformer(str_to_lower)) %>%
  tm_map(removeNumbers) %>%
  tm_map(removePunctuation) %>%
  tm_map(content_transformer(lemmatize_strings)) %>%
  tm_map(removeWords, c(
    "company", "job", "work", "place", "glassdoor",
    stopwords("en")
  )) %>%
  tm_map(stripWhitespace)
