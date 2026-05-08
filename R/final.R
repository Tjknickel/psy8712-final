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
library(parallel)
library(doParallel)
# Post an API request to Ollama to access the Ollama endpoint to use the nomic-embed-text LLM embeddings model. The URL is to the local server (own computer) and the model is the nomic-embed-text model to get the embeddings, where text is the review text. json is used as the encode option to translate the request from R to the API. An error check is used to prevent the script from crashing with a warning message instead of an error message to diagnose problems. content(response) is used to extract the headings, timing, and metadata to get the embeddings needed for the model with $embeddings[1] to only extract the necessary information
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
# Load the datafile and create reviews_tbl, a dataframe with two columns, overall_rating and all_text. read_csv was used as the dataset is large and read.csv would be much slower and less efficient, also producing tibbles that work best with tidyverse functions. all_text was created using mutate() from the headline, pros, and cons columns as the instructions state to include all avaliable review text as source material. Finally, only the overall_rating and all_text columns are needed for the analysis, so all other columns were removed using select(). 
reviews_tbl <- read_csv("../data/glassdoor_reviews.csv") %>%
  mutate(all_text = paste(headline, pros, cons, sep = " ")) %>%
  select(overall_rating, all_text)
# Create a corpus using VCorpus, which is a specific data strucure with both data and metadata. VectorSource represents the data component and for this analysis all_text is the text vector to be used. 
corpus <- VCorpus(VectorSource(reviews_tbl$all_text))
# Pre-processing pipeline for the corpus using tm_map functions to convert the raw corpus into an object that can be used in the data analysis to predict overall_ratings from all_text. tm_map applies a function to the corpus and other functions from qdap or base R are wrapped in content_transformer. An anonymous function was used to merge acronymns (removes periods), followed by replacing contractions with their full actual word, and then coverting all text to lowercase. Other functions included removing numbers, punctuation, converting words to dictionary root words, removing certain stop words that do not convey important or meaningful information that would be useful to predict ratings, and finally stripping any extra spaces. 
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
# Create the unigram + bigram tokenizer using an anonymous function with lapply as a work around to using JAVA instead of NGramTokenzier and Weka_control which require the RWeka package. 1 = unigram, 2 = bigram. 
myTokenizer <- function(x) {
  unlist(lapply(ngrams(words(x), 1:2), paste, collapse = " "))
}
# Create a DTM using DocumentTermMatrix, the pre-processed corpus (corpus_prep), and the tokenizer (myTokenizer). 
reviews_dtm <- DocumentTermMatrix(
  corpus_prep, 
  control = list(tokenize = myTokenizer))
# Remove sparse columns from the dataset with removeSparseTerms, which removes columns that have more than 95% of zero values. 
reviews_slim_dtm <- removeSparseTerms(reviews_dtm, .95)
# Topic extraction: convert dtm to stm format with readCorpus for an LDA model
dtm_stm <- readCorpus(reviews_slim_dtm, type="slam")
# Find the optimal number of topics (K) and plotting result. Due to the scale of the dataset, a subset was used to find the optimal K to save on processing time. For reproducibility, set.seed was used when selecting the subset of indices to be used. 
set.seed(62)
sample_size <- 25000  
sample_indices <- sample(1:length(dtm_stm$documents), sample_size)
# searchK was used to find the optimal number of topics for meaningful topic extraction. For processing time, seq was restricted to 5 values and the data was subsetted to the sample_indices from the previous step. 
kresult <- searchK(
  documents = dtm_stm$documents[sample_indices],
  vocab = dtm_stm$vocab,
  K = seq(10, 50, 10),
  data = dtm_stm$meta[sample_indices, ],
  cores = 1
)
# Plotting the results of the searchK loop showed a bend/elbow in the plot at around 10-20 topics, although 40 topics had the highest semantic coherence. Having higher than 20 topics would make the dataset less interpretable, so 20 topics would be a reasonable in-between value to choose. 
plot(kresult)
# The final topic_model was created with K = 20 topics based on the plots from the previous step and by using stm. labelTopics was used to see what the most frequency uniquely occuring words were in each of the 20 topics in order to create the topic labels. A summary of the model was also created using plot() to guide the decision for the topic labels. 
topic_model <- stm(dtm_stm$documents,
                   dtm_stm$vocab,
                   20)
labelTopics(topic_model, n=10)
plot(topic_model, type="summary", n=2)
# Extract the ID of reviews that made it into the topic model
kept_indices <- as.integer(names(dtm_stm$documents))
# Indices of row numbers
all_indices <- 1:nrow(reviews_slim_dtm)
# Identify empty/dropped reviews 
dropped_indices <- setdiff(all_indices, kept_indices)
# Add a rn (row number) column to reviews_tbl
reviews_mod_tbl <- mutate(reviews_tbl, rn = row_number())
# Filters reviews_tbl to keep only reviews that were used in the model
reviews_kept_tbl <- reviews_mod_tbl[kept_indices, ]
# Create topic labels based on FREX output from labelTopics(), which highlights occuring words that don't appear in other topics 
all_labels <- c("Work_Life_Balance", "Work_Hours", "Work_Environment", "Satisfaction",
                "Work_Environment", "Project_Deadlines", "Work_Culture", "Pay",
                "People", "Time", "Career_Opportunities", "Effort", "Management",
                "Salary", "Team_Environment", "Experience", "Employees", "Scale",
                "Benefits", "Customers")
# Creates a tibble with the topic numbers and labels 
topic_labels <- tibble(
  topic = 1:20,
  topic_label = paste0(all_labels)
)
# Create topics_tbl, which combines the row numbers, all_text, topic, topic probabilities, and ratings into a single tibble. This tibble is then joined to topic labels by matching the topic column to combine into a single topics_tbl for analysis. 
topics_tbl <- tibble(
  doc_id = reviews_kept_tbl$rn,
  original = reviews_kept_tbl$all_text,
  topic = apply(topic_model$theta, 1, which.max),
  probability = apply(topic_model$theta, 1, max),
  ratings = reviews_kept_tbl$overall_rating
) %>%
  left_join(topic_labels, by = "topic")
# Create a slim version of the tbl by converting reviews_slim_dtm into a matrix and then into a tibble, and by using the kept_indices to only keep reviews that made it into the topic model
reviews_slim_tbl <- reviews_slim_dtm %>% as.matrix %>% as_tibble
reviews_slim_kept_tbl <- reviews_slim_tbl[kept_indices, ]
# Filter again for only 25000 reviews due to the size of the datatset to be used to extract embeddings for the analysis from topics_tbl (topics_sample)
reviews_slim_sample <- reviews_slim_kept_tbl[1:25000, ]
topics_sample <- topics_tbl[1:25000, ]
# Create a list of embeddings from the embeddings model from the POST request (get_embeddings). A progress bar was used to keep track of how long this process took.
embeddings_list <- map(reviews_kept_tbl$all_text[1:25000], 
                       possibly(get_embeddings, rep(NA_real_, 768)), 
                       .progress = TRUE)
# Convert the list of vectors into a clean 768 variable dataframe (embeddings_df)
embeddings_df <- as_tibble(do.call(rbind, embeddings_list))
embeddings_df <- as.data.frame(lapply(embeddings_df, unlist))
embeddings_df <- as.data.frame(lapply(embeddings_df, as.numeric))
# Rename columns for the ML analysis
colnames(embeddings_df) <- paste0("emb_", 1:768)
# Extract the full matrix of probabilities (Theta), again on a subset of the data to save on processing time
theta_matrix <- as.data.frame(topic_model$theta[1:25000, ]) 
# Rename columns for the ML analysis
colnames(theta_matrix) <- paste0("topic_prob_", 1:20)
# Create the final dataframe to be used in the machine learning analysis that binds together the overall_ratings, topic, theta_matrix, and embeddings together using bind_cols instead of cbind, as bind_cols works better with tibbles and lists of dataframes. NA values were also filtered out based on if they were missing an overall_rating or embedding value as this is needed in the analysis. 
ml_tbl <- reviews_slim_sample %>%
  bind_cols(
    overall_rating = topics_sample$ratings,
    topic = as.factor(topics_sample$topic)
    ) %>%
  bind_cols(theta_matrix) %>%
  bind_cols(embeddings_df) %>%
  filter(!is.na(overall_rating), !is.na(emb_1))
# Per the project requirements, this final dataframe was saved as an RDS file.
# saveRDS(ml_tbl, "../out/data.RDS")
# Created holdout and training sets from the ml_tbl. set.seed was used for reproducibility and createDataPartition was used to partition the data into holdout (reviews_holdout) and training (reviews_training) sets to be used in the training model or holdout cross-validation. 
set.seed(62)
holdout_indices <- createDataPartition(ml_tbl$overall_rating, 
                                       p = .25, 
                                       list=F)
reviews_holdout <- ml_tbl[holdout_indices,]
reviews_training <- ml_tbl[-holdout_indices,]
# Set up a shared train control across all models with 10 folds of cross-validation. 
train_ctrl <- trainControl(
  method      = "cv",
  number      = 10,
  verboseIter = TRUE
)
# Model 1: Tokenization only (dtm_cols from reviews_slim_sample to be used on the subset of data) model to predict overall_rating using rigiorous tokenization only. Glmnet models were used as this currently works well with the caret package in terms of processing speed and can handle sparse matrices and colinerities in this complex data structure. 
dtm_cols <- names(reviews_slim_sample)
model_dtm <- train(
  overall_rating ~ .,
  data      = select(reviews_training, all_of(dtm_cols), overall_rating),
  method    = "glmnet",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "scale", "nzv"),
  trControl  = train_ctrl
)
# Model 2: Embeddings only. Instead of selecting the dtm_cols, the embedding columns were selected to train this model to predict overall ratings.  
model_embed <- train(
  overall_rating ~ ., 
  data = select(reviews_training, starts_with("emb_"), overall_rating), 
  method = "glmnet", 
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "scale", "nzv"),
  trControl = train_ctrl
)
# Comparing the results of the two models using resamples(), summary(), and dotplot() to evaluate R2 and RMSE values for both models. The model with higher R2 and lower RMSE values performed best, and these results were used to answer RQ1. 
rq1_results <- resamples(list(
  "tokenization" = model_dtm,
  "embeddings"   = model_embed
))
summary(rq1_results)
dotplot(rq1_results, main = "RQ1: Embeddings vs Tokenization")

# RQ1. Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?
# The use of embeddings does drastically improve the prediction of satisfaction beyond using a rigorous tokenization strategy, as the R2 when adding the embeddings was 0.422 compared to an R2 value of 0.2 when just using tokenization. The RMSE was also lower for embeddings (0.927) compared to tokenization (1.091). 

# Model 3: Topics only. topic_prob columns were used to train this model to predict overall ratings.  
model_topic <- train(
  overall_rating ~ .,
  data      = select(reviews_training, starts_with("topic_prob"), overall_rating),
  method    = "glmnet",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "scale", "nzv"),
  trControl  = train_ctrl
)
# Compare the results of the tokenization only model to the topic only model to answer RQ2. 
rq2_results <- resamples(list(
  "tokenization"       = model_dtm,
  "topic" = model_topic
))
summary(rq2_results)
dotplot(rq2_results, main = "RQ2: Topics vs Tokenization")

# RQ2. Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?
# The use of topics with tokenization compared to rigorous tokenization alone did not improve the prediction of satisfaction, with an R2 value of 0.169 (compared to an R2 value of 0.2) and an RMSE value of 1.113 (compared to an RMSE value of 1.091). 

# Model 4: Embeddings + Topics. Combining both emb and topic_prob columns to predict overall ratings.  
model_emb_topic <- train(
  overall_rating ~ .,
  data      = select(reviews_training, starts_with("emb_"), starts_with("topic_prob"), overall_rating),
  method    = "glmnet",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "scale", "nzv"),
  trControl  = train_ctrl
)
# Compare the results of three models: embeddings only, topic only, and embeddings + topics to answer RQ3. 
rq3_results <- resamples(list(
  "embeddings" = model_embed,
  "topic" = model_topic,
  "embeddings+topic" = model_emb_topic
))
summary(rq3_results)
dotplot(rq3_results, main = "RQ3: Embeddings + Topics vs Either Alone")

# RQ3. Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?
# The use of embeddings pluse topics does slightly improve the prediction of satisfaction beyond either embeddings (R2 = 0.422) or topics (R2 = 0.168) alone, with an R2 value of 0.433 and RMSE value of 0.927. 

# Model 5: Full model using glmnet. A final full model was used to see if using the full reviews_training variables to predict overall ratings beyond the other 4 models which were a subset of these variables. 
model_full_glmnet <- train(
  overall_rating ~ .,
  data      = reviews_training,
  method    = "glmnet",
  na.action = na.pass,
  preProcess = c("medianImpute", "center", "scale", "nzv"),
  trControl  = train_ctrl
)
# Compare all 5 models to each other to answer RQ4. 
rq4_results <- resamples(list(
  "full" = model_full_glmnet,
  "tokenization"     = model_dtm,
  "embeddings"    = model_embed,
  "token+embed" = model_emb_topic,
  "topic" = model_topic
))
summary(rq4_results)
dotplot(rq4_results, main = "RQ4: Best Model Comparison (Full Features)")
# Tokenization only model holdout CV = 0.47 (correlation between predictions from trained model and actual criterion values, how the trained model performed on holdout data)
pred_dtm <- predict(model_dtm, newdata = reviews_holdout)
cor(pred_dtm, reviews_holdout$overall_rating)
# Embeddings only model holdout CV = 0.666
pred_embed <- predict(model_embed, newdata = reviews_holdout)
cor(pred_embed, reviews_holdout$overall_rating)
# Topics only model holdout CV = 0.436
pred_topic <- predict(model_topic, newdata = reviews_holdout)
cor(pred_topic, reviews_holdout$overall_rating)
# Embeddings + Topics model holdout CV = 0.675
pred_emb_topic <- predict(model_emb_topic, newdata = reviews_holdout)
cor(pred_emb_topic, reviews_holdout$overall_rating)
# Full model holdout CV = 0.68
pred_full <- predict(model_full_glmnet, newdata = reviews_holdout)
cor(pred_full, reviews_holdout$overall_rating)

#  RQ4. What is the best prediction of overall job satisfaction achievable using text reviews as source data?
# The best prediction of overall job satisfaction using text reviews as source data was from using the full model will embeddings, topics, and tokenization to predict overall ratings with a holdout correlation of 0.68. This indiciates that the model was able to accurately predict overall ratings on the holdout data from the trained model and a strong positive correlation between these predictions and the actual rating values. However, the full model only slightly outperfomed the combined embeddings and topics model, indicating that these two in combination were able to almost as accurately predict ratings compared to the full model that also used tokenization.  

save.image("../out/final.RData")