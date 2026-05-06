# Set the working directory using getActiveDocumentContext instead of setwd followed by the path name directory or by using R Studio settings directly as this is more reproducible for running the script on a different machine
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Library calls for necessary packages to run the script. 
library(tidyverse)
library(rvest)
library(httr)

# Read in the csv file 