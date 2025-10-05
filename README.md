# RAG-based-AI-Teaching-Assistant
RAG Base Project – A work-in-progress project implementing Retrieval-Augmented Generation (RAG). It combines document retrieval with LLMs to deliver accurate, context-aware answers. Includes data ingestion, embeddings, retrieval pipeline, and a simple web interface.


# How to use this RAG AI Teaching Assistant on your on data 
## Step 1 - Collect your videos 
Move all your video files to the videos folder 

## Step 2 - Convert to mp3
convert all the videos files to mp3 by running video_to_mp3

## Step 3 - Convert mp3 to json 
Convert all the mp3 files to json by running mp3_to_json

## Step 4 - Convert the json files to vectors 
Use the file preprocess_json to convert the json files to a dataframe with Embedding and save it a joblib pickle

## step 5 - Prompt generation and feeding to LLM

Read the joblib file and load it into the memory. Then create a relevent prompt as per the user query and feed it to the LLM