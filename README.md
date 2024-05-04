To-do
Group Members:
1. Medha Sreenivasan ms1112
2. Thwisha Nahender tn130
3. Ritu Toshniwal rt333

This repository contains the codebase for a chatbot powered by the RAG (Retrieval-Augmented Generation) process and integrated with ChromaDB for efficient document storage and retrieval.

Setting up the ChromaDB
If you wish to set up the chromadb server locally follow the below steps

To start the chromadb server run the below command: chroma run --path <path to your DB>
To add, delete or create a new collection use the manage_collection.py script python manage_collection.py NEW <collection_name> <file_path>
If you wish to use chromadb on colab/local with a persist directory follow the below steps:

run the script RAG/manage_collection.py to add, delete or create a new collection
Use a persistent client to access the data chromadb.PersistentClient(path=<path to DB>)
Running the chatbot application
The functionality currently uses mistralai/Mistral-7B-Instruct-v0.2 model from huggingface for the RAG application.

Run the application python app.py to get the gradio interface
Evaluation
To perform evaluation run eval.py For evaluating the RAG pipeline, we have adapted the metrics from ragas repository. For more details on evaluation check out https://github.com/explodinggradients/ragas
