BASIC DOCS:

rag_core.py: defines EmbeddingManager, VectorStore, and RAGRetriever classes so that you can run any script in the root directory or rag_prep (provided the file is moved to root first; see rag_prep/README.md for more info)
search.py: the most basic search functionality. Provides individual BM25, Dense Vector Search, and RRF headline results for a given query, along with amt of time taken to retrieve each.
search_w_filter_decay.py: uses publishing date and associated stocks in metadata of BM25/ChromaDB to constrain search to before a certain date/a specified set of stocks. Additionally, sets a decay function that penalizes news much earlier than the specified date. Only provides RRF results.
search_interleaved.py: same functionality as search_w_filter_decay, except two lists (decay, no decay) are calculated to equally prioritize the latest news and important historical headlines. also returns an RRF with labels saying if the headline is historical or recent.


read through the scripts to gain a better understanding of how the RAG is queried.