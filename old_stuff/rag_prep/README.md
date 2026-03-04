BASIC DOCS:

eda_outdated.ipynb: looks at dataset statistics, news articles per quarter, etc.
preprocess.py: condenses the dataset so that duplicate articles referring to different stocks are coalesced, then checks if dataset is still usable, how many words per doc for cost estimation purposes, etc.
ingest.py: uses gemini-embeddings-001 and bm25 to build vector DB of news.
modify_bm25_index.py: realized that ingest did not give bm25 enough information about headline publishing date or affiliated stocks, so modified data stored in the original corpus to allow for those filters
ingest_macro_only.py: takes macro/fed/trade headlines from macro_events.csv and adds them to the existing vector store, recomputes BM25 (because I made this dataset later to address gaps in original news with macro events)

Note that none of these need to be run as of right now: both news_bm25_index and vector_store have been put on Google Drive and are accessible on request; analyst_ratings_consolidated.csv exists on Sharabh's local machine, but it has also been split into two and pushed to Github (you will need to write a script to combine the two csvs before running ingest if you REALLY need to).

If you did need to run any of these scripts, move them to the root dir of this project and run from there, since I haven't updated imports or file paths. Same for locally_embedded, although those files are now deprecated.