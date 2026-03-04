# Many articles in the analyst_ratings_processed.csv have repeated titles referring to different stocks. Looking to consolidate all of these
# articles into single entries with combined stocks. Will then save as a new csv to be ingested.

import pandas as pd

""" Uncomment to run consolidation
df = pd.read_csv("dataset/analyst_ratings_processed.csv")

print(f"Original data has {len(df)} rows.")

# Remove rows with faulty data
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df = df.dropna(subset=['date'])

print(f"After cleaning, data has {len(df)} rows.")

# Group by title and date, then aggregate the stocks into a list
grouped_df = df.copy().groupby(['title', 'date'], as_index=False).agg({'stock': lambda x: ','.join(sorted(set(x)))})

print(f"After grouping, data has {len(grouped_df)} rows.")

# Let's look at how much data we have for each quarter
grouped_df['q_index'] = grouped_df['date'].dt.to_period('Q').astype(str)
articles_by_q = grouped_df['q_index'].value_counts(dropna=False).reset_index()
print(articles_by_q)

# Order df by earliest to latest date and add primary index col numbered from 0 to n-1
grouped_df = grouped_df.sort_values(by='date').reset_index(drop=True)
grouped_df.insert(0, 'index', range(len(grouped_df)))

# Save the consolidated DataFrame to a new CSV file
grouped_df.to_csv("dataset/analyst_ratings_consolidated.csv", index=False)"""

# Are there any duplicate headline titles?

df = pd.read_csv("dataset/analyst_ratings_consolidated.csv")
duplicate_titles = df[df.duplicated(subset=['title'], keep=False)]
if not duplicate_titles.empty:
    print("Duplicate titles found:")
    print(len(duplicate_titles), "duplicates")

# What about duplicate (title, date, stock) combinations?

duplicate_combinations = df[df.duplicated(subset=['title', 'date', 'stock'], keep=False)]
if not duplicate_combinations.empty:
    print("Duplicate (title, date, stock) combinations found:")
    print(len(duplicate_combinations), "duplicates")
else:
    print("No duplicate (title, date, stock) combinations found.")

# How many words are in each article title on average?

df['word_count'] = df['title'].apply(lambda x: len(str(x).split()))
average_word_count = df['word_count'].mean()
print(f"Average word count in article titles: {average_word_count:.2f}")

# Split into 2 csvs to push to Github; will need to write a separate script to recombined before ingestion

mid_index = len(df) // 2
df.iloc[:mid_index].to_csv("dataset/analyst_ratings_consolidated_part1.csv", index=False)
df.iloc[mid_index:].to_csv("dataset/analyst_ratings_consolidated_part2.csv", index=False)