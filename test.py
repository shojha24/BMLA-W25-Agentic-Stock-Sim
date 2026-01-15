import csv
import os

def split_csv_in_two(input_file, output_file1, output_file2):
    # Validate file existence
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"File '{input_file}' not found.")

    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = list(csv.reader(infile))
        
        if not reader:
            raise ValueError("The CSV file is empty.")

        header = reader[0]
        rows = reader[1:]

        if not rows:
            raise ValueError("The CSV file contains only a header and no data.")

        # Calculate split point
        mid_index = len(rows) // 2

        # First half
        with open(output_file1, 'w', newline='', encoding='utf-8') as out1:
            writer = csv.writer(out1)
            writer.writerow(header)
            writer.writerows(rows[:mid_index])

        # Second half
        with open(output_file2, 'w', newline='', encoding='utf-8') as out2:
            writer = csv.writer(out2)
            writer.writerow(header)
            writer.writerows(rows[mid_index:])

    print(f"✅ Split complete: '{output_file1}' and '{output_file2}' created.")

# Example usage
try:
    split_csv_in_two("dataset/analyst_ratings_processed.csv", "dataset/news_p1.csv", "dataset/news_p2.csv")
except Exception as e:
    print(f"Error: {e}")