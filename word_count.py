import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Reload the dataset while skipping the first row (which contains 'sep=')
file_path = 'final_consolidated_with_sentiment.csv'
df = pd.read_csv(file_path, header=0)

# Function to extract hashtags from the original text
def extract_hashtags(Text):
    hashtags = re.findall(r'#\w+', Text)
    return hashtags

# Apply the function to extract hashtags from the 'text' column
df['hashtags'] = df['Text'].apply(lambda x: extract_hashtags(str(x)))

df_cleaned = df.copy()

# Clean and tokenize the 'text' column by removing special characters, converting to lowercase
df_cleaned['clean_text'] = df_cleaned['Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower()))

# Tokenize the text by splitting on whitespace
df_cleaned['tokens'] = df_cleaned['clean_text'].apply(lambda x: x.split())

# Download stopwords
nltk.download('stopwords')

# Define stopwords set
stop_words = set(stopwords.words('english'))

# Remove stopwords from the tokens
df_cleaned['filtered_tokens'] = df_cleaned['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

# Combine all filtered tokens into a single list
all_words_filtered = [word for tokens in df_cleaned['filtered_tokens'] for word in tokens]

# Count the frequency of the filtered words
word_counts_filtered = Counter(all_words_filtered)

# Display the 10 most common words after removing stopwords
common_words_filtered = word_counts_filtered.most_common(10)

# Combine all filtered tokens into a single list
all_words_filtered = [word for tokens in df_cleaned['filtered_tokens'] for word in tokens]

# Count the frequency of the filtered words
word_counts_filtered = Counter(all_words_filtered)

# Get the 10 most common words
common_words_filtered = word_counts_filtered.most_common(10)

# Extract words and counts separately for visualization
words, counts = zip(*common_words_filtered)

# # Create a bar chart to visualize the most common words
# plt.bar(words, counts, color='blue')
# plt.title('Top 10 Most Common Keywords')
# plt.xlabel('Words')
# plt.ylabel('Frequency')
# plt.xticks(rotation=45)  # Rotate the words for better readability
# plt.show()
#
# Combine all filtered tokens into a single string
all_words_string = ' '.join(all_words_filtered)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words_string)

# # Display the word cloud
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # Turn off the axis
# plt.show()

# Create a function to return the necessary data
def get_visualization_data():
    return words, counts, wordcloud