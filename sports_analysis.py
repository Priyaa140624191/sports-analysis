import streamlit as st
import pandas as pd
from word_count import get_visualization_data
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objects as go
import openai
from open_ai_integration import get_chatgpt_response

# Load the sports data CSV file
file_path = "nba_events.csv"  # Update this with the correct file path
sports_data = pd.read_csv(file_path)

# Set a title for the Streamlit app
st.title('NBA Events Data')

# Display the sports data in a table
st.write('Here are the first few rows of the NBA events data:')
st.dataframe(sports_data.head())  # Display the first 5 rows of the dataset

# Cache the data conversion to CSV with st.cache_data
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(sports_data)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='nba_events.csv',
    mime='text/csv',
)

# Load social media data CSV file
file_path = "final_consolidated_with_sentiment_download.csv"  # Update with your correct file path
social_media_data = pd.read_csv(file_path)

# Set a title for the Streamlit app
st.title('Social Media Sentiment Analysis')

# Display the dataset with filter options
st.write('Filter and view the social media data based on sentiment.')

# Add filtering options for sentiment with capitalized display
sentiment_filter = st.selectbox(
    'Select Sentiment:',
    options=['Positive', 'Neutral', 'Negative', 'All'],  # Keep capitalized options
    index=3  # Default to "All"
)

# Filter based on selected sentiment
if sentiment_filter != 'All':
    filtered_data = social_media_data[social_media_data['Sentiment'].str.lower() == sentiment_filter.lower()]
else:
    filtered_data = social_media_data

# Show filtered data
st.write('Filtered Social Media Data:')
st.dataframe(filtered_data)

# Cache the data conversion to CSV
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(filtered_data)

st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name='filtered_social_media_data.csv',
    mime='text/csv',
)
import plotly.express as px
# Add option to select which sentiment column to visualize
sentiment_column = st.selectbox(
    'Select Sentiment Source:',
    options=['Social Searcher Sentiment', 'TextBlob Sentiment'],
    index=0,
    key='sentiment_source'  # Unique key to avoid duplicate ID error
)

# Ensure your CSV has two sentiment columns:
# "Social_Searcher_Sentiment" and "TextBlob_Sentiment"
if sentiment_column == 'Social Searcher Sentiment':
    sentiment_column_name = 'Sentiment'  # Replace with the actual column name
else:
    sentiment_column_name = 'TextBlob_Sentiment'  # Replace with the actual column name

# Ensure both columns are lowercase to avoid case mismatch
social_media_data[sentiment_column_name] = social_media_data[sentiment_column_name].str.lower()

# Group by the selected sentiment column and count occurrences
sentiment_counts = social_media_data[sentiment_column_name].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']  # Rename columns for clarity

# Plot the pie chart with Plotly
fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title=f'{sentiment_column} Distribution')
# Show the pie chart in Streamlit
st.plotly_chart(fig)

# Plot the bar chart with Plotly
fig = px.bar(sentiment_counts, x='Sentiment', y='Count', title=f'{sentiment_column} Distribution', color='Sentiment')
# Show the bar chart in Streamlit
st.plotly_chart(fig)

# Get the data from word_count.py
words, counts, wordcloud = get_visualization_data()

df_word = pd.DataFrame({'Words': words, 'Counts': counts})

st.title('Top 10 Most Common Keywords')

# Create the bar chart using Plotly Express
fig_bar = px.bar(df_word, x='Words', y='Counts', title='Top 10 Most Common Keywords')
fig_bar.update_layout(
    xaxis_title='Words',
    yaxis_title='Frequency',
    xaxis_tickangle=-45  # Rotate x-axis labels for better readability
)

# Display the bar chart in Streamlit
st.plotly_chart(fig_bar)

# Display the word cloud in Streamlit
st.title('Word Cloud')

# Render the word cloud image in Streamlit
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')  # Turn off the axis

st.pyplot(fig)

st.title("Consolidated Sentiments")
# Read the consolidated social media data from CSV
consolidated_data = pd.read_csv('final_consolidated_with_sentiment.csv')
st.dataframe(consolidated_data)

# Ask for user's name
user_query = st.text_input("Enter your question to ChatGPT")
chatgpt_response = get_chatgpt_response(user_query)

st.write(chatgpt_response)

