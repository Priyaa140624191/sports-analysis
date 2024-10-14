import openai
import pandas as pd

# Set your OpenAI API key
openai.api_key = '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'

def get_relevant_data(nba_events_df, social_media_df, user_query):
    # Extract all unique team names dynamically from nba_events_df
    all_teams = pd.concat([nba_events_df['Team_1'], nba_events_df['Team_2']]).unique()

    # Ensure there are no NaN values in the team columns
    nba_events_df['Team_1'] = nba_events_df['Team_1'].fillna('')
    nba_events_df['Team_2'] = nba_events_df['Team_2'].fillna('')

    # Identify teams mentioned in the user query
    teams_in_query = [team for team in all_teams if team in user_query]

    if not teams_in_query:
        return None, None  # If no teams are found in the query, return None

    # Filter NBA events containing the relevant teams
    relevant_events = nba_events_df[
        (nba_events_df['Team_1'].isin(teams_in_query)) | (nba_events_df['Team_2'].isin(teams_in_query))
        ]

    # Ensure 'Keyword' column in social_media_df doesn't contain NaN values
    social_media_df['Keyword'] = social_media_df['Keyword'].fillna('')

    # Filter social media sentiment related to the teams in query
    relevant_social_media = social_media_df[
        social_media_df['Keyword'].str.contains('|'.join(teams_in_query), case=False, na=False)
    ]

    return relevant_events, relevant_social_media


# Example usage
user_query = "What is the sentiment around the game between Golden State Warriors and Los Angeles Lakers?"

# Assuming nba_events_df and social_media_df are already loaded DataFrames
nba_events_df = pd.read_csv('nba_events.csv')  # Load your NBA events data
social_media_df = pd.read_csv('final_consolidated_with_sentiment.csv')  # Load your social media data

# Get relevant data based on query
relevant_events, relevant_social_media = get_relevant_data(nba_events_df, social_media_df, user_query)

# Display the relevant data
if relevant_events is not None and relevant_social_media is not None:
    print("Relevant NBA Events:")
    print(relevant_events)
    print("\nRelevant Social Media Data:")
    print(relevant_social_media)
else:
    print("No relevant teams found in the query.")

def generate_prompt(relevant_events, relevant_social_media, user_query):
    # Check if relevant events are None or empty
    if relevant_events is None or relevant_events.empty:
        event_summary = "No relevant NBA events found."
    else:
        event_summary = relevant_events.head(10).to_string()  # Limit to 20 events

    # Check if relevant social media data is None or empty
    if relevant_social_media is None or relevant_social_media.empty:
        social_media_summary = "No relevant social media sentiment found."
    else:
        social_media_summary = relevant_social_media.head(10).to_string()  # Limit to 20 social media posts

    prompt_message = f"User query: {user_query}\n\nRelevant NBA Events:\n{event_summary}\n\nSocial Media Sentiment:\n{social_media_summary}"
    return prompt_message

def handle_user_query(user_query):
    # Get relevant data for the query
    relevant_events, relevant_social_media = get_relevant_data(nba_events_df, social_media_df, user_query)

    # Generate a prompt for ChatGPT
    prompt_message = generate_prompt(relevant_events, relevant_social_media, user_query)

    # Send request to GPT-4 with rate limiting considerations
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Translate and Summarise" + prompt_message,
        max_tokens=150  # Limit tokens for the response
    )

    # Return ChatGPT's response
    return response.choices[0].text

# Example query
user_query = "sentiment about Golden State Warriors?"
chatgpt_response = handle_user_query(user_query)


def chunk_dataframe(df, chunk_size):
    """ Split a DataFrame into chunks of a given size. """
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size]


def generate_prompt_chunked(relevant_events, relevant_social_media, user_query, chunk_size=10):
    # Initialize response list
    all_responses = []

    # Chunk events
    event_chunks = list(chunk_dataframe(relevant_events, chunk_size))
    social_media_chunks = list(chunk_dataframe(relevant_social_media, chunk_size))

    # Iterate through event and social media chunks
    for event_chunk, social_media_chunk in zip(event_chunks, social_media_chunks):
        event_summary = event_chunk.to_string() if not event_chunk.empty else "No relevant NBA events found."
        social_media_summary = social_media_chunk.to_string() if not social_media_chunk.empty else "No relevant social media sentiment found."

        prompt_message = (
            f"User query: {user_query}\n\n"
            f"Relevant NBA Events:\n{event_summary}\n\n"
            f"Social Media Sentiment:\n{social_media_summary}"
        )

        # Process each chunk with GPT-4 or another model
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt="Translate and Summarise" + prompt_message,
            max_tokens=150  # Limit tokens for the response
        )

        # Append the model's response to the list
        all_responses.append(response.choices[0].text)

    # Combine all chunk responses
    combined_response = "\n".join(all_responses)
    return combined_response


def get_chatgpt_response(user_query):
    # Example query
    #user_query = "What are the trending keywords or hashtags for the 2023-2024 NBA season in social media?"
    chatgpt_response = generate_prompt_chunked(relevant_events, relevant_social_media, user_query)
    return chatgpt_response