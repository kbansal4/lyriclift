import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec
import numpy as np

# Read the CSV file
music_data = pd.read_csv("tcc_ceds_music.csv")

# Filter data by release date and reset index
filtered_data = music_data[(music_data['release_date'] >= 2000) & (music_data['release_date'] <= 2019)]
filtered_data.reset_index(drop=True, inplace=True)

# Sentiment analysis setup
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(lyrics):
    sentiment_scores = analyzer.polarity_scores(lyrics)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.75:
        return 'very positive'
    elif 0.05 <= compound_score < 0.75:
        return 'positive'
    elif -0.75 <= compound_score <= -0.05:
        return 'negative'
    elif compound_score <= -0.75:
        return 'very negative'
    else:
        return 'neutral'

# Use .loc to avoid SettingWithCopyWarning
filtered_data.loc[:, 'sentiment'] = filtered_data['lyrics'].apply(analyze_sentiment)

# Drop rows with missing or empty lyrics
filtered_data = filtered_data.dropna(subset=['lyrics'])
filtered_data = filtered_data[filtered_data['lyrics'].str.strip() != '']

def interpret_metric(metric_name, value):
    if metric_name == 'danceability':
        if value <= 0.25:
            interpretation = "Your track should have low danceability. Opt for slow tempos, soft percussion, and minimal rhythmic elements. Ambient, experimental, or folk genres might fit well."
        elif value <= 0.5:
            interpretation = "Your track should have moderate danceability. Aim for a balanced rhythm and groove without being overpowering. Consider using relaxed beats, subtle bass lines, and soft syncopation."
        elif value <= 0.75:
            interpretation = "Your track should have moderately high danceability. Incorporate clear rhythms, moderate tempos, and catchy, repetitive hooks. Pop, indie, or soft electronic genres may work well."
        else:
            interpretation = "Your track should have high danceability. Use energetic, tight rhythms, upbeat tempos, and groovy bass lines. Genres like EDM, funk, disco, and hip-hop are great examples."

    elif metric_name == 'loudness':
        if value <= -20:
            interpretation = "Your track should have low loudness. Keep the overall volume and dynamics subtle. Acoustic, chillout, or classical tracks might fit this level with softer instruments and delicate production."
        elif value <= -10:
            interpretation = "Your track should have moderate loudness. Aim for controlled dynamics with moments of intensity. Use a mix of soft and louder sections to build emotion without overwhelming the listener."
        else:
            interpretation = "Your track should have high loudness. Push for maximum impact with strong dynamics and loud instrumentation. Genres like heavy rock, metal, and EDM benefit from bold volume and aggressive soundscapes."

    elif metric_name == 'acousticness':
        if value <= 0.1:
            interpretation = "Your track should have very low acousticness. Focus on electronic and synthesized sounds rather than organic instruments. Use digital production, heavy processing, and effects."
        elif value <= 0.25:
            interpretation = "Your track should have low acousticness. Emphasize electronic elements over acoustic ones, though some organic sounds may be present."
        elif value <= 0.5:
            interpretation = "Your track should have moderate acousticness. Combine acoustic and electronic elements, such as using live instruments with some post-processing. Indie or electro-acoustic genres might fit well."
        elif value <= 0.75:
            interpretation = "Your track should have moderately high acousticness. Acoustic instrumentation should be prominent, with a warm, organic sound."
        else:
            interpretation = "Your track should have high acousticness. Focus on live recordings and minimal processing. Acoustic genres like folk and unplugged versions work well here."

    elif metric_name == 'energy':
        if value <= 0.25:
            interpretation = "Your track should have low energy. Keep the arrangement simple and calm with slow tempos and gentle instrumentation. Ambient, ballads, or downtempo genres work well."
        elif value <= 0.5:
            interpretation = "Your track should have moderate energy. Use moderate tempos with gradual build-ups. Soft rock, jazz, and indie music often fall into this category, with a mix of laid-back and dynamic moments."
        elif value <= 0.75:
            interpretation = "Your track should have moderately high energy. Use vibrant tempos, energetic beats, and dynamic arrangements. Pop, indie rock, and upbeat R&B are good fits here."
        else:
            interpretation = "Your track should have high energy. Go for fast tempos, driving rhythms, and intense instrumentation. Genres like EDM, punk, and dance music thrive on high-energy elements."

    return interpretation

# Tokenize the lyrics for Word2Vec
filtered_data['lyrics_tokens'] = filtered_data['lyrics'].apply(lambda x: word_tokenize(x.lower()))

# Train a Word2Vec model
vector_size = 100
w2v_model = Word2Vec(sentences=filtered_data['lyrics_tokens'], vector_size=vector_size, window=5, min_count=1, workers=4)


def get_average_word2vec(tokens_list, model, vector_size):
    vectorized = [model.wv[word] for word in tokens_list if word in model.wv]
    if len(vectorized) == 0:
        return np.zeros(vector_size)
    else:
        return np.mean(vectorized, axis=0)

def find_top_songs_word2vec(user_lyrics, filtered_data, user_sentiment, w2v_model, vector_size):
    # Compute embeddings for dataset if not already computed
    if 'w2v_embedding' not in filtered_data.columns:
        filtered_data['w2v_embedding'] = filtered_data['lyrics_tokens'].apply(lambda tokens: get_average_word2vec(tokens, w2v_model, vector_size))
    lyrics_w2v_embeddings = np.stack(filtered_data['w2v_embedding'].values)
    user_tokens = word_tokenize(user_lyrics.lower())
    user_w2v_embedding = get_average_word2vec(user_tokens, w2v_model, vector_size)
    cosine_similarities = cosine_similarity([user_w2v_embedding], lyrics_w2v_embeddings).flatten()
    sorted_indices = cosine_similarities.argsort()[::-1]
    top_songs = []
    for idx in sorted_indices:
        song = filtered_data.iloc[idx]
        if song['sentiment'] == user_sentiment:
            top_songs.append(song)
            if len(top_songs) == 5:
                break
    return top_songs


# Get user input
user_lyrics = input("Please enter your song lyrics: ")

start_time = time.time()

# Analyze sentiment of the user's lyrics
user_sentiment = analyze_sentiment(user_lyrics)
print(f"\nThe sentiment of your lyrics is: {user_sentiment}")


# Find top songs using Word2Vec
print("\nFinding top songs using Word2Vec...")
top_songs_w2v = find_top_songs_word2vec(user_lyrics, filtered_data, user_sentiment, w2v_model, vector_size)


# Function to generate recommendations
def generate_recommendations(top_songs, method_name):
    if top_songs:
        print(f"\nTop 5 similar songs using {method_name} with the same sentiment as your input are:")
        for song in top_songs:
            print(f"- {song['track_name'].title()} by {song['artist_name'].title()}")
        top_df = pd.DataFrame(top_songs)
        recommended_metrics = top_df[['danceability', 'loudness', 'acousticness', 'energy']].mean().round(2)
        print(f"\nRecommended production metrics based on the top similar songs using {method_name}:")
        for metric in recommended_metrics.index:
            value = recommended_metrics[metric]
            print(f"\nRecommended {metric}: {value}. \n{interpret_metric(metric, value)}")
    else:
        print(f"\nSorry, no similar songs with the same sentiment were found using {method_name}.")

# Generate recommendations for each method
generate_recommendations(top_songs_w2v, "Word2Vec")

# Calculate and print the elapsed time
elapsed_time = time.time() - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
