import pandas as pd
import re
from textblob import TextBlob

def cleaningTweets(data):
    if not isinstance(data, str):
        return ""
    tweet_without_url = re.sub(r'http\S+',' ', data)
    tweet_without_hashtag = re.sub(r'#\w+', ' ', tweet_without_url)
    tweet_without_mentions = re.sub(r'@\w+',' ', tweet_without_hashtag)
    cleaned_tweet = re.sub('[^A-Za-z]+', ' ', tweet_without_mentions)
    return cleaned_tweet.strip()

def cleaningDate(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.date

def filterData(df):
    df_filtered = df[['date', 'text']]
    return df_filtered

def getSentiment(data):
    score = TextBlob(data).sentiment.polarity
    if score > 0:
        return 'Positive'
    # elif score == 0:
    #    return 'Other'
    else:
        return 'Negative'

df = pd.read_csv('./combined.csv')
# df_combined = pd.concat([df1, df2], ignore_index=True)

# cleaningDate(df)
# df = filterData(df)
# df['text'] = df['text'].apply(cleaningTweets)
# df['text'] = df['text'].replace('', pd.NA)
# df.dropna(subset=['text'], inplace=True)

# df['sentiment'] = df['text'].apply(getSentiment)

# df.rename(columns={'Date': 'date', 'text': 'text', 'Sentiment': 'sentiment'}, inplace=True)
# print(df.columns)

# df_combined.to_csv('./combined_PN.csv', index=False)

num_rows = df.shape[0]
print(f"The number of rows in the CSV file is: {num_rows}")
print(df.head())
# sentiment_counts = df['sentiment'].value_counts()
# print(sentiment_counts)

