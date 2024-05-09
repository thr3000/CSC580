import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt

def cleaningTweets(data):
    if not isinstance(data, str):
        return ""
    tweet_without_url = re.sub(r'http\S+',' ', data)
    tweet_without_hashtag = re.sub(r'#\w+', ' ', tweet_without_url)
    tweet_without_mentions = re.sub(r'@\w+',' ', tweet_without_hashtag)
    cleaned_tweet = re.sub('[^A-Za-z]+', ' ', tweet_without_mentions)
    return cleaned_tweet.strip()

def cleaningDate(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d')
    df = df.dropna(subset=['date'])
    #df['date'] = df['date'].dt.date
    return df

def filterData(df):
    df_filtered = df[['date', 'text']]
    return df_filtered

def getSentiment(data):
    score = TextBlob(data).sentiment.polarity
    if score > 0:
        return 'Positive'
    elif score == 0:
       return 'Other'
    else:
        return 'Negative'
    
def generate_graph(result):
    plt.figure(figsize=(10, 5))

    for sentiment in result.columns:
        plt.plot(result.index, result[sentiment], marker='o', label=sentiment)

    plt.title('Sentiment Trend Over Years')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig('./sentiment_trend.png')

    ax = result.plot(kind='bar', stacked=True, figsize=(10, 5))

    ax.set_title('Sentiment Distribution Over Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    ax.legend(title='Sentiment')

    plt.savefig('./sentiment_distribution.png')

df = pd.read_csv('./combined_without_other.csv')
# df_combined = pd.concat([df1, df2], ignore_index=True)

df = cleaningDate(df)
# df = filterData(df)
df['text'] = df['text'].apply(cleaningTweets)
# df['text'] = df['text'].replace('', pd.NA)
# df['date'] = pd.to_datetime(df['date'])
# df['year'] = df['date'].dt.year
# df_2019 = df[df['year'] == 2019]
# df_2022 = df[df['year'] == 2022]
# df_2019 = cleaningDate(df_2019)
# df_2022 = cleaningDate(df_2022)
# df_2019['text'] = df_2019['text'].apply(cleaningTweets)
# df_2022['text'] = df_2022['text'].apply(cleaningTweets)
df.dropna(inplace=True)
df.to_csv('./combined_without_other.csv', index=False)
# df_2022.dropna(inplace=True)
# filtered_df = df[df['sentiment'] != 'Other']
# filtered_df.to_csv('filtered_file.csv', index=False)

# df['sentiment'] = df['text'].apply(getSentiment)
# df.drop('year', axis=1, inplace=True)

# df.rename(columns={'Date': 'date', 'text': 'text', 'Sentiment': 'sentiment'}, inplace=True)
# print(df.columns)
# result = df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
# print(result)

# df.to_csv('./combined_PN.csv', index=False)

# num_rows = df.shape[0]
# print(f"The number of rows in the CSV file is: {num_rows}")
# print(df.head())
# sentiment_counts = df['sentiment'].value_counts()
# print(sentiment_counts)