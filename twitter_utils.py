import tweepy
import time

test_tweet_id = 1367961367882444800
consumer_key = "3rJOl1ODzm9yZy63FACdg"
consumer_secret = "5jPoQ5kQvMJFDYRNE8bQ4rHuds4xJqhvgNJM4awaE8"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)


def get_tweet_text(tweet_id: int) -> str:
    time.sleep(3)
    print(f"[GETTING TWEET ID]: {tweet_id}")
    tweet = api.get_status(tweet_id, tweet_mode="extended")
    return tweet.full_text


def print_tweet(tweet_id: int):
    print(f"tweet id: {tweet_id}")
    print(get_tweet_text(test_tweet_id))
