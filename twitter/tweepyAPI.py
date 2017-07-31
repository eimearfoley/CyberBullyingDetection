import tweepy

auth = tweepy.OAuthHandler("U0Em10ONhazDk2skYDul5AOWl", "LGziXlksL69b3K5SJEBtGs0wdddNLJHohS7I92Yhgas7NaAbyK")
auth.set_access_token("404490975-u3s0cO6bWwHEIuRqESCTQvKo99lejMAcwgOmTpKT", "SXL9NK6iFtWaXWhFksAO58x6y9V3OE7AvFPCaMAHOa5fi")

api = tweepy.API(auth)
test_data = {}
public_tweets = api.home_timeline()
queries = ["donald trump","brexit","gay", "beyonce", "flowers","katie hopkins","priests","obama", "adele", "ireland"]
search_results = ""
for tag in queries:
    search_results = api.search(q=tag, count=100, lang="en")
    for result in search_results:
        test_data[result.text] = True

print(test_data)
