import sys
import json
import urllib
import requests

# password = sys.argv[1]
# bearer_token = sys.argv[1]

# bearer_token = 'D8MPXCSE0RdkNES5CrPLSxYd2'
bearer_token = '46526752-og4BYM3wKbDNwnMcy1LwhMWuPYFumCGIQp9jLM7YA'
query = 'trump'
fname = 'test.csv'
max_tweets = 100


def make_request(method, url):
    r = requests.request(method, url, headers={'Authorization': 'Bearer {}'.format(bearer_token)})
    assert r.status_code < 400, r.content
    return r


def search_tweets(max_tweets, query_string):
    """ Find some tweets matching a particular query

    Parameters
    ----------
    max_tweets : int
        the maximum number of tweets to return

    query_string : str
        the search to make, compatible with Twitter's Search API

    Yields
    ------
    tweet : dict
        a single tweet, as returned by twitter's search API

    Examples
    --------
    Find up to fifty tweets using '#nastywoman'
    search = search_tweets(50, '#nastywoman')
    for tweet in search:
        print tweet['text']
    """
    query_string = urllib.urlencode({'q': query_string})
    url = 'https://api.twitter.com/1.1/search/tweets.json?lang=en&{}'.format(query_string)
    next_search = url
    total_count = 0
    while total_count < max_tweets:
        content = json.loads(make_request('GET', next_search).content)
        min_id = 'not set'
        if not content['statuses']:
            break
        for tweet in content['statuses']:
            if tweet['id'] < min_id:
                min_id = tweet['id']
            yield tweet
            total_count += 1
            if total_count >= max_tweets:
                break
        next_search = '{}&max_id={}'.format(url, min_id - 1)


def log_tweets(filename, query_string, max_tweets):
    """ Collect a series of tweets and write them to a file

    File will be formatted like
    polarity, text
    0, "lorem ipsum dolor sit amet, consectetur adipiscing elit :("
    4, "sed do eiusmod tempor incididunt ut labore et dolore :)"
    """
    with open(filename, 'w') as out_f:
        out_f.write('polarity,text\n')
        for tweet in search_tweets(max_tweets, query_string + '" :)"'):
            out_f.write(u'4,"{}"\n'.format(tweet['text']).encode('UTF-8'))
        for tweet in search_tweets(max_tweets, query_string + '" :("'):
            out_f.write(u'0,"{}"\n'.format(tweet['text']).encode('UTF-8'))
