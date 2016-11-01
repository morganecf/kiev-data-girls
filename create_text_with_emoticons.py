import re
import codecs

punctuation = re.compile(r'[!\?\.](\s|$)')

lines = codecs.open('processed_noemoticon.csv', 'r', 'ISO-8859-1').readlines()
out = open('tweets_with_emoticons.csv', 'wb')
out.write('tweet\n'.encode('utf-8'))

for line in lines[1:]:
  # There are commas within text >:(
  comma_index = line.find(',')
  label = line[:comma_index].strip()
  tweet = line[comma_index + 1:].strip()

  # Find first punctuation
  match = re.search(punctuation, tweet)
  i = match.start() if match else len(line) - 1

  # Insert emoticon after punctuation
  emoticon = ' :( ' if label == '0' else ' :) '
  new_tweet = (tweet[:i + 1] + emoticon + tweet[i:]) + '\n'
  new_tweet = new_tweet.encode('utf-8')

  # Save tweet
  out.write(new_tweet)

out.close()
