import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from stop_words import get_stop_words
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import matplotlib.pyplot as plt

from lift import lift

def format_data(filename, sample_size):
  '''
  Get data, clean, and find positive/negative rows and label them.
  '''
  # Read in data
  df = pd.read_csv(filename, encoding='iso-8859-1').head(1599950)['tweet']

  # Take subset
  subset = df.sample(n=sample_size)

  # Set up regexes
  happy = r':\)'
  sad = r':\('
  mention = r'^(.*\s)?@\w+'
  url = r'^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$'

  # Set up regex to find smiley faces
  # TODO and doesn't contain sad?
  targets = subset.str.contains(happy)

  print('Target breakdown:', targets.value_counts())

  # Remove smiley-faces from text -- don't want to train on this!
  # TODO - could just use one regex
  tweets = subset.str.replace(happy, '')
  tweets = tweets.str.replace(sad, '')

  # Remove other stuff that might pollute training set
  # Especially if we want to make this extensible for
  # non-twitter data. This includes mentions and embedded
  # urls.
  # TODO - could use these as features instead??
  tweets = tweets.str.replace(mention, '')
  tweets = tweets.str.replace(url, '')

  # Clean up - remove stopwords and single-character words
  stop_words = get_stop_words('en')
  cleaned_tweets = []
  for i, tweet in enumerate(tweets):
    tweet = ' '.join(list(filter(lambda w: w not in stop_words and len(w) > 1, tweet.split())))
    cleaned_tweets.append(tweet)

  # Return two series -- reindex to get rid of old indices
  return pd.Series(cleaned_tweets), pd.Series(list(targets))

def split_data(tweets, targets, train_folds, test_folds):
  train_X = tweets[train_folds]
  train_Y = targets[train_folds]
  test_X = tweets[test_folds]
  test_Y = targets[test_folds]
  return train_X, train_Y, test_X, test_Y

def learn_features(tweets, targets, train_folds, test_folds):
  '''
  # Isolate hashtags?
  '''
  # Partition data based on folds
  train_X, train_Y, test_X, test_Y = split_data(tweets, targets, train_folds, test_folds)

  # Initialize word vectorizer
  word_vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word')

  # Create word by document matrix - creates a sparse matrix
  # This must be learned on the training data
  word_doc_train = word_vectorizer.fit_transform(train_X)
  # n x m, where n = number of rows, m = number of words

  # Find word frequencies on test data (only uses words from train)
  word_doc_test = word_vectorizer.transform(test_X)

  # Fit TF-IDF; also learned on training data
  tfidf_transformer = TfidfTransformer(use_idf=False).fit(word_doc_train)

  # Apply TF-IDF transform to training data
  # Alternatively can combine these two steps: tfidf_transformer.fit_transform(word_doc)
  word_doc_tfidf_train = tfidf_transformer.transform(word_doc_train)

  # Apply learned TF-IDF transform to test data
  word_doc_tfidf_test = tfidf_transformer.transform(word_doc_test)

  return word_doc_tfidf_train, word_doc_tfidf_test, train_Y, test_Y

def cross_validation(tweets, targets, nfolds, model, metrics):
  folds = KFold(n_splits=nfolds).split(tweets)
  output = []
  # For each cross-validation fold
  for i, fold in enumerate(folds):
    train, test = fold
    print('\t====>Fold', i)
    # Extract features
    print('\t\tExtracting features')
    train_features, test_features, train_targets, test_targets = learn_features(tweets, targets, train, test)
    # Train classifier
    print('\t\tTraining classifier')
    classifier = model().fit(train_features.toarray(), train_targets)
    # Make predictions
    print('\t\tPredicting on test fold')
    predictions = classifier.predict(test_features.toarray())
    # Evaluate
    print('\t\tEvaluating fold')
    scores = {
      'predicted': predictions,
      'actual': test_targets
    }
    for name, metric in metrics.items():
      score = metric(test_targets, predictions)
      scores[name] = score
    output.append(scores)
    print('\t\tAccuracy:', scores['accuracy'])
    print('\t\tConfusion matrix:', scores['confusion'])
    print('\t\tROC:', scores['roc'])
    print('\t\tAUC:', scores['auc'])
    print('\t\tROC_AUC:', scores['roc_auc'])
  return output

def run_all_models():
  results = []
  for name, model in models.iteritems():
    print('Training and testing:', name)
    scores = cross_validation(tweets, targets, n_folds, model, metrics)
    results[name] = scores
    print()
  return results


def plot_roc(scores):
  fp_rate, tp_rate, _ = scores['roc']
  auc = scores['roc_auc']
  plt.figure()
  plt.plot(fp_rate, tp_rate, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
  plt.plot([0, 1],[0, 1], color='navy', linestyle='--')
  print('Area under the curve:', auc)

def plot_lift(actual, predicted, nbins):
  lift_data = lift(actual, predicted, n=nbins)
  print(lift_data)

models = {
  'gaussian_naive_bayes': GaussianNB,             # ~0.6 accuracy (10k)
  'bernoulli_naive_bayes': BernoulliNB,           # ~0.7 accuracy (10k)
  'svm': svm.SVC,                                 # ~0.5 accuracy (5k) -- really effing slow, would improve with more data
  'decision_tree': tree.DecisionTreeClassifier    # ~0.68 accuracy (5k) -- also really slow
}
metrics = {
  'accuracy': metrics.accuracy_score,
  'auc': metrics.auc,
  'precision': metrics.average_precision_score,
  'confusion': metrics.confusion_matrix,
  'variance': metrics.explained_variance_score,
  'f1': metrics.f1_score,
  'logloss': metrics.log_loss,
  'mae': metrics.mean_absolute_error,
  'precision_recall': metrics.precision_recall_curve,
  'roc_auc': metrics.roc_auc_score,
  'roc': metrics.roc_curve
}
n_folds = 5

# Format X and Y
tweets, targets = format_data('../tweets_with_emoticons.csv', 5000)

# Perform cross validation
output = cross_validation(tweets, targets, 5, models['decision_tree'], metrics)

# ROC for one of the folds
plot_roc(output[0])

# Plot lift
plot_lift(output[0]['actual'], output[0]['predicted'], 30)

'''
TODO
- make separate notebooks based on steps of lecture
'''