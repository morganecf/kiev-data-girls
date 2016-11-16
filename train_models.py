import pickle
import pandas as pd
from model import Model
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

# Training data
df = pd.read_csv('sample_tweets.csv', encoding='latin-1')

# Isolate target and tweets
subset = df.sample(n=5000)
target = subset.polarity
tweets = subset.text

def train_and_save(model_type, out):
  model = Model(model_type, tweets, target)
  model.extract_features()
  model.train()
  pickle.dump(model, open(out, 'wb'))

# Gaussian Naive Bayes
print('Training gaussian NB')
train_and_save(GaussianNB, 'gaussian_naive_bayes.pkl')

# Bernoulli Naive Bayes
print('Training bernoulli NB')
train_and_save(BernoulliNB, 'bernoulli_naive_bayes.pkl')

# SVM
print('Training SVM')
train_and_save(svm.SVC, 'svm.pkl')

# Decision Tree
print('Training decision tree')
train_and_save(tree.DecisionTreeClassifier, 'decision_tree.pkl')
