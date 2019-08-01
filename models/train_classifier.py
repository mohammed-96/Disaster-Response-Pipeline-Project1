import sys
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
nltk.download()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def fix_relered(col_val):
    if col_val == 2 :
        return 1
    else:
        return col_val
    
def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM disaster_response', engine)
    X = df['message']
    Y = df.drop(['id' , 'message' , 'original' , 'genre'] ,axis=1)
    Y['related'] = Y.related.apply(lambda x:fix_relered(x))
    category_names = Y.columns.tolist()
    return X ,Y , category_names


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    
    parameters = {
        'tfidf__use_idf': ( False , True),
        'clf__estimator__n_estimators': [20],
        'clf__estimator__min_samples_split': [4]
    }
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test) , columns=y_test.columns)
    total_f1_score = []
    print(classification_report(y_test, y_pred ,target_names=category_names) )
    for col in y_test.columns.unique():
        total_f1_score.append(f1_score(y_test[col].values, y_pred[col].values , average='weighted' ))
    print('avr f1 score is:' , np.mean(total_f1_score))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()