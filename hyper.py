import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import  datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

def parse_date(x):
    return pd.to_datetime(x, format="%d%b%y:%H:%M:%S")

def one_hot_encode(train, test):
    cols_to_onehot_encode = [
        #Single Characters, (class?)
        'VAR_0001', 'VAR_0005',

        #'S', 'H', 'P'
        'VAR_0283', 'VAR_0305', 'VAR_0325',

        # 'O', 'R', 'U', -1
        'VAR_0352', 'VAR_0353', 'VAR_0354',

        #VAR_1934: ['IAPS' 'RCC' 'BRANCH' 'MOBILE']
        'VAR_1934'
    ]

    for col in cols_to_onehot_encode:
        dummy_cols = pd.get_dummies(train[col]).rename(columns=lambda c: col + '_' + str(c))
        train = pd.concat([train, dummy_cols], axis=1)
        train.drop([col], axis=1)

        test_dummy_cols = pd.get_dummies(test[col]).rename(columns=lambda c: col + '_' + str(c))
        test = pd.concat([test, test_dummy_cols], axis=1)
        test.drop([col], axis=1)



def get_data():
    date_columns = [204, 75, 73, 217, 158, 159, 156, 157, 176, 177, 178, 179, 166, 167, 168, 169]

    logger.info("Loading train.csv ...")
    train = pd.read_csv("train.csv", nrows=5000, parse_dates=date_columns, date_parser = parse_date)

    logger.info("Loading test.csv ... ")
    test = pd.read_csv("test.csv",  nrows=5000, parse_dates=date_columns, date_parser = parse_date)

    logger.info("Converting Date Columns")

    for col in date_columns:
        train[train.columns[col]] = train[train.columns[col]].astype(int)
        test[test.columns[col]] = test[test.columns[col]].astype(int)

    cols_to_factorize = [
        #STATES
        'VAR_0237', 'VAR_0274',
    ]

    for col in cols_to_factorize:
        factorized = pd.factorize(np.concatenate([np.array(train[col]), np.array(test[col])]))[0]
        train[col], test[col] = factorized[: len(train[col])], factorized[len(train[col]):]

    features = train.select_dtypes(include=['float', 'int']).columns
    features = np.setdiff1d(features,['ID','target'])

    test_ids = test.ID
    y_train = train.target

    x_train = train[features]
    x_test = test[features]

    return x_train, y_train, x_test, test_ids

logger.info("Loading Data")
x_train, y_train, x_test, test_ids = get_data()


logger.info("Data Loaded")

def train_model(**kwargs):
    logger.info("Training Model %.3f" % kwargs['test_size'])
    a_train, a_test, b_train, b_test = train_test_split(x_train, y_train, test_size=kwargs['test_size'])
    xgb_params = {"objective": "binary:logistic", "max_depth": 10, "silent": 1, "eta": 0.1}
    num_rounds = 200

    dtrain = xgb.DMatrix(a_train, label=b_train)
    gbdt = xgb.train(xgb_params, dtrain, num_rounds)

    logger.info("Testing Model")
    dtest = xgb.DMatrix(a_test)
    preds = gbdt.predict(dtest)
    score = roc_auc_score(b_test, preds)
    logger.info("Score: %.4f" % score)
    return score

def compute_avg_roc(test_size, rounds=5):
    return sum([train_model(test_size=test_size) for x in range(rounds)])/rounds

scores = [ [x,compute_avg_roc(x)] for x in np.arange(0.5,0.9,0.025) ]

print scores

