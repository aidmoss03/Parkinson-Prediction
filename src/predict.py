import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_DIR = "../data/"
FILE_NAME = "parkinsons"

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_classes': 2,
    'learning_rate': 0.01,
    'num_boost_round': 3000,
    'num_leaves': 100,
    'bagging_fraction': 0.90,
    'is_unbalance': True,
    'max_bin': 256,
    'early_stopping_rounds': 500,
}

if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR + FILE_NAME + ".csv").drop(['name'], axis=1)

    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", '', x))

    train, test = train_test_split(df, test_size=0.2, shuffle=True)

    train_x = train.drop(['status'], axis=1)
    train_y = train['status']
    test_x = test.drop(['status'], axis=1)
    test_y = test['status']

    train_data = lgb.Dataset(train_x, label=train_y)
    test_data = lgb.Dataset(test_x, label=test_y)

    estimator = lgb.train(lgb_params,
                          train_data,
                          valid_sets=[test_data],
                          )

    pred_test_y = estimator.predict(test_x)
    print(pred_test_y)

    pred_labels = list(map(np.argmax, pred_test_y))

    print(precision_score(test_y,pred_labels,average=None).mean())
    print(confusion_matrix(test_y,pred_labels))
    print(np.unique(pred_labels, return_counts=True))
