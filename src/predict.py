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
    'verbose': -1,
}

if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR + FILE_NAME + ".csv").drop(['name'], axis=1)

    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", '', x))

    train, test = train_test_split(df, test_size=0.2, shuffle=True)

    train_x = train.drop(['status'], axis=1)
    train_y = train['status']
    test_x = test.drop(['status'], axis=1)
    test_y = test['status']

    train_data = lgb.Dataset(train_x, label=train_y, params={'verbose': -1}, free_raw_data=False)
    test_data = lgb.Dataset(test_x, label=test_y, params={'verbose': -1}, free_raw_data=False)

    estimator = lgb.train(lgb_params,
                          train_data,
                          valid_sets=[test_data],
                          verbose_eval=False,
                          )

    pred_test_y = estimator.predict(test_x)

    pred_labels = list(map(np.argmax, pred_test_y))

    print("precision:", precision_score(test_y,pred_labels,average=None).mean())
    print("confusion matrix:", confusion_matrix(test_y,pred_labels))
    print("np.unique (counts of predicted classes):", np.unique(pred_labels, return_counts=True))
