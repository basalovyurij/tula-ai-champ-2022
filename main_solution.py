import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', int(1e+4))


def get_key(dt2):
    return '%02d.%02d' % (dt2 % 12 + 1, dt2 // 12)


def create_train_data(window, delta):
    train = pd.read_csv('train.csv', sep=';', index_col=None,
                        dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str,
                               'AGE_CATEGORY': str, 'PATIENT_ID_COUNT': int})

    data_by_dt = dict()
    all_keys = set()
    for dt, gr in train.groupby(['VISIT_MONTH_YEAR']):
        data_by_dt[dt] = dict()
        for k, gr2 in gr.groupby(['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'])['PATIENT_ID_COUNT']:
            data_by_dt[dt][k] = gr2.values[0]
            all_keys.add(k)

    min_month = 18 * 12
    max_month = 21 * 12 - window - delta
    X = []
    y = []
    for key in all_keys:
        for dt in range(min_month, max_month):
            vec = list(key)
            for i in range(window):
                vec.append(data_by_dt[get_key(dt + i)].get(key, 0))
            X.append(vec)
            y.append(data_by_dt[get_key(dt + window + delta)].get(key, 0))

    return X, y


def create_test_data(window, delta):
    train = pd.read_csv('train.csv', sep=';', index_col=None,
                        dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str,
                               'AGE_CATEGORY': str, 'PATIENT_ID_COUNT': int})

    test = pd.read_csv('test.csv', sep=';', index_col=None,
                        dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str,
                               'AGE_CATEGORY': str, 'PATIENT_ID_COUNT': int})

    data_by_dt = dict()
    for dt, gr in train.groupby(['VISIT_MONTH_YEAR']):
        data_by_dt[dt] = dict()
        for k, gr2 in gr.groupby(['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'])['PATIENT_ID_COUNT']:
            data_by_dt[dt][k] = gr2.values[0]

    all_keys = set()
    for k, gr2 in test.groupby(['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY']):
        all_keys.add(k)

    dt = 21 * 12 + 4 - window - delta
    X = []
    for key in all_keys:
        vec = list(key)
        for i in range(window):
            vec.append(data_by_dt[get_key(dt + i)].get(key, 0))
        X.append(vec)
    return X


print('                  iterations test-R2-mean  test-R2-std  train-R2-mean  train-R2-std  test-RMSE-mean  test-RMSE-std  train-RMSE-mean  train-RMSE-std')
test = pd.read_csv('test.csv', sep=';', index_col=None,
                   dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str,
                          'AGE_CATEGORY': str, 'PATIENT_ID_COUNT': int})
test['PATIENT_ID_COUNT'] = 0

for window in [8, 9, 10, 12]:
    for delta in [4]:
        X, y = create_train_data(window, delta)
        X_test = create_test_data(window, delta)

        cv_dataset = Pool(X, y, cat_features=list(range(4)))
        for depth in [2]:
            for learning_rate in [0.01]:
                params = {
                    'iterations': 1000,
                    'verbose': False,
                    'depth': depth,
                    'learning_rate': learning_rate,
                    'loss_function': 'RMSE',
                    'eval_metric': 'R2',
                    'early_stopping_rounds': 30
                }

                scores = cv(cv_dataset, params, fold_count=4, logging_level='Silent')
                vals = scores.tail(1).values[0]
                print('window', window, 'delta', delta,
                      'depth', depth, 'learning_rate', learning_rate, int(vals[0]),
                      np.round(scores.tail(1).values[0][1:], 4))

                model = CatBoostRegressor(iterations=10000, verbose=1000, depth=depth,
                                          learning_rate=learning_rate, loss_function='RMSE')
                model.fit(X, y, cat_features=list(range(4)))

                y_test = model.predict(X_test)
                y_dict = dict()
                for i in range(len(y_test)):
                    key = tuple(X_test[i][0:4])
                    y_dict[key] = y_test[i]

                for index, row in test.iterrows():
                    key = tuple(row[['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY']])
                    row['PATIENT_ID_COUNT'] = int(y_dict[key] + 0.5)

                test.to_csv(f'sample_solution{window}.csv', sep=';', index=None)
