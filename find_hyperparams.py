import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool, cv


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', int(1e+4))


def get_key(dt2):
    return '%02d.%02d' % (dt2 % 12 + 1, dt2 // 12)


def create_train_data(top, window, delta):
    fx = f'X_t{top}_w{window}_d{delta}.npy'
    fy = f'y_t{top}_w{window}_d{delta}.npy'
    if os.path.exists(fy):
        return np.load(fx), np.load(fy)

    # Считывание данных в DataFrame
    train = pd.read_csv('train.csv', sep=';', index_col=None,
                        dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str,
                               'AGE_CATEGORY': str, 'PATIENT_ID_COUNT': int})[:top]

    data_by_dt = dict()
    all_keys = set()
    for dt, gr in train.groupby(['VISIT_MONTH_YEAR']):
        data_by_dt[dt] = dict()
        for k, gr2 in gr.groupby(['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY'])['PATIENT_ID_COUNT']:
            data_by_dt[dt][k] = gr2.values[0]
            all_keys.add(k)
    # print('group data')

    min_month = 18 * 12
    max_month = 21 * 12 - 1 - window - delta
    X = []
    y = []
    for key in all_keys:
        for dt in range(min_month, max_month):
            vec = list(key)
            for i in range(window):
                vec.append(data_by_dt[get_key(dt + i)].get(key, 0))
            X.append(vec)
            y.append(data_by_dt[get_key(dt + window + delta)].get(key, 0))
    # print('transform X')

    np.save(fx, X)
    np.save(fy, y)
    return X, y


print('                  iterations test-R2-mean  test-R2-std  train-R2-mean  train-R2-std  test-RMSE-mean  test-RMSE-std  train-RMSE-mean  train-RMSE-std')

for window in [12, 15]:
    for delta in [4, 6]:
        X, y = create_train_data(40000, window, delta)
        cv_dataset = Pool(X, y, cat_features=list(range(4)))
        for depth in [2]:
            for learning_rate in [0.01, 0.03]:
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


# 40k
#                   iterations test-R2-mean  test-R2-std  train-R2-mean  train-R2-std  test-RMSE-mean  test-RMSE-std  train-RMSE-mean  train-RMSE-std
# window 3 delta 4 depth 2 learning_rate 0.01 999 [0.5904 0.0364 0.6761 0.0463 2.0031 0.4472 1.7999 0.2307]
# window 3 delta 4 depth 2 learning_rate 0.03 487 [0.5909 0.0331 0.7006 0.0527 1.9999 0.4347 1.7294 0.2472]
# window 3 delta 6 depth 2 learning_rate 0.01 999 [0.6163 0.0997 0.7356 0.049  1.9715 0.2537 1.6516 0.186 ]
# window 3 delta 6 depth 2 learning_rate 0.03 378 [0.6167 0.1036 0.75   0.0422 1.9698 0.266  1.6064 0.1621]
# window 6 delta 4 depth 2 learning_rate 0.01 999 [0.7065 0.1185 0.8167 0.0535 1.7152 0.283  1.3801 0.2116]
# window 6 delta 4 depth 2 learning_rate 0.03 634 [0.7182 0.1254 0.8418 0.0342 1.6723 0.28   1.2871 0.1603]
# window 6 delta 6 depth 2 learning_rate 0.01 999 [0.7253 0.0861 0.8412 0.0233 1.6796 0.3279 1.2957 0.0935]
# window 6 delta 6 depth 2 learning_rate 0.03 385 [0.7213 0.0864 0.8483 0.0189 1.6935 0.3369 1.2671 0.0793]
# window 9 delta 4 depth 2 learning_rate 0.01 999 [0.7583 0.0601 0.8734 0.0212 1.5759 0.2639 1.1453 0.063 ]
# window 9 delta 4 depth 2 learning_rate 0.03 699 [0.7618 0.0611 0.8844 0.0257 1.5646 0.2744 1.092  0.0935]
# window 9 delta 6 depth 2 learning_rate 0.01 999 [0.7567 0.0905 0.8565 0.0327 1.5092 0.3939 1.1886 0.0881]
# window 9 delta 6 depth 2 learning_rate 0.03 678 [0.7608 0.0894 0.8688 0.0311 1.499  0.4058 1.1351 0.0881]
# window 12 delta 4 depth 2 learning_rate 0.01 999 [0.693  0.1035 0.8768 0.0167 1.6851 0.1644 1.0914 0.0412]
# window 12 delta 4 depth 2 learning_rate 0.03 999 [0.7136 0.0805 0.8926 0.0234 1.6359 0.0968 1.0154 0.0773]
# window 12 delta 6 depth 2 learning_rate 0.01 999 [0.6947 0.0922 0.8469 0.0288 1.6654 0.1609 1.2273 0.1822]
# window 12 delta 6 depth 2 learning_rate 0.03 536 [0.7018 0.0977 0.8601 0.0289 1.6404 0.1417 1.1723 0.1833]
# window 15 delta 4 depth 2 learning_rate 0.01 999 [0.7608 0.063  0.8773 0.013  1.4944 0.0337 1.1013 0.0274]
# window 15 delta 4 depth 2 learning_rate 0.03 732 [0.7645 0.0629 0.8923 0.0211 1.4841 0.078  1.0279 0.0819]
# window 15 delta 6 depth 2 learning_rate 0.01 999 [0.6163 0.1522 0.8291 0.0449 1.8055 0.516  1.3214 0.2955]
# window 15 delta 6 depth 2 learning_rate 0.03 711 [0.6335 0.1434 0.883  0.0441 1.7683 0.5288 1.0736 0.2193]

# all
# iterations test-R2-mean  test-R2-std  train-R2-mean  train-R2-std  test-RMSE-mean  test-RMSE-std  train-RMSE-mean  train-RMSE-std
# window 2 delta 4 depth 2 learning_rate 0.01 999 [7.0800e-01 3.0400e-02 7.2450e-01 7.2000e-03 8.7989e+00 8.3300e-01 8.5613e+00 2.7880e-01]