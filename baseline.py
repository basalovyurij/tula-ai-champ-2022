import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

# Считывание данных в DataFrame
tr = pd.read_csv('train.csv', sep=';', index_col=None,
                    dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str,
                           'AGE_CATEGORY': str, 'PATIENT_ID_COUNT': int})
# test = pd.read_csv('test.csv', sep=';', index_col=None,
#                    dtype={'PATIENT_SEX': str, 'MKB_CODE': str, 'ADRES': str, 'VISIT_MONTH_YEAR': str,
#                           'AGE_CATEGORY': str})


train = tr[~tr['VISIT_MONTH_YEAR'].isin(['09.21', '10.21', '11.21', '12.21'])]
test = tr[tr['VISIT_MONTH_YEAR'].isin(['12.21'])]
print('train', len(train), 'test', len(test))

# Отделение меток от данных
X_train = train[['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']]
y_train = train[['PATIENT_ID_COUNT']]

X_test = test[['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']]
y_test = test[['PATIENT_ID_COUNT']]

# Создание объекта данных Pool, плюсы: возможность указать какие признаки являются категориальными
pool_train = Pool(X_train, y_train,
                  cat_features=['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])
pool_test = Pool(X_test, cat_features=['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])

# Объявление CatBoostRegressor и обучение
model = CatBoostRegressor(task_type='CPU', verbose=30)
model.fit(pool_train)

# Получение ответов модели на тестовой выборке в локальном тестировании
y_pred = model.predict(pool_test)

# На локальном тестировании модель выдаёт такой результат
print("Значение метрики R2 на test: ", r2_score(y_test, y_pred))

# # Формируем sample_solution. В обучении используется весь train, ответы получаем на test
# pool_train_solution = Pool(X, y, cat_features=['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])
# pool_test_solution = Pool(test, cat_features=['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY'])
#
# model_solution = CatBoostRegressor(task_type='CPU')
# model_solution.fit(pool_train_solution)
#
# # Получение ответов
# y_pred_solution = model_solution.predict(pool_test_solution)
#
# # Формируем sample_solution для отправки на платформу
# test['PATIENT_ID_COUNT'] = y_pred_solution.astype(int)
#
# # Сохраняем в csv файл
# test.to_csv('sample_solution.csv', sep=';', index=None)
