import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

df = pd.read_csv('titanic.csv')
original_size = len(df)
print(f"   Исходный размер данных: {original_size} записей, {df.shape[1]} столбцов")

print("\n2. Первые 5 строк исходных данных:")
print(df.head())

print("\n3. Информация о данных:")
print(df.info())

print("\n4. Статистика по данным:")
print(df.describe())

#Удаление строк с пропусками
print("\n" + "-"*40)
print("УДАЛЕНИЕ СТРОК С ПРОПУСКАМИ")
print("-"*40)

#Проверяем пропуски перед удалением
missing_before = df.isnull().sum()
print("Пропуски до удаления:")
for col, count in missing_before.items():
    if count > 0:
        print(f"  {col}: {count} пропусков ({count/len(df)*100:.1f}%)")

#Удаляем строки с пропусками
df_clean = df.dropna()
rows_removed_11 = len(df) - len(df_clean)
print(f"\nУдалено строк с пропусками: {rows_removed_11}")

#Удаление нечисловых столбцов (кроме Sex и Embarked)
print("\n" + "-"*40)
print("УДАЛЕНИЕ НЕЧИСЛОВЫХ СТОЛБЦОВ")
print("-"*40)

print("Столбцы перед удалением:")
for col in df_clean.columns:
    dtype = df_clean[col].dtype
    print(f"  {col}: {dtype}")

#Удаляем нечисловые столбцы
columns_to_drop = []
for col in df_clean.columns:
    if df_clean[col].dtype == 'object' and col not in ['Sex', 'Embarked']:
        columns_to_drop.append(col)

print(f"\nУдаляемые столбцы: {columns_to_drop}")
df_clean = df_clean.drop(columns=columns_to_drop)

print("\nСтолбцы после удаления:")
for col in df_clean.columns:
    print(f"  {col}: {df_clean[col].dtype}")

print("\n" + "-"*40)
print("ПЕРЕКОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")

print("Перекодирование Sex:")
print(f"  До: {df_clean['Sex'].unique()}")
sex_mapping = {'male': 0, 'female': 1}
df_clean['Sex'] = df_clean['Sex'].map(sex_mapping)
print(f"  После: {df_clean['Sex'].unique()}")
print(f"  Соответствие: male -> 0, female -> 1")

print("\nПерекодирование Embarked:")
print(f"  До: {df_clean['Embarked'].unique()}")
embarked_mapping = {'C': 1, 'Q': 2, 'S': 3}
df_clean['Embarked'] = df_clean['Embarked'].map(embarked_mapping)
print(f"  После: {df_clean['Embarked'].unique()}")
print(f"  Соответствие: C -> 1, Q -> 2, S -> 3")

#Удаление PassengerId
print("\n" + "-"*40)
print("УДАЛЕНИЕ PassengerId")
print("-"*40)

if 'PassengerId' in df_clean.columns:
    df_clean = df_clean.drop(columns=['PassengerId'])
    print("Столбец PassengerId удален")
else:
    print("Столбец PassengerId уже удален")

print("\n" + "-"*40)
print("ВЫЧИСЛЕНИЕ ПРОЦЕНТА ПОТЕРЯННЫХ ДАННЫХ")

final_size = len(df_clean)
data_loss = (original_size - final_size) / original_size * 100
rows_removed_total = original_size - final_size

print(f"Исходный размер: {original_size} записей")
print(f"Финальный размер: {final_size} записей")
print(f"Удалено записей: {rows_removed_total}")
print(f"Потеря данных: {data_loss:.2f}%")

print("\n6. Финальный датасет:")
print(f"   Размер: {df_clean.shape[0]} записей, {df_clean.shape[1]} столбцов")
print("\n   Первые 5 строк после предобработки:")
print(df_clean.head())

print("\n7. Статистика финального датасета:")
print(df_clean.describe())

print("\n8. Распределение целевой переменной (Survived):")
survival_counts = df_clean['Survived'].value_counts()
print(survival_counts)
print(f"\n   Выживших: {survival_counts.get(1, 0)} ({survival_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
print(f"   Погибших: {survival_counts.get(0, 0)} ({survival_counts.get(0, 0)/len(df_clean)*100:.1f}%)")


#Задание 2: Машинное обучение
print("ЗАДАНИЕ 2")
print("="*80)

print("\n" + "-"*40)
print("РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
print("-"*40)

X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']

print(f"Признаки (X): {X.shape}")
print(f"Целевая переменная (y): {y.shape}")

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nОбучающая выборка: {X_train.shape[0]} записей")
print(f"Тестовая выборка: {X_test.shape[0]} записей")

print(f"\nРаспределение классов в обучающей выборке:")
print(f"  Выживших: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.1f}%)")
print(f"  Погибших: {sum(y_train == 0)} ({sum(y_train == 0)/len(y_train)*100:.1f}%)")

print(f"\nРаспределение классов в тестовой выборке:")
print(f"  Выживших: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")
print(f"  Погибших: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")

#Обучение модели логистической регрессии
print("\n" + "-"*40)
print("ОБУЧЕНИЕ МОДЕЛИ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
print("-"*40)

clf_all = LogisticRegression(random_state=42, max_iter=1000)
clf_all.fit(X_train, y_train)

print("Модель обучена со всеми признаками:")
print(f"  Коэффициенты: {clf_all.coef_[0]}")
print(f"  Свободный член: {clf_all.intercept_[0]:.4f}")

# 2.3 Оценка точности
print("\n" + "-"*40)
print("ОЦЕНКА ТОЧНОСТИ МОДЕЛИ (со всеми признаками)")
print("-"*40)

y_pred_all = clf_all.predict(X_test)

accuracy_all = accuracy_score(y_test, y_pred_all)
print(f"Точность модели: {accuracy_all:.4f} ({accuracy_all*100:.2f}%)")

print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred_all, 
                            target_names=['Погиб (0)', 'Выжил (1)']))

cm_all = confusion_matrix(y_test, y_pred_all)
print("Матрица ошибок:")
print("[[TN FP]")
print(" [FN TP]] = ")
print(cm_all)
print(f"\nИстинно отрицательные (TN): {cm_all[0, 0]}")
print(f"Ложно положительные (FP): {cm_all[0, 1]}")
print(f"Ложно отрицательные (FN): {cm_all[1, 0]}")
print(f"Истинно положительные (TP): {cm_all[1, 1]}")

print("\n" + "-"*40)
print("ОЦЕНКА ВЛИЯНИЯ ПРИЗНАКА EMBARKED НА ТОЧНОСТЬ")
print("-"*40)

X_train_no_embarked = X_train.drop('Embarked', axis=1)
X_test_no_embarked = X_test.drop('Embarked', axis=1)

clf_no_embarked = LogisticRegression(random_state=42, max_iter=1000)
clf_no_embarked.fit(X_train_no_embarked, y_train)

y_pred_no_embarked = clf_no_embarked.predict(X_test_no_embarked)
accuracy_no_embarked = accuracy_score(y_test, y_pred_no_embarked)

print(f"Точность БЕЗ признака Embarked: {accuracy_no_embarked:.4f} ({accuracy_no_embarked*100:.2f}%)")
print(f"Точность СО всеми признаками: {accuracy_all:.4f} ({accuracy_all*100:.2f}%)")
print(f"Разница: {abs(accuracy_all - accuracy_no_embarked):.4f}")


print("\n" + "="*80)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
print("="*80)

print("\n1. ВАЖНОСТЬ ПРИЗНАКОВ:")
feature_importance = pd.DataFrame({
    'Признак': X_train.columns,
    'Коэффициент': clf_all.coef_[0],
    'Абсолютное значение': np.abs(clf_all.coef_[0])
}).sort_values('Абсолютное значение', ascending=False)

print(feature_importance.to_string(index=False))

#Визуализация важности признаков
plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance['Признак'], 
                feature_importance['Абсолютное значение'])
plt.xlabel('Абсолютное значение коэффициента')
plt.title('Важность признаков в логистической регрессии')
plt.gca().invert_yaxis()  

for bar, coef in zip(bars, feature_importance['Коэффициент']):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{coef:.3f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('titanic_feature_importance.png', dpi=150, bbox_inches='tight')
print("\nГрафик важности признаков сохранен в 'titanic_feature_importance.png'")
plt.show()

print("\n2. АНАЛИЗ ОШИБОК КЛАССИФИКАЦИИ:")
errors = y_test != y_pred_all
error_indices = np.where(errors)[0]

if len(error_indices) > 0:
    print(f"Количество ошибок: {len(error_indices)}")
    
    print("\nПримеры ошибочных предсказаний (первые 5):")
    error_samples = X_test.iloc[error_indices[:5]].copy()
    error_samples['True_Label'] = y_test.iloc[error_indices[:5]].values
    error_samples['Predicted_Label'] = y_pred_all[error_indices[:5]]
    print(error_samples[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                         'True_Label', 'Predicted_Label']])

print("\n3. ПРОГНОЗ ВЕРОЯТНОСТЕЙ (первые 5 тестовых записей):")
probabilities = clf_all.predict_proba(X_test.iloc[:5])
for i, (true_label, probs) in enumerate(zip(y_test.iloc[:5], probabilities)):
    print(f"Запись {i+1}: Истинный класс={true_label}, "
          f"Вероятности: [Класс 0: {probs[0]:.3f}, Класс 1: {probs[1]:.3f}]")

print("ИТОГ")
print(f"\n1. ПРЕДОБРАБОТКА ДАННЫХ:")
print(f"Исходный размер: {original_size} записей")
print(f"Финальный размер: {final_size} записей")
print(f"Потеря данных: {data_loss:.2f}%")
print(f"Сохраненные признаки: {list(X.columns)}")

print(f"\n2. РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ:")
print(f"Точность модели (со всеми признаками): {accuracy_all:.4f}")
print(f"Точность (без Embarked): {accuracy_no_embarked:.4f}")
print(f"Влияние Embarked: {accuracy_all - accuracy_no_embarked:.4f}")

print(f"\n3. ВАЖНОСТЬ ПРИЗНАКОВ (по убыванию):")
for i, row in feature_importance.iterrows():
    sign = "+" if row['Коэффициент'] > 0 else "-"
    print(f"   {i+1}. {row['Признак']}: {sign}{abs(row['Коэффициент']):.3f}")
