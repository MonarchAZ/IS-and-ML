import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("ЗАГРУЗКА И ИССЛЕДОВАНИЕ ДАННЫХ IRIS")

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"\nИмена сортов: {iris.target_names}")
print(f"\nСоответствие кодов и имен:")
print("  0 = setosa")
print("  1 = versicolor")
print("  2 = virginica")

print("\nПервые 5 строк датасета:")
print(df.head())

print("\nКоличество записей по сортам:")
print(df['target_name'].value_counts())

print("\n" + "="*70)
print("ВИЗУАЛИЗАЦИЯ ДАННЫХ")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# График 1
colors = ['red', 'green', 'blue']
target_names = iris.target_names

for i, (color, name) in enumerate(zip(colors, target_names)):
    subset = df[df['target'] == i]
    axes[0].scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                   color=color, alpha=0.7, label=name, edgecolor='black', s=80)

axes[0].set_title('Sepal: Length vs Width')
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# График 2
for i, (color, name) in enumerate(zip(colors, target_names)):
    subset = df[df['target'] == i]
    axes[1].scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                   color=color, alpha=0.7, label=name, edgecolor='black', s=80)

axes[1].set_title('Petal: Length vs Width')
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_scatter_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("ВИЗУАЛИЗАЦИЯ ДАННЫХ SEABORN PAIRPLOT")

df_pairplot = df.copy()
df_pairplot['species'] = df_pairplot['target_name']

pairplot = sns.pairplot(df_pairplot, 
                        hue='species', 
                        vars=['sepal length (cm)', 'sepal width (cm)', 
                              'petal length (cm)', 'petal width (cm)'],
                        palette='husl',
                        plot_kws={'alpha': 0.7, 'edgecolor': 'black', 's': 60})
pairplot.fig.suptitle('Pairplot всех признаков Iris dataset', y=1.02)
plt.savefig('iris_pairplot.png', dpi=150, bbox_inches='tight')
print("Pairplot сохранен в файл: 'iris_pairplot.png'")
plt.show()

print("\n" + "="*70)
print("СОЗДАНИЕ ДВУХ ДАТАСЕТОВ ДЛЯ БИНАРНОЙ КЛАССИФИКАЦИИ")

# Датасет 1
df_setosa_versicolor = df[df['target'].isin([0, 1])].copy()
print(f"\nДатасет 1 (setosa и versicolor):")
print(f"  Размер: {df_setosa_versicolor.shape[0]} записей")
print(f"  Setosa: {sum(df_setosa_versicolor['target'] == 0)} записей")
print(f"  Versicolor: {sum(df_setosa_versicolor['target'] == 1)} записей")

# Датасет 2
df_versicolor_virginica = df[df['target'].isin([1, 2])].copy()
df_versicolor_virginica.loc[:, 'target_binary'] = df_versicolor_virginica['target'] - 1

print(f"\nДатасет 2 (versicolor и virginica):")
print(f"  Размер: {df_versicolor_virginica.shape[0]} записей")
print(f"  Versicolor (перекодировано в 0): {sum(df_versicolor_virginica['target_binary'] == 0)} записей")
print(f"  Virginica (перекодировано в 1): {sum(df_versicolor_virginica['target_binary'] == 1)} записей")

def train_and_evaluate_binary_classification(df_dataset, dataset_name, features, target_col='target'):
    print(f"БИНАРНАЯ КЛАССИФИКАЦИЯ: {dataset_name}")
    X = df_dataset[features].values
    y = df_dataset[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")
    print(f"Соотношение классов в обучающей выборке: {np.bincount(y_train)}")
    print(f"Соотношение классов в тестовой выборке: {np.bincount(y_test)}")
    
    clf = LogisticRegression(random_state=0, max_iter=200)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = clf.score(X_test, y_test)
    accuracy_custom = accuracy_score(y_test, y_pred)
    
    print(f"\nРезультаты классификации:")
    print(f"  Точность модели (score): {accuracy:.4f}")
    print(f"  Точность (accuracy_score): {accuracy_custom:.4f}")
    
    print(f"\nОтчет о классификации:")
    print(classification_report(y_test, y_pred, 
                                target_names=[f'Класс {i}' for i in np.unique(y)]))
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    print(f"Матрица ошибок:")
    print(f"[[TN FP]")
    print(f" [FN TP]] = ")
    print(cm)
    
    print(f"\nКоэффициенты модели:")
    for i, (feature, coef) in enumerate(zip(features, clf.coef_[0])):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Свободный член (intercept): {clf.intercept_[0]:.4f}")
    
    return clf, X_train, X_test, y_train, y_test, y_pred

features = ['sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)']

model1, X1_train, X1_test, y1_train, y1_test, y1_pred = train_and_evaluate_binary_classification(
    df_setosa_versicolor, "Setosa vs Versicolor", features
)

model2, X2_train, X2_test, y2_train, y2_test, y2_pred = train_and_evaluate_binary_classification(
    df_versicolor_virginica, "Versicolor vs Virginica", features, 'target_binary'
)


print("\n" + "="*70)
print("ГЕНЕРАЦИЯ И КЛАССИФИКАЦИЯ СИНТЕТИЧЕСКОГО ДАТАСЕТА")
print("="*70)

X_synth, y_synth = make_classification(
    n_samples=1000, 
    n_features=2, 
    n_redundant=0, 
    n_informative=2,
    random_state=1, 
    n_clusters_per_class=1
)

print(f"Размер сгенерированного датасета: {X_synth.shape[0]} записей, {X_synth.shape[1]} признака")
print(f"Соотношение классов: {np.bincount(y_synth)}")

# Визуализация сгенерированного датасета
plt.figure(figsize=(10, 6))
plt.scatter(X_synth[y_synth == 0, 0], X_synth[y_synth == 0, 1], 
            color='blue', alpha=0.6, label='Класс 0', edgecolor='black', s=50)
plt.scatter(X_synth[y_synth == 1, 0], X_synth[y_synth == 1, 1], 
            color='red', alpha=0.6, label='Класс 1', edgecolor='black', s=50)
plt.title('Сгенерированный датасет для бинарной классификации')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('synthetic_dataset.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("КЛАССИФИКАЦИЯ СИНТЕТИЧЕСКОГО ДАТАСЕТА")
print("="*70)

X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
    X_synth, y_synth, test_size=0.3, random_state=42, stratify=y_synth
)

print(f"Размер обучающей выборки: {X_synth_train.shape[0]}")
print(f"Размер тестовой выборки: {X_synth_test.shape[0]}")


clf_synth = LogisticRegression(random_state=0)
clf_synth.fit(X_synth_train, y_synth_train)

# Предсказание
y_synth_pred = clf_synth.predict(X_synth_test)

# Оценка точности
accuracy_synth = clf_synth.score(X_synth_test, y_synth_test)
print(f"\nТочность модели на синтетическом датасете: {accuracy_synth:.4f}")

print(f"\nОтчет о классификации:")
print(classification_report(y_synth_test, y_synth_pred, 
                            target_names=['Класс 0', 'Класс 1']))

# Матрица ошибок
cm_synth = confusion_matrix(y_synth_test, y_synth_pred)
print(f"Матрица ошибок:")
print(cm_synth)

# Коэффициенты модели
print(f"\nКоэффициенты модели:")
print(f"  Признак 1: {clf_synth.coef_[0][0]:.4f}")
print(f"  Признак 2: {clf_synth.coef_[0][1]:.4f}")
print(f"  Свободный член (intercept): {clf_synth.intercept_[0]:.4f}")

# Визуализация разделяющей границы
def plot_decision_boundary(X, y, model, title):
    h = 0.02  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], 
                color='blue', alpha=0.7, label='Класс 0', edgecolor='black')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], 
                color='red', alpha=0.7, label='Класс 1', edgecolor='black')
    plt.title(title)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('decision_boundary.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\nВизуализация разделяющей границы.")
plot_decision_boundary(X_synth_test, y_synth_test, clf_synth, 
                       'Разделяющая граница логистической регрессии')

print("\n1. РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ IRIS:")
print(f"Setosa vs Versicolor: точность = {model1.score(X1_test, y1_test):.4f}")
print(f"Versicolor vs Virginica: точность = {model2.score(X2_test, y2_test):.4f}")

print("\n2. РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ СИНТЕТИЧЕСКИХ ДАННЫХ:")
print(f"Точность: {accuracy_synth:.4f}")
