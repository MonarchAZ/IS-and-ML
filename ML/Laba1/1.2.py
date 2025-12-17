import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#Загрузка и исследование данных
print("АНАЛИЗ НАБОРА ДАННЫХ DIABETES")
print("=" * 60)

#Загрузка данных
diabetes = datasets.load_diabetes()

#Создаем DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

print(f"\nРазмер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"\nСтолбцы (признаки): {diabetes.feature_names}")
print(f"\nПервые 5 строк данных:")
print(df.head())

#Выбор столбца для регрессии
print("\n" + "=" * 60)
print("ВЫБОР СТОЛБЦА ДЛЯ ЛИНЕЙНОЙ РЕГРЕССИИ")
print("=" * 60)

correlations = df.corr()['target'].sort_values(ascending=False)

print("\nКорреляция признаков с целевой переменной:")
for feature, corr in correlations.items():
    if feature != 'target':
        print(f"  {feature}: {corr:.4f}")

#Признак с наибольшей корреляцией
selected_feature = correlations.drop('target').idxmax()
print(f"\nВыбранный признак: '{selected_feature}'")
print(f"Корреляция с target: {correlations[selected_feature]:.4f}")

#Подготовка данных
X = df[[selected_feature]].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

#Реализация регрессии (2 способа)
print("\n" + "=" * 60)
print("РЕАЛИЗАЦИЯ ЛИНЕЙНОЙ РЕГРЕССИИ")
print("=" * 60)

#Способ 1: Scikit-Learn
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_intercept = sklearn_model.intercept_
sklearn_coef = sklearn_model.coef_[0]

y_pred_sklearn = sklearn_model.predict(X_test)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print("\n1. Scikit-Learn реализация:")
print(f"   Уравнение: y = {sklearn_intercept:.4f} + {sklearn_coef:.4f} * {selected_feature}")
print(f"   MSE: {mse_sklearn:.4f}")
print(f"   R²: {r2_sklearn:.4f}")

#Способ 2
def manual_linear_regression(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    b1 = numerator / denominator  # коэффициент
    b0 = y_mean - b1 * X_mean     # свободный член
    
    return b0, b1

manual_intercept, manual_coef = manual_linear_regression(X_train.flatten(), y_train)
y_pred_manual = manual_intercept + manual_coef * X_test.flatten()
mse_manual = mean_squared_error(y_test, y_pred_manual)
r2_manual = r2_score(y_test, y_pred_manual)

print("\n2. Собственная реализация:")
print(f"   Уравнение: y = {manual_intercept:.4f} + {manual_coef:.4f} * {selected_feature}")
print(f"   MSE: {mse_manual:.4f}")
print(f"   R²: {r2_manual:.4f}")
print("\n3. Сравнение коэффициентов:")
print(f"   Intercept разница: {abs(sklearn_intercept - manual_intercept):.10f}")
print(f"   Coefficient разница: {abs(sklearn_coef - manual_coef):.10f}")

#Создание таблицы

print("\n" + "=" * 60)
print("ТАБЛИЦА ПРЕДСКАЗАНИЙ (первые 10 строк)")

results_df = pd.DataFrame({
    f'{selected_feature}': X_test.flatten()[:10],
    'Actual': y_test[:10],
    'Predicted (Scikit-Learn)': y_pred_sklearn[:10],
    'Predicted (Ручная)': y_pred_manual[:10],
    'Error (Scikit-Learn)': y_test[:10] - y_pred_sklearn[:10],
    'Error (Ручная)': y_test[:10] - y_pred_manual[:10]
})

pd.set_option('display.float_format', lambda x: f'{x:.2f}')
print(results_df.to_string(index=False))

full_results = pd.DataFrame({
    f'{selected_feature}': X_test.flatten(),
    'Actual': y_test,
    'Predicted (Scikit-Learn)': y_pred_sklearn,
    'Predicted (Ручная)': y_pred_manual
})

full_results.to_csv('diabetes_predictions.csv', index=False, float_format='%.4f')

#Визуализация
print("\n" + "=" * 60)
print("СОЗДАНИЕ ГРАФИКОВ")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(X_train, y_train, color='blue', alpha=0.6, 
                   label='Обучающая', edgecolor='black', s=50)
axes[0, 0].scatter(X_test, y_test, color='red', alpha=0.6, 
                   label='Тестовая', edgecolor='black', s=50)
axes[0, 0].set_title('Исходные данные (разделение на выборки)')
axes[0, 0].set_xlabel(selected_feature)
axes[0, 0].set_ylabel('target')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

x_line = np.array([[X.min()], [X.max()]])
y_line_sklearn = sklearn_model.predict(x_line)
y_line_manual = manual_intercept + manual_coef * x_line.flatten()

axes[0, 1].scatter(X, y, color='blue', alpha=0.3, edgecolor='black', s=50)
axes[0, 1].plot(x_line, y_line_sklearn, color='red', linewidth=3, 
                label=f'Scikit-Learn')
axes[0, 1].plot(x_line, y_line_manual, color='green', linewidth=3, 
                linestyle='--', label=f'Ручная')
axes[0, 1].set_title('Сравнение регрессионных прямых')
axes[0, 1].set_xlabel(selected_feature)
axes[0, 1].set_ylabel('target')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(X_test, y_test - y_pred_sklearn, 
                   color='red', alpha=0.6, edgecolor='black', s=50)
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 0].set_title('Ошибки предсказаний (Scikit-Learn)')
axes[1, 0].set_xlabel(selected_feature)
axes[1, 0].set_ylabel('Ошибка (Actual - Predicted)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(X_test, y_test - y_pred_manual, 
                   color='green', alpha=0.6, edgecolor='black', s=50)
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 1].set_title('Ошибки предсказаний (Ручная реализация)')
axes[1, 1].set_xlabel(selected_feature)
axes[1, 1].set_ylabel('Ошибка (Actual - Predicted)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diabetes_regression_plots.png', dpi=150, bbox_inches='tight')

#Итог
print("\n" + "=" * 60)
print("ИТОГ")
print("=" * 60)

print(f"   - Набор данных: diabetes (sklearn)")
print(f"   - Всего признаков: {len(diabetes.feature_names)}")
print(f"   - Объем данных: {len(df)} записей")

print(f"   - {selected_feature} (корреляция: {correlations[selected_feature]:.4f})")

print(f"   - Уравнение: target = {sklearn_intercept:.2f} + {sklearn_coef:.2f} * {selected_feature}")
plt.show()
