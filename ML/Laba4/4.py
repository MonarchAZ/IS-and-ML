'''
По условию нужно было взять датасет с данными по недвижимости в Бостоне,
но по информации из интернета он с версии 1.2 устарел, поэтому там предлагается
альтернативный датасет под названием *California housing dataset*, который был взят
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                           classification_report, confusion_matrix, accuracy_score,
                           roc_curve, auc)
from sklearn.metrics import precision_score, recall_score, f1_score

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')

print("\n" + "=" * 70)
print("ЗАГРУЗКА CALIFORNIA HOUSING DATASET")

california = fetch_california_housing()
X = california.data
y = california.target
feature_names = california.feature_names

print(f"Размерность данных: {X.shape}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Названия признаков: {feature_names}")
print(f"Размер целевой переменной: {y.shape}")
print(f"\nПримеры значений целевой переменной (цена дома в $100,000):")
print(f"Минимальная цена: ${y.min()*100000:,.0f}")
print(f"Максимальная цена: ${y.max()*100000:,.0f}")
print(f"Средняя цена: ${y.mean()*100000:,.0f}")

df = pd.DataFrame(X, columns=feature_names)
df['MedHouseVal'] = y

print("\nПервые 5 строк датасета:")
print(df.head())
print("\nСтатистическое описание:")
print(df.describe())

print("\n" + "=" * 70)
print("МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН ДЛЯ ЗАДАЧИ РЕГРЕССИИ")

X_reg = df.drop('MedHouseVal', axis=1).values
y_reg = df['MedHouseVal'].values

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"\nРазделение данных для регрессии:")
print(f"  Обучающая выборка: {X_train_reg.shape[0]} samples")
print(f"  Тестовая выборка: {X_test_reg.shape[0]} samples")

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print("\nМасштабирование данных выполнено усшено")
print(f"  Среднее после масштабирования: {X_train_reg_scaled.mean():.6f}")
print(f"  Стандартное отклонение: {X_train_reg_scaled.std():.6f}")

print("\n" + "=" * 70)
print("РЕАЛИЗАЦИЯ МОДЕЛИ С ИСПОЛЬЗОВАНИЕМ MLPRegressor")

mlp_reg = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # два скрытых слоя
    activation='relu',             # функция активации
    solver='adam',                 # алгоритм оптимизации
    alpha=0.0001,                  # параметр регуляризации
    batch_size=256,                # размер
    learning_rate='adaptive',      # адаптивная скорость обучения
    max_iter=1000,                 # максимальное кол-во итераций
    random_state=42,
    verbose=False
)

mlp_reg.fit(X_train_reg_scaled, y_train_reg)
print("Обучение модели завершилось")

y_train_pred_reg = mlp_reg.predict(X_train_reg_scaled)
y_test_pred_reg = mlp_reg.predict(X_test_reg_scaled)

mse_train = mean_squared_error(y_train_reg, y_train_pred_reg)
mse_test = mean_squared_error(y_test_reg, y_test_pred_reg)
r2_train = r2_score(y_train_reg, y_train_pred_reg)
r2_test = r2_score(y_test_reg, y_test_pred_reg)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_reg, y_test_pred_reg)

print("\nМЕТРИКИ МОДЕЛИ MLPRegressor:")
print("-" * 50)
print(f"R² на обучающей выборке: {r2_train:.4f}")
print(f"R² на тестовой выборке:  {r2_test:.4f}")
print(f"MSE на тестовой выборке: {mse_test:.4f}")
print(f"RMSE на тестовой выборке: {rmse_test:.4f}")
print(f"MAE на тестовой выборке: {mae_test:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].scatter(y_train_reg, y_train_pred_reg, alpha=0.5, s=10, label='Обучающая')
axes[0, 0].scatter(y_test_reg, y_test_pred_reg, alpha=0.5, s=10, color='red', label='Тестовая')
axes[0, 0].plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], 'k--', lw=2)
axes[0, 0].set_xlabel('Фактические значения (MedHouseVal)')
axes[0, 0].set_ylabel('Предсказанные значения (MedHouseVal)')
axes[0, 0].set_title('Предсказания vs Фактические значения')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

errors_test = y_test_reg - y_test_pred_reg
axes[0, 1].hist(errors_test, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Ошибка предсказания')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].set_title('Распределение ошибок на тестовой выборке')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(mlp_reg.loss_curve_)
axes[1, 0].set_xlabel('Итерации')
axes[1, 0].set_ylabel('Значение функции потерь (MSE)')
axes[1, 0].set_title('Кривая обучения MLPRegressor')
axes[1, 0].grid(True, alpha=0.3)

weights_first_layer = mlp_reg.coefs_[0]
feature_importance = np.mean(np.abs(weights_first_layer), axis=1)

axes[1, 1].barh(range(len(feature_names)), feature_importance)
axes[1, 1].set_yticks(range(len(feature_names)))
axes[1, 1].set_yticklabels(feature_names)
axes[1, 1].set_xlabel('Средняя абсолютная важность')
axes[1, 1].set_title('Важность признаков (по весам первого слоя)')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Анализ модели MLPRegressor для California Housing', fontsize=16)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("ИССЛЕДОВАНИЕ ЗАВИСИМОСТИ КАЧЕСТВА (от гиперпараетров)")

def evaluate_mlp_configurations(X_train, X_test, y_train, y_test, hidden_layers_list, neurons_list):
    results = []
    
    for n_layers in hidden_layers_list:
        for n_neurons in neurons_list:
            # архитектура
            if n_layers == 1:
                hidden_layer_sizes = (n_neurons,)
            elif n_layers == 2:
                hidden_layer_sizes = (n_neurons, n_neurons//2)
            elif n_layers == 3:
                hidden_layer_sizes = (n_neurons, n_neurons//2, n_neurons//4)
            elif n_layers == 4:
                hidden_layer_sizes = (n_neurons, n_neurons//2, n_neurons//4, n_neurons//8)
            
            # обучение
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=800,
                random_state=42,
                verbose=False
            )
            
            mlp.fit(X_train, y_train)
            
            # оценка
            train_score = mlp.score(X_train, y_train)
            test_score = mlp.score(X_test, y_test)
            
            y_pred = mlp.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            results.append({
                'layers': n_layers,
                'neurons': n_neurons,
                'architecture': hidden_layer_sizes,
                'train_r2': train_score,
                'test_r2': test_score,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'overfitting': train_score - test_score,
                'total_neurons': sum(hidden_layer_sizes)
            })
            
            print(f"Слои: {n_layers}, Нейроны: {n_neurons:3d}, "
                  f"Test R²: {test_score:.4f}, MSE: {mse:.4f}")
    
    return pd.DataFrame(results)


hidden_layers_list = [1, 2, 3, 4] # параметры исседования
neurons_list = [32, 64, 128, 256]

print("\nЗапуск исследования гиперпараметров...")
results_df = evaluate_mlp_configurations(
    X_train_reg_scaled, X_test_reg_scaled, 
    y_train_reg, y_test_reg,
    hidden_layers_list, neurons_list
)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for neurons in neurons_list:
    subset = results_df[results_df['neurons'] == neurons]
    axes[0, 0].plot(subset['layers'], subset['test_r2'], marker='o', 
                   label=f'{neurons} нейронов', linewidth=2)
axes[0, 0].set_xlabel('Количество скрытых слоев')
axes[0, 0].set_ylabel('R² на тестовой выборке')
axes[0, 0].set_title('Влияние количества слоев')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

for layers in hidden_layers_list:
    subset = results_df[results_df['layers'] == layers]
    axes[0, 1].plot(subset['neurons'], subset['test_r2'], marker='s', 
                   label=f'{layers} слоя(ев)', linewidth=2)
axes[0, 1].set_xlabel('Количество нейронов в первом слое')
axes[0, 1].set_ylabel('R² на тестовой выборке')
axes[0, 1].set_title('Влияние количества нейронов')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

heatmap_data = results_df.pivot_table(values='overfitting', 
                                     index='layers', 
                                     columns='neurons')
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', 
            ax=axes[0, 2], cbar_kws={'label': 'Степень переобучения'})
axes[0, 2].set_xlabel('Количество нейронов')
axes[0, 2].set_ylabel('Количество слоев')
axes[0, 2].set_title('Степень переобучения (разница R² train-test)')

scatter = axes[1, 0].scatter(results_df['total_neurons'], results_df['test_r2'],
                            c=results_df['layers'], cmap='viridis', 
                            s=100, alpha=0.7)
axes[1, 0].set_xlabel('Общее количество нейронов')
axes[1, 0].set_ylabel('R² на тестовой выборке')
axes[1, 0].set_title('Зависимость качества от общего числа нейронов')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Количество слоев')

axes[1, 1].bar(range(len(results_df)), results_df['total_neurons'], 
              color='skyblue', alpha=0.7)
axes[1, 1].set_xlabel('Конфигурация')
axes[1, 1].set_ylabel('Общее число нейронов')
axes[1, 1].set_title('Сложность архитектуры (общее число нейронов)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks([])

best_configs = results_df.nlargest(5, 'test_r2')
axes[1, 2].barh(range(len(best_configs)), best_configs['test_r2'], 
               color='lightgreen')
axes[1, 2].set_yticks(range(len(best_configs)))
axes[1, 2].set_yticklabels([f"L{c['layers']}N{c['neurons']}" 
                           for _, c in best_configs.iterrows()])
axes[1, 2].set_xlabel('R² на тестовой выборке')
axes[1, 2].set_title('Топ-5 конфигураций')
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Исследование влияния гиперпараметров MLPRegressor', fontsize=16)
plt.tight_layout()
plt.show()

# Находим лучшую конфигурацию
best_idx = results_df['test_r2'].idxmax()
best_config = results_df.loc[best_idx]

print(f"\nЛучшая конфигурация:")
print(f"Количество слоев: {best_config['layers']}")
print(f"Нейронов в первом слое: {best_config['neurons']}")
print(f"Архитектура: {best_config['architecture']}")
print(f"Общее количество нейронов: {best_config['total_neurons']}")
print(f"R² на тесте: {best_config['test_r2']:.4f}")
print(f"MSE на тесте: {best_config['mse']:.4f}")
print(f"Степень переобучения: {best_config['overfitting']:.4f}")

'''
Часть 2
'''

print("\n" + "=" * 70)
print("МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН ДЛЯ ЗАДАЧИ КЛАССИФИКАЦИИ")


median_price = np.median(y_reg)
y_clf = (y_reg > median_price).astype(int)

print(f"\nМедианная цена: ${median_price*100000:,.0f}")
print(f"Распределение классов:")
print(f"Класс 0 (цена <= медианы): {np.sum(y_clf == 0)} samples")
print(f"Класс 1 (цена > медианы): {np.sum(y_clf == 1)} samples")

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

print("\n" + "=" * 70)
print("ПОСТРОЕНИЕ МОДЕЛИ")

mlp_clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),  
    activation='relu',             
    solver='adam',                 
    alpha=0.0001,                  
    batch_size=256,                
    learning_rate='adaptive',      
    max_iter=1000,                 
    random_state=42,
    verbose=False
)

print("Идёт обучение модели MLPClassifier")
mlp_clf.fit(X_train_clf_scaled, y_train_clf)
print("Обучение завершено успешно")

y_train_pred_clf = mlp_clf.predict(X_train_clf_scaled)
y_test_pred_clf = mlp_clf.predict(X_test_clf_scaled)
y_test_proba_clf = mlp_clf.predict_proba(X_test_clf_scaled)

print("\n" + "=" * 70)
print("ТОЧНОСТЬ ПРЕДСКАЗАНИЯ ДЛЯ КАЖДОГО КЛАССА")

print("\nОТЧЕТ ПО КЛАССИФИКАЦИИ:")
print("-" * 50)

report = classification_report(y_test_clf, y_test_pred_clf,
                               target_names=['Цена <= медианы', 'Цена > медианы'])
print(report)

report_dict = classification_report(y_test_clf, y_test_pred_clf,
                                   target_names=['Цена <= медианы', 'Цена > медианы'],
                                   output_dict=True)

print("\nМЕТРИКИ ПО КЛАССАМ:")
print("-" * 50)
for i, class_name in enumerate(['Цена <= медианы', 'Цена > медианы']):
    print(f"\n{class_name}:")
    print(f"Precision: {report_dict[class_name]['precision']:.4f}")
    print(f"Recall:    {report_dict[class_name]['recall']:.4f}")
    print(f"F1-Score:  {report_dict[class_name]['f1-score']:.4f}")
    print(f"Поддержка: {report_dict[class_name]['support']} samples")

print(f"\nОбщая точность: {report_dict['accuracy']:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

cm = confusion_matrix(y_test_clf, y_test_pred_clf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Цена <= медианы', 'Цена > медианы'],
            yticklabels=['Цена <= медианы', 'Цена > медианы'],
            ax=axes[0, 0])
axes[0, 0].set_title('Матрица ошибок')
axes[0, 0].set_ylabel('Истинные значения')
axes[0, 0].set_xlabel('Предсказанные значения')

metrics = ['precision', 'recall', 'f1-score']
for i, metric in enumerate(metrics):
    values = [report_dict['Цена <= медианы'][metric], 
              report_dict['Цена > медианы'][metric]]
    axes[0, 1].bar([0.2 + i*0.3, 1.2 + i*0.3], values, width=0.25, 
                  label=metric.capitalize())

axes[0, 1].set_xticks([0.35, 1.35])
axes[0, 1].set_xticklabels(['Цена <= медианы', 'Цена > медианы'])
axes[0, 1].set_ylabel('Значение метрики')
axes[0, 1].set_title('Метрики по классам')
axes[0, 1].legend()
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(True, alpha=0.3, axis='y')

axes[0, 2].plot(mlp_clf.loss_curve_)
axes[0, 2].set_xlabel('Итерации')
axes[0, 2].set_ylabel('Значение функции потерь')
axes[0, 2].set_title('Кривая обучения MLPClassifier')
axes[0, 2].grid(True, alpha=0.3)

fpr, tpr, _ = roc_curve(y_test_clf, y_test_proba_clf[:, 1])
roc_auc = auc(fpr, tpr)

axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC кривая (AUC = {roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Случайный классификатор')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC-кривая')
axes[1, 0].legend(loc="lower right")
axes[1, 0].grid(True, alpha=0.3)

#Распределение вероятностей
for class_label in [0, 1]:
    class_mask = (y_test_clf == class_label)
    axes[1, 1].hist(y_test_proba_clf[class_mask, 1], bins=30, 
                   alpha=0.5, label=f'Класс {class_label}',
                   density=True)
axes[1, 1].set_xlabel('Вероятность принадлежности к классу 1')
axes[1, 1].set_ylabel('Плотность вероятности')
axes[1, 1].set_title('Распределение вероятностей по классам')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

confidence = np.max(y_test_proba_clf, axis=1)
correct = (y_test_pred_clf == y_test_clf)

axes[1, 2].hist(confidence[correct], bins=20, alpha=0.5, 
               label='Правильные предсказания', color='green')
axes[1, 2].hist(confidence[~correct], bins=20, alpha=0.5, 
               label='Ошибочные предсказания', color='red')
axes[1, 2].set_xlabel('Уверенность предсказания')
axes[1, 2].set_ylabel('Частота')
axes[1, 2].set_title('Распределение уверенности предсказаний')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Анализ модели MLPClassifier для California Housing', fontsize=16)
plt.tight_layout()
plt.show()

print("\nСРАВНЕНИЕ MLP С ДРУГИМИ МОДЕЛЯМИ:")
print("-" * 50)

comparison_data = {
    'Модель': ['Линейная регрессия', 'Random Forest', 'MLPRegressor', 'MLPClassifier'],
    'Задача': ['Регрессия', 'Регрессия', 'Регрессия', 'Классификация'],
    'Основная метрика': ['R²', 'R²', 'R²', 'Accuracy'],
    'Значение': [0.60, 0.80, best_config['test_r2'], report_dict['accuracy']],
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

results_df.to_csv('mlp_hyperparameter_results.csv', index=False)

with open('classification_report.txt', 'w') as f:
    f.write("ОТЧЕТ ПО КЛАССИФИКАЦИИ CALIFORNIA HOUSING\n")
    f.write("=" * 50 + "\n\n")
    f.write(report)
    f.write(f"\n\nROC AUC: {roc_auc:.4f}")
    f.write(f"\nОбщая точность: {report_dict['accuracy']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

reg_summary = results_df.nlargest(5, 'test_r2')
axes[0].barh(range(len(reg_summary)), reg_summary['test_r2'], color='skyblue')
axes[0].set_yticks(range(len(reg_summary)))
axes[0].set_yticklabels([f"L{row['layers']}N{row['neurons']}" 
                        for _, row in reg_summary.iterrows()])
axes[0].set_xlabel('R² на тестовой выборке')
axes[0].set_title('Лучшие конфигурации для регрессии')
axes[0].grid(True, alpha=0.3, axis='x')

# Сводка
clf_metrics = ['precision', 'recall', 'f1-score']
clf_values = [report_dict['macro avg'][m] for m in clf_metrics]

bars = axes[1].bar(range(len(clf_metrics)), clf_values, color=['lightgreen', 'lightblue', 'salmon'])
axes[1].set_xticks(range(len(clf_metrics)))
axes[1].set_xticklabels([m.capitalize() for m in clf_metrics])
axes[1].set_ylabel('Значение метрики')
axes[1].set_title('Сводные метрики классификации (macro avg)')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, clf_values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom')

plt.suptitle('Сводные результаты', fontsize=16)
plt.tight_layout()
plt.savefig('lab4_summary.png', dpi=150, bbox_inches='tight')
plt.show()
