import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("\n" + "="*80)
print("ЗАГРУЗКА ДАННЫХ")
df = pd.read_csv('diabetes.csv')

print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

X = df.drop('Outcome', axis=1)
y = df['Outcome']
feature_names = X.columns.tolist()

print(f"\nПризнаки: {feature_names}")
print(f"Целевая переменная: Outcome (0 = нет диабета, 1 = диабет)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nРазделение данных:")
print(f"  Обучающая выборка: {X_train.shape[0]} записей")
print(f"  Тестовая выборка: {X_test.shape[0]} записей")
print(f"  Соотношение классов: {np.bincount(y_train)} (обучение), {np.bincount(y_test)} (тест)")

print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ RANDOM FOREST")

def evaluate_model(model, X_test, y_test, model_name):
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predict_time': predict_time,
        'y_pred': y_pred
    }

print("\n" + "-"*40)
print("ИССЛЕДОВАНИЕ ЗАВИСИМОСТИ ОТ ГЛУБИНЫ ДЕРЕВЬЕВ")

max_depths = list(range(1, 21)) + [None]  
depth_metrics = {'f1': [], 'accuracy': [], 'time': []}

for depth in max_depths:
    start_time = time.time()
    
    if depth is None:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=depth,
            random_state=42,
            n_jobs=-1
        )
    else:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=depth,
            random_state=42,
            n_jobs=-1
        )
    
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    
    depth_metrics['f1'].append(f1_score(y_test, y_pred))
    depth_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    depth_metrics['time'].append(train_time)

# Находим оптимальную глубину
if None in max_depths:
    depth_labels = [str(d) if d is not None else 'None' for d in max_depths]
else:
    depth_labels = [str(d) for d in max_depths]

optimal_depth_idx = np.argmax(depth_metrics['f1'])
optimal_depth = max_depths[optimal_depth_idx]

print(f"Оптимальная глубина: {optimal_depth}")
print(f"Максимальный F1-score: {depth_metrics['f1'][optimal_depth_idx]:.4f}")

# График зависимости от глубины
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# F1-score и глубина
ax1.plot(range(1, len(max_depths)+1), depth_metrics['f1'], 'b-o', linewidth=2, markersize=6)
ax1.axvline(x=optimal_depth_idx+1, color='r', linestyle='--', alpha=0.7)
ax1.set_xlabel('Максимальная глубина дерева', fontsize=12)
ax1.set_ylabel('F1-score', fontsize=12)
ax1.set_title('Зависимость F1-score от глубины деревьев', fontsize=14)
ax1.set_xticks(range(1, len(max_depths)+1))
ax1.set_xticklabels(depth_labels, rotation=45)
ax1.grid(True, alpha=0.3)

# Время обучения и глубина
ax2.plot(range(1, len(max_depths)+1), depth_metrics['time'], 'g-s', linewidth=2, markersize=6)
ax2.set_xlabel('Максимальная глубина дерева', fontsize=12)
ax2.set_ylabel('Время обучения (сек)', fontsize=12)
ax2.set_title('Время обучения в зависимости от глубины', fontsize=14)
ax2.set_xticks(range(1, len(max_depths)+1))
ax2.set_xticklabels(depth_labels, rotation=45)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "-"*40)
print("ИССЛЕДОВАНИЕ ЗАВИСИМОСТИ ОТ КОЛИЧЕСТВА ПРИЗНАКОВ")
print("-"*40)

max_features_options = ['sqrt', 'log2', None] + list(range(1, len(feature_names)+1))
max_features_metrics = {'f1': [], 'accuracy': [], 'time': []}

for max_feat in max_features_options:
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=optimal_depth,
        max_features=max_feat,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    
    max_features_metrics['f1'].append(f1_score(y_test, y_pred))
    max_features_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    max_features_metrics['time'].append(train_time)

# оптимальное кол-во признаков
optimal_feat_idx = np.argmax(max_features_metrics['f1'])
optimal_max_features = max_features_options[optimal_feat_idx]

print(f"Оптимальное количество признаков: {optimal_max_features}")
print(f"Максимальный F1-score: {max_features_metrics['f1'][optimal_feat_idx]:.4f}")

# График зависимости от количества признаков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x_labels = [str(mf) if mf is not None else 'None' for mf in max_features_options]
x_positions = range(len(x_labels))

ax1.bar(x_positions, max_features_metrics['f1'], alpha=0.7, color='purple')
ax1.axhline(y=max_features_metrics['f1'][optimal_feat_idx], color='r', linestyle='--', alpha=0.7)
ax1.set_xlabel('max_features', fontsize=12)
ax1.set_ylabel('F1-score', fontsize=12)
ax1.set_title('Зависимость F1-score от количества признаков', fontsize=14)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(x_positions, max_features_metrics['time'], alpha=0.7, color='orange')
ax2.set_xlabel('max_features', fontsize=12)
ax2.set_ylabel('Время обучения (сек)', fontsize=12)
ax2.set_title('Время обучения в зависимости от max_features', fontsize=14)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(x_labels, rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rf_features_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


print("\n" + "-"*40)
print("ИССЛЕДОВАНИЕ ЗАВИСИМОСТИ ОТ КОЛИЧЕСТВА ДЕРЕВЬЕВ")

n_estimators_range = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500]
trees_metrics = {'f1': [], 'accuracy': [], 'time': []}

for n_est in n_estimators_range:
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=optimal_depth,
        max_features=optimal_max_features,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    
    trees_metrics['f1'].append(f1_score(y_test, y_pred))
    trees_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    trees_metrics['time'].append(train_time)

# Находим оптимальное количество деревьев
optimal_trees_idx = np.argmax(trees_metrics['f1'])
optimal_n_estimators = n_estimators_range[optimal_trees_idx]

print(f"Оптимальное количество деревьев: {optimal_n_estimators}")
print(f"Максимальный F1-score: {trees_metrics['f1'][optimal_trees_idx]:.4f}")

# График зависимости от количества деревьев
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# F1-score vs n_estimators
ax1.plot(n_estimators_range, trees_metrics['f1'], 'b-o', linewidth=2, markersize=6)
ax1.axvline(x=optimal_n_estimators, color='r', linestyle='--', alpha=0.7)
ax1.set_xlabel('Количество деревьев (n_estimators)', fontsize=12)
ax1.set_ylabel('F1-score', fontsize=12)
ax1.set_title('Зависимость F1-score от количества деревьев', fontsize=14)
ax1.grid(True, alpha=0.3)

# Время обучения vs n_estimators
ax2.plot(n_estimators_range, trees_metrics['time'], 'g-s', linewidth=2, markersize=6)
ax2.set_xlabel('Количество деревьев (n_estimators)', fontsize=12)
ax2.set_ylabel('Время обучения (сек)', fontsize=12)
ax2.set_title('Время обучения в зависимости от n_estimators', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_trees_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Оптимальная модель Random Forest
print("\n" + "-"*40)
print("ОПТИМАЛЬНАЯ МОДЕЛЬ RANDOM FOREST")
print("-"*40)

optimal_rf = RandomForestClassifier(
    n_estimators=optimal_n_estimators,
    max_depth=optimal_depth,
    max_features=optimal_max_features,
    random_state=42,
    n_jobs=-1
)

start_time = time.time()
optimal_rf.fit(X_train, y_train)
rf_train_time = time.time() - start_time

rf_results = evaluate_model(optimal_rf, X_test, y_test, "Random Forest")

print(f"Оптимальные параметры Random Forest:")
print(f"  n_estimators: {optimal_n_estimators}")
print(f"  max_depth: {optimal_depth}")
print(f"  max_features: {optimal_max_features}")
print(f"\nМетрики Random Forest:")
print(f"  F1-score: {rf_results['f1']:.4f}")
print(f"  Accuracy: {rf_results['accuracy']:.4f}")
print(f"  Precision: {rf_results['precision']:.4f}")
print(f"  Recall: {rf_results['recall']:.4f}")
print(f"  ROC-AUC: {rf_results['roc_auc']:.4f}")
print(f"  Время обучения: {rf_train_time:.2f} сек")
print(f"  Время предсказания: {rf_results['predict_time']:.4f} сек")

print("\n" + "-"*40)
print("ЭКСПЕРИМЕНТЫ С ГИПЕРПАРАМЕТРАМИ XGBOOST")

# Определяем наборы параметров для экспериментов
param_sets = [
    # Базовые параметры
    {
        'name': 'Базовые',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.3,
        'subsample': 1.0,
        'reg_alpha': 0,
        'reg_lambda': 1
    },
    # Более глубокая модель
    {
        'name': 'Глубокая',
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1
    },
    # Модель с регуляризацией
    {
        'name': 'С регуляризацией',
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0
    },
    # Быстрая модель
    {
        'name': 'Быстрая',
        'n_estimators': 50,
        'max_depth': 3,
        'learning_rate': 0.5,
        'subsample': 1.0,
        'reg_alpha': 0,
        'reg_lambda': 1
    }
]

xgb_results = []

for params in param_sets:
    print(f"\nТестируем параметры: {params['name']}")
    print(f"  n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
          f"learning_rate={params['learning_rate']}")
    
    # Создаем и обучаем модель
    start_time = time.time()
    
    xgb_model = XGBClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    
    xgb_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Оцениваем модель
    results = evaluate_model(xgb_model, X_test, y_test, f"XGBoost {params['name']}")
    results.update({
        'name': params['name'],
        'train_time': train_time,
        'params': params
    })
    
    xgb_results.append(results)
    
    print(f"  F1-score: {results['f1']:.4f}")
    print(f"  Время обучения: {train_time:.2f} сек")

print("\n" + "-"*40)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ XGBOOST")
print("-"*40)

xgb_comparison = pd.DataFrame([{
    'Параметры': r['name'],
    'F1-score': r['f1'],
    'Accuracy': r['accuracy'],
    'Время обучения': r['train_time'],
    'Параметры детально': f"n_est={r['params']['n_estimators']}, "
                        f"depth={r['params']['max_depth']}, "
                        f"lr={r['params']['learning_rate']}"
} for r in xgb_results])

print("\nСравнение различных настроек XGBoost:")
print(xgb_comparison.to_string(index=False))

# Выбираем лучшую модель XGBoost
best_xgb_idx = np.argmax([r['f1'] for r in xgb_results])
best_xgb_result = xgb_results[best_xgb_idx]
best_xgb_params = best_xgb_result['params']

print(f"\nЛучшая модель XGBoost: {best_xgb_result['name']}")
print(f"F1-score: {best_xgb_result['f1']:.4f}")
print(f"Время обучения: {best_xgb_result['train_time']:.2f} сек")

print("\n" + "-"*40)
print("2.3. ФИНАЛЬНАЯ МОДЕЛЬ XGBOOST")
print("-"*40)

final_xgb = XGBClassifier(
    n_estimators=best_xgb_params['n_estimators'],
    max_depth=best_xgb_params['max_depth'],
    learning_rate=best_xgb_params['learning_rate'],
    subsample=best_xgb_params['subsample'],
    reg_alpha=best_xgb_params['reg_alpha'],
    reg_lambda=best_xgb_params['reg_lambda'],
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

start_time = time.time()
final_xgb.fit(X_train, y_train)
xgb_train_time = time.time() - start_time

xgb_final_results = evaluate_model(final_xgb, X_test, y_test, "Final XGBoost")

print(f"Финальные параметры XGBoost:")
print(f"  n_estimators: {best_xgb_params['n_estimators']}")
print(f"  max_depth: {best_xgb_params['max_depth']}")
print(f"  learning_rate: {best_xgb_params['learning_rate']}")
print(f"  subsample: {best_xgb_params['subsample']}")
print(f"  reg_alpha: {best_xgb_params['reg_alpha']}")
print(f"  reg_lambda: {best_xgb_params['reg_lambda']}")
print(f"\nМетрики XGBoost:")
print(f"  F1-score: {xgb_final_results['f1']:.4f}")
print(f"  Accuracy: {xgb_final_results['accuracy']:.4f}")
print(f"  Precision: {xgb_final_results['precision']:.4f}")
print(f"  Recall: {xgb_final_results['recall']:.4f}")
print(f"  ROC-AUC: {xgb_final_results['roc_auc']:.4f}")
print(f"  Время обучения: {xgb_train_time:.2f} сек")
print(f"  Время предсказания: {xgb_final_results['predict_time']:.4f} сек")

print("\n" + "="*80)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ RANDOM FOREST И XGBOOST")
print("="*80)

# Создаем таблицу сравнения
comparison_data = {
    'Метрика': ['F1-score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC', 
                'Время обучения (сек)', 'Время предсказания (сек)'],
    'Random Forest': [
        rf_results['f1'],
        rf_results['accuracy'],
        rf_results['precision'],
        rf_results['recall'],
        rf_results['roc_auc'],
        rf_train_time,
        rf_results['predict_time']
    ],
    'XGBoost': [
        xgb_final_results['f1'],
        xgb_final_results['accuracy'],
        xgb_final_results['precision'],
        xgb_final_results['recall'],
        xgb_final_results['roc_auc'],
        xgb_train_time,
        xgb_final_results['predict_time']
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nСравнение Random Forest и XGBoost:")
print(comparison_df.to_string(index=False))

# Визуализация сравнения
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Сравнение метрик
metrics_to_plot = ['F1-score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC']
rf_metrics = [rf_results['f1'], rf_results['accuracy'], rf_results['precision'], 
              rf_results['recall'], rf_results['roc_auc']]
xgb_metrics = [xgb_final_results['f1'], xgb_final_results['accuracy'], 
               xgb_final_results['precision'], xgb_final_results['recall'], 
               xgb_final_results['roc_auc']]

x = np.arange(len(metrics_to_plot))
width = 0.35

axes[0].bar(x - width/2, rf_metrics, width, label='Random Forest', alpha=0.8)
axes[0].bar(x + width/2, xgb_metrics, width, label='XGBoost', alpha=0.8)
axes[0].set_xlabel('Метрики', fontsize=12)
axes[0].set_ylabel('Значение', fontsize=12)
axes[0].set_title('Сравнение метрик качества', fontsize=14)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_to_plot, rotation=45)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Сравнение времени
time_metrics = ['Время обучения', 'Время предсказания']
rf_times = [rf_train_time, rf_results['predict_time']]
xgb_times = [xgb_train_time, xgb_final_results['predict_time']]

x2 = np.arange(len(time_metrics))
axes[1].bar(x2 - width/2, rf_times, width, label='Random Forest', alpha=0.8)
axes[1].bar(x2 + width/2, xgb_times, width, label='XGBoost', alpha=0.8)
axes[1].set_xlabel('Временные характеристики', fontsize=12)
axes[1].set_ylabel('Время (сек)', fontsize=12)
axes[1].set_title('Сравнение времени выполнения', fontsize=14)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(time_metrics)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rf_vs_xgb_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nВАЖНОСТЬ ПРИЗНАКОВ")

# Random Forest важность признаков
rf_feature_importance = pd.DataFrame({
    'Признак': feature_names,
    'Важность (RF)': optimal_rf.feature_importances_
}).sort_values('Важность (RF)', ascending=False)

print("\nВажность признаков в Random Forest:")
print(rf_feature_importance.to_string(index=False))

# XGBoost важность признаков
xgb_feature_importance = pd.DataFrame({
    'Признак': feature_names,
    'Важность (XGBoost)': final_xgb.feature_importances_
}).sort_values('Важность (XGBoost)', ascending=False)

print("\nВажность признаков в XGBoost:")
print(xgb_feature_importance.to_string(index=False))

# Визуализация важности признаков
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest
bars1 = axes[0].barh(rf_feature_importance['Признак'], rf_feature_importance['Важность (RF)'])
axes[0].set_xlabel('Важность признака', fontsize=12)
axes[0].set_title('Важность признаков - Random Forest', fontsize=14)
axes[0].invert_yaxis()
for bar, imp in zip(bars1, rf_feature_importance['Важность (RF)']):
    axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', ha='left', va='center')

# XGBoost
bars2 = axes[1].barh(xgb_feature_importance['Признак'], xgb_feature_importance['Важность (XGBoost)'])
axes[1].set_xlabel('Важность признака', fontsize=12)
axes[1].set_title('Важность признаков - XGBoost', fontsize=14)
axes[1].invert_yaxis()
for bar, imp in zip(bars2, xgb_feature_importance['Важность (XGBoost)']):
    axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', ha='left', va='center')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n5.2. МАТРИЦЫ ОШИБОК")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

rf_cm = confusion_matrix(y_test, rf_results['y_pred'])
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Матрица ошибок - Random Forest', fontsize=14)
axes[0].set_xlabel('Предсказанный класс')
axes[0].set_ylabel('Истинный класс')

# XGBoost матрица ошибок
xgb_cm = confusion_matrix(y_test, xgb_final_results['y_pred'])
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
axes[1].set_title('Матрица ошибок - XGBoost', fontsize=14)
axes[1].set_xlabel('Предсказанный класс')
axes[1].set_ylabel('Истинный класс')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ИТОГ")

print(f"\n1. ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ МОДЕЛЕЙ:")
print(f"Random Forest:")
print(f"n_estimators: {optimal_n_estimators}")
print(f"max_depth: {optimal_depth}")
print(f"max_features: {optimal_max_features}")
print(f"XGBoost:")
print(f"n_estimators: {best_xgb_params['n_estimators']}")
print(f"max_depth: {best_xgb_params['max_depth']}")
print(f"learning_rate: {best_xgb_params['learning_rate']}")
print(f"subsample: {best_xgb_params['subsample']}")

print(f"\n2. СРАВНЕНИЕ КАЧЕСТВА:")
print(f"   Random Forest - F1: {rf_results['f1']:.4f}, Accuracy: {rf_results['accuracy']:.4f}")
print(f"   XGBoost - F1: {xgb_final_results['f1']:.4f}, Accuracy: {xgb_final_results['accuracy']:.4f}")

# Определяем лучшую модель
if rf_results['f1'] > xgb_final_results['f1']:
    better_model = "Random Forest"
    f1_diff = rf_results['f1'] - xgb_final_results['f1']
elif rf_results['f1'] < xgb_final_results['f1']:
    better_model = "XGBoost"
    f1_diff = xgb_final_results['f1'] - rf_results['f1']
else:
    better_model = "Обе модели одинаковы"
    f1_diff = 0

print(f"\n3. ЛУЧШАЯ МОДЕЛЬ ПО F1-SCORE: {better_model}")
if better_model != "Обе модели одинаковы":
    print(f"   Преимущество в F1-score: {f1_diff:.4f}")

print(f"\n4. СРАВНЕНИЕ ВРЕМЕНИ:")
print(f"   Random Forest - обучение: {rf_train_time:.2f} сек, предсказание: {rf_results['predict_time']:.4f} сек")
print(f"   XGBoost - обучение: {xgb_train_time:.2f} сек, предсказание: {xgb_final_results['predict_time']:.4f} сек")

if rf_train_time < xgb_train_time:
    print(f"   Random Forest обучается быстрее на {xgb_train_time - rf_train_time:.2f} сек")
else:
    print(f"   XGBoost обучается быстрее на {rf_train_time - xgb_train_time:.2f} сек")




