import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from xgboost import XGBClassifier

# Для Hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Настройка отображения
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*80)
print("ЛАБОРАТОРНАЯ РАБОТА: ПОДБОР ГИПЕРПАРАМЕТРОВ XGBOOST")
print("Сравнение Random Search и TPE (Hyperopt)")
print("="*80)

# ==============================================
# 1. Загрузка и подготовка данных
# ==============================================

print("\n" + "="*80)
print("1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
print("="*80)

# Загрузка данных
df = pd.read_csv('diabetes.csv')

print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

# Разделение на признаки и целевую переменную
X = df.drop('Outcome', axis=1)
y = df['Outcome']
feature_names = X.columns.tolist()

print(f"\nПризнаки: {feature_names}")
print(f"Целевая переменная: Outcome (0 = нет диабета, 1 = диабет)")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nРазделение данных:")
print(f"  Обучающая выборка: {X_train.shape[0]} записей")
print(f"  Тестовая выборка: {X_test.shape[0]} записей")
print(f"  Соотношение классов: {np.bincount(y_train)} (обучение), {np.bincount(y_test)} (тест)")

# Функция для оценки модели
def evaluate_model(model, X_test, y_test, model_name=""):
    """Оценка модели и вывод метрик"""
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'predict_time': predict_time,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# ==============================================
# 2. Базовые настройки XGBoost (для сравнения)
# ==============================================

print("\n" + "="*80)
print("2. БАЗОВАЯ МОДЕЛЬ XGBOOST (ДЛЯ СРАВНЕНИЯ)")
print("="*80)

# Базовая модель с ручными настройками из предыдущей работы
base_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

base_model = XGBClassifier(
    **base_params,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0
)

print("Обучение базовой модели...")
start_time = time.time()
base_model.fit(X_train, y_train)
base_train_time = time.time() - start_time

base_results = evaluate_model(base_model, X_test, y_test, "Базовая модель")

print(f"\nБазовые параметры XGBoost:")
for param, value in base_params.items():
    print(f"  {param}: {value}")
print(f"\nМетрики базовой модели:")
print(f"  F1-score: {base_results['f1']:.4f}")
print(f"  Accuracy: {base_results['accuracy']:.4f}")
print(f"  ROC-AUC: {base_results['roc_auc']:.4f}")
print(f"  Время обучения: {base_train_time:.2f} сек")

# ==============================================
# 3. Задание 1: Random Search
# ==============================================

print("\n" + "="*80)
print("ЗАДАНИЕ 1: RANDOM SEARCH С ИСПОЛЬЗОВАНИЕМ SCIKIT-LEARN")
print("="*80)

# Определяем пространство параметров для Random Search
param_distributions = {
    'n_estimators': [50, 100, 150, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
    'reg_lambda': [0.5, 1, 2, 5, 10, 20],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0]
}

print(f"\nПространство параметров для Random Search:")
print(f"  Всего комбинаций (теоретически): {np.prod([len(v) for v in param_distributions.values()]):,}")
print(f"  Будет протестировано: 50 случайных комбинаций")

# Создаем модель XGBoost
xgb_model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0
)

# Создаем RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=50,  # Количество случайных комбинаций
    scoring='f1',  # Используем F1-score как метрику
    cv=5,  # 5-кратная кросс-валидация
    verbose=1,
    random_state=42,
    n_jobs=-1  # Используем все ядра процессора
)

print("\nЗапуск Random Search...")
print("Это может занять несколько минут в зависимости от производительности компьютера.")
start_time = time.time()
random_search.fit(X_train, y_train)
random_search_time = time.time() - start_time

print("\nRandom Search завершен!")
print(f"Общее время выполнения: {random_search_time:.2f} сек")

# Получаем лучшую модель
best_random_model = random_search.best_estimator_
best_random_params = random_search.best_params_
best_random_score = random_search.best_score_

print(f"\nЛучшие параметры (Random Search):")
for param, value in best_random_params.items():
    print(f"  {param}: {value}")
print(f"\nЛучший F1-score (кросс-валидация): {best_random_score:.4f}")

# Оцениваем на тестовой выборке
random_results = evaluate_model(best_random_model, X_test, y_test, "Random Search")

print(f"\nМетрики на тестовой выборке (Random Search):")
print(f"  F1-score: {random_results['f1']:.4f}")
print(f"  Accuracy: {random_results['accuracy']:.4f}")
print(f"  ROC-AUC: {random_results['roc_auc']:.4f}")

# Анализ результатов Random Search
print("\nАнализ результатов Random Search:")
print(f"  Лучшие параметры отличаются от базовых:")
print(f"    n_estimators: {base_params['n_estimators']} -> {best_random_params.get('n_estimators', 'N/A')}")
print(f"    max_depth: {base_params['max_depth']} -> {best_random_params.get('max_depth', 'N/A')}")
print(f"    learning_rate: {base_params['learning_rate']} -> {best_random_params.get('learning_rate', 'N/A')}")

# Визуализация результатов Random Search
print("\nСоздание визуализаций для Random Search...")

# 1. Топ-10 комбинаций параметров из Random Search
results_df = pd.DataFrame(random_search.cv_results_)
top_results = results_df.nsmallest(10, 'rank_test_score')[['params', 'mean_test_score', 'std_test_score']]

print("\nТоп-10 комбинаций параметров (Random Search):")
for i, row in top_results.iterrows():
    print(f"\n#{int(row.name)+1}: F1 = {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    # Выводим только ключевые параметры для читаемости
    key_params = {k: v for k, v in row['params'].items() 
                  if k in ['n_estimators', 'max_depth', 'learning_rate', 'subsample']}
    for param, value in key_params.items():
        print(f"  {param}: {value}")

# ==============================================
# 4. Задание 2: TPE с использованием Hyperopt
# ==============================================

print("\n" + "="*80)
print("ЗАДАНИЕ 2: TPE (TREE-STRUCTURED PARZEN ESTIMATOR) С ИСПОЛЬЗОВАНИЕМ HYPEROPT")
print("="*80)

# Определяем пространство поиска для Hyperopt
space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 150, 200, 300, 400]),
    'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8, 9, 10]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(10)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(0.5), np.log(20)),
    'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7]),
    'gamma': hp.uniform('gamma', 0, 0.4),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.6, 1.0)
}

print(f"\nПространство параметров для Hyperopt (TPE):")
print("  Используются непрерывные распределения для некоторых параметров")
print(f"  Будет выполнено: 50 итераций оптимизации")

# Функция цели для Hyperopt
def objective(params):
    """Функция цели для оптимизации Hyperopt"""
    
    # Преобразуем параметры в нужные типы
    params_converted = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'learning_rate': float(params['learning_rate']),
        'subsample': float(params['subsample']),
        'reg_alpha': float(params['reg_alpha']),
        'reg_lambda': float(params['reg_lambda']),
        'min_child_weight': int(params['min_child_weight']),
        'gamma': float(params['gamma']),
        'colsample_bytree': float(params['colsample_bytree']),
        'colsample_bylevel': float(params['colsample_bylevel'])
    }
    
    # Создаем и обучаем модель
    model = XGBClassifier(
        **params_converted,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    
    # Используем кросс-валидацию
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, 
        scoring='f1', 
        n_jobs=-1
    )
    
    # Возвращаем средний F1-score (минимизируем 1 - F1)
    return {
        'loss': 1 - cv_scores.mean(),
        'status': STATUS_OK,
        'cv_score': cv_scores.mean(),
        'params': params_converted
    }

# Запускаем оптимизацию Hyperopt
print("\nЗапуск оптимизации Hyperopt (TPE)...")
print("Это может занять несколько минут в зависимости от производительности компьютера.")

trials = Trials()  # Для хранения истории оптимизации
start_time = time.time()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,  # Количество итераций
    trials=trials,
    rstate=np.random.RandomState(42)
)

hyperopt_time = time.time() - start_time

print("\nHyperopt (TPE) оптимизация завершена!")
print(f"Общее время выполнения: {hyperopt_time:.2f} сек")

# Получаем лучшие параметры
best_hyperopt_params = {}
for key in space.keys():
    if key in best:
        if key in ['n_estimators', 'max_depth', 'min_child_weight']:
            # Для дискретных параметров нужно получить значение из choices
            choices = space[key].pos_args[0].pos_args[1].obj
            best_hyperopt_params[key] = choices[best[key]]
        else:
            best_hyperopt_params[key] = best[key]

# Но проще взять параметры из лучшего trial
best_trial = trials.best_trial
best_hyperopt_params = best_trial['result']['params']
best_hyperopt_score = 1 - best_trial['result']['loss']  # Преобразуем обратно в F1

print(f"\nЛучшие параметры (Hyperopt TPE):")
for param, value in best_hyperopt_params.items():
    print(f"  {param}: {value}")
print(f"\nЛучший F1-score (кросс-валидация): {best_hyperopt_score:.4f}")

# Обучаем финальную модель с лучшими параметрами
best_hyperopt_model = XGBClassifier(
    **best_hyperopt_params,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("\nОбучение финальной модели с лучшими параметрами Hyperopt...")
hyperopt_train_start = time.time()
best_hyperopt_model.fit(X_train, y_train)
hyperopt_train_time = time.time() - hyperopt_train_start

# Оцениваем на тестовой выборке
hyperopt_results = evaluate_model(best_hyperopt_model, X_test, y_test, "Hyperopt TPE")

print(f"\nМетрики на тестовой выборке (Hyperopt TPE):")
print(f"  F1-score: {hyperopt_results['f1']:.4f}")
print(f"  Accuracy: {hyperopt_results['accuracy']:.4f}")
print(f"  ROC-AUC: {hyperopt_results['roc_auc']:.4f}")
print(f"  Время обучения финальной модели: {hyperopt_train_time:.2f} сек")

# Анализ истории оптимизации Hyperopt
print("\nАнализ истории оптимизации Hyperopt:")
print(f"  Всего выполнено итераций: {len(trials)}")
print(f"  Лучший результат достигнут на итерации: {trials.best_trial['tid'] + 1}")

# Визуализация прогресса Hyperopt
hyperopt_scores = [1 - trial['result']['loss'] for trial in trials.trials]
hyperopt_best_scores = [max(hyperopt_scores[:i+1]) for i in range(len(hyperopt_scores))]

# ==============================================
# 5. Задание 3: Сравнение результатов
# ==============================================

print("\n" + "="*80)
print("ЗАДАНИЕ 3: СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*80)

# Сравнительная таблица
comparison_data = {
    'Метод': ['Базовая модель', 'Random Search', 'Hyperopt TPE'],
    'F1-score (тест)': [
        base_results['f1'],
        random_results['f1'],
        hyperopt_results['f1']
    ],
    'Accuracy': [
        base_results['accuracy'],
        random_results['accuracy'],
        hyperopt_results['accuracy']
    ],
    'ROC-AUC': [
        base_results['roc_auc'],
        random_results['roc_auc'],
        hyperopt_results['roc_auc']
    ],
    'Время оптимизации (сек)': [
        base_train_time,
        random_search_time,
        hyperopt_time
    ],
    'Общее время (сек)': [
        base_train_time,
        random_search_time + random_search.best_estimator_.get_params()['n_jobs'],
        hyperopt_time + hyperopt_train_time
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nСравнительная таблица результатов:")
print(comparison_df.to_string(index=False))

# Сравнение ключевых параметров
print("\nСравнение найденных параметров:")
params_comparison = pd.DataFrame({
    'Параметр': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                 'reg_alpha', 'reg_lambda'],
    'Базовая': [
        base_params.get('n_estimators', 'N/A'),
        base_params.get('max_depth', 'N/A'),
        base_params.get('learning_rate', 'N/A'),
        base_params.get('subsample', 'N/A'),
        base_params.get('reg_alpha', 'N/A'),
        base_params.get('reg_lambda', 'N/A')
    ],
    'Random Search': [
        best_random_params.get('n_estimators', 'N/A'),
        best_random_params.get('max_depth', 'N/A'),
        best_random_params.get('learning_rate', 'N/A'),
        best_random_params.get('subsample', 'N/A'),
        best_random_params.get('reg_alpha', 'N/A'),
        best_random_params.get('reg_lambda', 'N/A')
    ],
    'Hyperopt TPE': [
        best_hyperopt_params.get('n_estimators', 'N/A'),
        best_hyperopt_params.get('max_depth', 'N/A'),
        best_hyperopt_params.get('learning_rate', 'N/A'),
        best_hyperopt_params.get('subsample', 'N/A'),
        best_hyperopt_params.get('reg_alpha', 'N/A'),
        best_hyperopt_params.get('reg_lambda', 'N/A')
    ]
})

print(params_comparison.to_string(index=False))

# ==============================================
# 6. Визуализация результатов
# ==============================================

print("\n" + "="*80)
print("6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("="*80)

# Создаем фигуру с несколькими графиками
fig = plt.figure(figsize=(16, 12))

# 1. Сравнение метрик
ax1 = plt.subplot(2, 2, 1)
methods = comparison_df['Метод']
f1_scores = comparison_df['F1-score (тест)']

bars = ax1.bar(methods, f1_scores, color=['blue', 'orange', 'green'])
ax1.set_xlabel('Метод оптимизации', fontsize=12)
ax1.set_ylabel('F1-score (тестовая выборка)', fontsize=12)
ax1.set_title('Сравнение F1-score разных методов', fontsize=14)
ax1.set_ylim([0, 1])
ax1.grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for bar, score in zip(bars, f1_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{score:.3f}', ha='center', va='bottom')

# 2. Сравнение времени оптимизации
ax2 = plt.subplot(2, 2, 2)
times = comparison_df['Время оптимизации (сек)']

bars2 = ax2.bar(methods, times, color=['blue', 'orange', 'green'])
ax2.set_xlabel('Метод оптимизации', fontsize=12)
ax2.set_ylabel('Время оптимизации (сек)', fontsize=12)
ax2.set_title('Сравнение времени оптимизации', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for bar, time_val in zip(bars2, times):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
            f'{time_val:.1f} сек', ha='center', va='bottom')

# 3. Прогресс оптимизации Hyperopt
ax3 = plt.subplot(2, 2, 3)
ax3.plot(range(1, len(hyperopt_scores)+1), hyperopt_scores, 'b-', alpha=0.5, label='F1 на каждой итерации')
ax3.plot(range(1, len(hyperopt_best_scores)+1), hyperopt_best_scores, 'r-', linewidth=2, label='Лучший F1')
ax3.set_xlabel('Номер итерации', fontsize=12)
ax3.set_ylabel('F1-score (кросс-валидация)', fontsize=12)
ax3.set_title('Прогресс оптимизации Hyperopt (TPE)', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Сравнение всех метрик
ax4 = plt.subplot(2, 2, 4)
metrics = ['F1-score', 'Accuracy', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

# Значения для каждого метода
base_metrics = [base_results['f1'], base_results['accuracy'], base_results['roc_auc']]
random_metrics = [random_results['f1'], random_results['accuracy'], random_results['roc_auc']]
hyperopt_metrics = [hyperopt_results['f1'], hyperopt_results['accuracy'], hyperopt_results['roc_auc']]

ax4.bar(x - width, base_metrics, width, label='Базовая', color='blue')
ax4.bar(x, random_metrics, width, label='Random Search', color='orange')
ax4.bar(x + width, hyperopt_metrics, width, label='Hyperopt TPE', color='green')

ax4.set_xlabel('Метрики', fontsize=12)
ax4.set_ylabel('Значение', fontsize=12)
ax4.set_title('Сравнение всех метрик качества', fontsize=14)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hyperparameter_tuning_comparison.png', dpi=150, bbox_inches='tight')
print("Графики сравнения сохранены в 'hyperparameter_tuning_comparison.png'")
plt.show()

# Дополнительная визуализация: распределение параметров в Random Search
print("\nДополнительный анализ: распределение оценок в Random Search...")

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))

# Распределение F1-score по итерациям Random Search
axes2[0, 0].hist(results_df['mean_test_score'], bins=15, alpha=0.7, color='orange', edgecolor='black')
axes2[0, 0].axvline(x=best_random_score, color='red', linestyle='--', linewidth=2, label=f'Лучший: {best_random_score:.3f}')
axes2[0, 0].set_xlabel('F1-score (кросс-валидация)', fontsize=10)
axes2[0, 0].set_ylabel('Частота', fontsize=10)
axes2[0, 0].set_title('Распределение F1-score в Random Search', fontsize=12)
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)

# Зависимость F1 от n_estimators
if 'param_n_estimators' in results_df.columns:
    axes2[0, 1].scatter(results_df['param_n_estimators'], results_df['mean_test_score'], 
                       alpha=0.6, color='blue')
    axes2[0, 1].set_xlabel('n_estimators', fontsize=10)
    axes2[0, 1].set_ylabel('F1-score', fontsize=10)
    axes2[0, 1].set_title('Зависимость F1 от n_estimators', fontsize=12)
    axes2[0, 1].grid(True, alpha=0.3)

# Зависимость F1 от max_depth
if 'param_max_depth' in results_df.columns:
    axes2[0, 2].scatter(results_df['param_max_depth'], results_df['mean_test_score'], 
                       alpha=0.6, color='green')
    axes2[0, 2].set_xlabel('max_depth', fontsize=10)
    axes2[0, 2].set_ylabel('F1-score', fontsize=10)
    axes2[0, 2].set_title('Зависимость F1 от max_depth', fontsize=12)
    axes2[0, 2].grid(True, alpha=0.3)

# Зависимость F1 от learning_rate
if 'param_learning_rate' in results_df.columns:
    axes2[1, 0].scatter(results_df['param_learning_rate'], results_df['mean_test_score'], 
                       alpha=0.6, color='red')
    axes2[1, 0].set_xlabel('learning_rate', fontsize=10)
    axes2[1, 0].set_ylabel('F1-score', fontsize=10)
    axes2[1, 0].set_title('Зависимость F1 от learning_rate', fontsize=12)
    axes2[1, 0].grid(True, alpha=0.3)

# Зависимость F1 от reg_alpha
if 'param_reg_alpha' in results_df.columns:
    axes2[1, 1].scatter(results_df['param_reg_alpha'], results_df['mean_test_score'], 
                       alpha=0.6, color='purple')
    axes2[1, 1].set_xlabel('reg_alpha', fontsize=10)
    axes2[1, 1].set_ylabel('F1-score', fontsize=10)
    axes2[1, 1].set_title('Зависимость F1 от reg_alpha', fontsize=12)
    axes2[1, 1].grid(True, alpha=0.3)

# Сравнение времени методов
axes2[1, 2].bar(['Базовая', 'Random Search', 'Hyperopt TPE'], 
               [base_train_time, random_search_time, hyperopt_time],
               color=['blue', 'orange', 'green'], alpha=0.7)
axes2[1, 2].set_xlabel('Метод', fontsize=10)
axes2[1, 2].set_ylabel('Время (сек)', fontsize=10)
axes2[1, 2].set_title('Сравнение времени выполнения', fontsize=12)
axes2[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('random_search_analysis.png', dpi=150, bbox_inches='tight')
print("Дополнительные графики анализа сохранены в 'random_search_analysis.png'")
plt.show()

# ==============================================
# 7. Итоговый отчет и выводы
# ==============================================

print("\n" + "="*80)
print("ИТОГОВЫЙ ОТЧЕТ И ВЫВОДЫ")
print("="*80)

# Определяем лучший метод
best_method_idx = np.argmax(comparison_df['F1-score (тест)'])
best_method = comparison_df.iloc[best_method_idx]['Метод']
best_f1 = comparison_df.iloc[best_method_idx]['F1-score (тест)']

print(f"\n1. ЛУЧШИЙ МЕТОД ОПТИМИЗАЦИИ: {best_method}")
print(f"   F1-score: {best_f1:.4f}")
print(f"   Улучшение относительно базовой модели: {(best_f1 - base_results['f1']):.4f}")

print(f"\n2. СРАВНЕНИЕ ВРЕМЕНИ:")
print(f"   Базовая модель: {base_train_time:.2f} сек")
print(f"   Random Search: {random_search_time:.2f} сек ({random_search_time/base_train_time:.1f}× дольше)")
print(f"   Hyperopt TPE: {hyperopt_time:.2f} сек ({hyperopt_time/base_train_time:.1f}× дольше)")

print(f"\n3. КЛЮЧЕВЫЕ ОТЛИЧИЯ В ПАРАМЕТРАХ:")
print(f"   n_estimators: База={base_params['n_estimators']}, "
      f"RS={best_random_params.get('n_estimators', 'N/A')}, "
      f"TPE={best_hyperopt_params.get('n_estimators', 'N/A')}")
print(f"   learning_rate: База={base_params['learning_rate']}, "
      f"RS={best_random_params.get('learning_rate', 'N/A'):.4f}, "
      f"TPE={best_hyperopt_params.get('learning_rate', 'N/A'):.4f}")

print(f"\n4. ЭФФЕКТИВНОСТЬ МЕТОДОВ:")
print(f"   Random Search: Проще в реализации, хорошо работает для дискретных параметров")
print(f"   Hyperopt TPE: Лучше исследует пространство, эффективнее для непрерывных параметров")
print(f"   Общее время Random Search: {random_search_time:.1f} сек")
print(f"   Общее время Hyperopt TPE: {hyperopt_time:.1f} сек")

print(f"\n5. ВЫВОДЫ И РЕКОМЕНДАЦИИ:")
print(f"   • {best_method} показал лучшие результаты на данном датасете")
print(f"   • Улучшение качества: {(best_f1 - base_results['f1'])*100:.1f}%")
print(f"   • Random Search быстрее для начального поиска хороших параметров")
print(f"   • Hyperopt TPE может найти более точные значения для непрерывных параметров")
print(f"   • Для production-систем: начать с Random Search, затем уточнить с Hyperopt")

print(f"\n6. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
print(f"   • При ограниченном времени: использовать Random Search с небольшим n_iter")
print(f"   • Для максимального качества: использовать Hyperopt TPE с большим max_evals")
print(f"   • Для больших датасетов: начать с coarse search, затем fine-tuning")
print(f"   • Всегда проверять результаты на отдельной тестовой выборке")

print(f"\n7. СОХРАНЕННЫЕ ФАЙЛЫ:")
print(f"   • hyperparameter_tuning_comparison.png - основные графики сравнения")
print(f"   • random_search_analysis.png - детальный анализ Random Search")

print("\n" + "="*80)
print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
print("="*80)

# Сохраняем результаты в CSV для дальнейшего анализа
results_summary = {
    'method': ['base', 'random_search', 'hyperopt_tpe'],
    'f1_score': [base_results['f1'], random_results['f1'], hyperopt_results['f1']],
    'accuracy': [base_results['accuracy'], random_results['accuracy'], hyperopt_results['accuracy']],
    'roc_auc': [base_results['roc_auc'], random_results['roc_auc'], hyperopt_results['roc_auc']],
    'optimization_time': [base_train_time, random_search_time, hyperopt_time],
    'best_params': [str(base_params), str(best_random_params), str(best_hyperopt_params)]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
print("\nПодробные результаты сохранены в 'hyperparameter_tuning_results.csv'")
