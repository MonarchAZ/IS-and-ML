import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve, auc,
                             roc_auc_score, average_precision_score)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("\n" + "="*80)
print("ЗАГРУЗКА ДАННЫХ")

df = pd.read_csv('diabetes.csv')

print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nПервые 5 строк данных:")
print(df.head())

print("\nИнформация о данных:")
print(df.info())

print("\nСтатистика по данным:")
print(df.describe())

# Проверка пропусков
print("\nПроверка пропущенных значений:")
print(df.isnull().sum())

print("\nРаспределение целевой переменной (Outcome):")
outcome_counts = df['Outcome'].value_counts()
print(outcome_counts)
print(f"Выживших: {outcome_counts.get(1, 0)} ({outcome_counts.get(1, 0)/len(df)*100:.1f}%)")
print(f"Погибших: {outcome_counts.get(0, 0)} ({outcome_counts.get(0, 0)/len(df)*100:.1f}%)")

print("\n" + "="*80)
print("ПОДГОТОВКА ДАННЫХ")
print("="*80)

X = df.drop('Outcome', axis=1)
y = df['Outcome']
feature_names = X.columns.tolist()

print(f"Признаки: {feature_names}")
print(f"Целевая переменная: Outcome (0 = нет диабета, 1 = диабет)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nРазделение данных:")
print(f"  Обучающая выборка: {X_train.shape[0]} записей")
print(f"  Тестовая выборка: {X_test.shape[0]} записей")
print(f"  Соотношение классов в обучающей выборке: {np.bincount(y_train)}")
print(f"  Соотношение классов в тестовой выборке: {np.bincount(y_test)}")


print("\n" + "="*80)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Матрица ошибок:")
    print(f"  [[TN={cm[0,0]} FP={cm[0,1]}]")
    print(f"   [FN={cm[1,0]} TP={cm[1,1]}]]")
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cm': cm
    }

print("\n" + "-"*40)
print("МОДЕЛЬ 1: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
print("-"*40)
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg_results = evaluate_model(logreg, X_train, X_test, y_train, y_test, "Логистическая регрессия")

print("\n" + "-"*40)
print("МОДЕЛЬ 2: РЕШАЮЩЕЕ ДЕРЕВО (стандартные настройки)")
print("-"*40)
tree_default = DecisionTreeClassifier(random_state=42)
tree_results = evaluate_model(tree_default, X_train, X_test, y_train, y_test, "Решающее дерево")

print("\n" + "-"*40)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("-"*40)

comparison_df = pd.DataFrame({
    'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'],
    'Логистическая регрессия': [
        logreg_results['accuracy'],
        logreg_results['precision'],
        logreg_results['recall'],
        logreg_results['f1'],
        logreg_results['roc_auc']
    ],
    'Решающее дерево': [
        tree_results['accuracy'],
        tree_results['precision'],
        tree_results['recall'],
        tree_results['f1'],
        tree_results['roc_auc']
    ]
})

print("\nСравнение метрик:")
print(comparison_df.to_string(index=False))

print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ ЗАВИСИМОСТИ МЕТРИКИ ОТ ГЛУБИНЫ ДЕРЕВА")
print("="*80)

print("\nВыбор метрики для исследования:")
print("  F1-score выбрана как компромисс между Precision и Recall.")
print("  Это важно для медицинских данных, где важны и точность, и полнота.")

# Исследование зависимости F1-score от глубины дерева
max_depths = range(1, 21)
f1_scores = []
train_scores = []

for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    # F1-score на тестовой выборке
    y_pred = tree.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)
    
    train_pred = tree.predict(X_train)
    train_f1 = f1_score(y_train, train_pred)
    train_scores.append(train_f1)

plt.figure(figsize=(12, 6))
plt.plot(max_depths, f1_scores, 'b-o', linewidth=2, markersize=8, label='Тестовая выборка (F1)')
plt.plot(max_depths, train_scores, 'r--s', linewidth=2, markersize=8, label='Обучающая выборка (F1)')
plt.xlabel('Максимальная глубина дерева', fontsize=12)
plt.ylabel('F1-score', fontsize=12)
plt.title('Зависимость F1-score от глубины решающего дерева', fontsize=14)
plt.xticks(max_depths)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# Нахождение оптимальной глубины
optimal_depth = max_depths[np.argmax(f1_scores)]
optimal_f1 = max(f1_scores)
print(f"\nОптимальная глубина дерева: {optimal_depth}")
print(f"Максимальный F1-score: {optimal_f1:.4f}")

# Отметка оптимальной глубины на графике
plt.axvline(x=optimal_depth, color='g', linestyle='--', linewidth=2, alpha=0.7)
plt.axhline(y=optimal_f1, color='g', linestyle='--', linewidth=2, alpha=0.7)
plt.scatter([optimal_depth], [optimal_f1], color='g', s=200, zorder=5, 
            label=f'Оптимум: глубина={optimal_depth}, F1={optimal_f1:.3f}')
plt.legend(fontsize=12)
plt.savefig('tree_depth_vs_f1.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("АНАЛИЗ ОПТИМАЛЬНОГО ДЕРЕВА")
print("="*80)

optimal_tree = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
optimal_tree.fit(X_train, y_train)

print("\nТекстовая визуализация дерева (первые 3 уровня):")
tree_rules = export_text(optimal_tree, feature_names=feature_names, max_depth=3)
print(tree_rules)

# Графическая визуализация
plt.figure(figsize=(20, 10))
plot_tree(optimal_tree, 
          feature_names=feature_names,
          class_names=['No Diabetes', 'Diabetes'],
          filled=True, 
          rounded=True,
          fontsize=10,
          max_depth=3)
plt.title(f'Решающее дерево (глубина={optimal_depth})', fontsize=14)
plt.tight_layout()
plt.savefig('optimal_decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()


feature_importance = pd.DataFrame({
    'Признак': feature_names,
    'Важность': optimal_tree.feature_importances_
}).sort_values('Важность', ascending=False)

print("\nВажность признаков:")
print(feature_importance.to_string(index=False))

plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance['Признак'], feature_importance['Важность'])
plt.xlabel('Важность признака', fontsize=12)
plt.title('Важность признаков в решающем дереве', fontsize=14)
plt.gca().invert_yaxis()

for bar, importance in zip(bars, feature_importance['Важность']):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
             f'{importance:.3f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

y_pred_proba = optimal_tree.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve', fontsize=14)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

axes[1].plot(recall, precision, color='red', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curve', fontsize=14)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pr_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ ПАРАМЕТРА max_features")

max_features_options = ['sqrt', 'log2', None] + list(range(1, len(feature_names)+1))
f1_by_max_features = []

for max_feat in max_features_options:
    try:
        tree = DecisionTreeClassifier(
            max_depth=optimal_depth,
            max_features=max_feat,
            random_state=42
        )
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1_by_max_features.append(f1)
    except:
        f1_by_max_features.append(0)

plt.figure(figsize=(12, 6))
x_labels = [str(mf) for mf in max_features_options]
x_positions = range(len(x_labels))

plt.bar(x_positions, f1_by_max_features, alpha=0.7, color='purple')
plt.xlabel('max_features', fontsize=12)
plt.ylabel('F1-score', fontsize=12)
plt.title('Зависимость F1-score от параметра max_features', fontsize=14)
plt.xticks(x_positions, x_labels, rotation=45)
plt.grid(True, alpha=0.3, axis='y')

optimal_idx = np.argmax(f1_by_max_features)
optimal_max_features = max_features_options[optimal_idx]
optimal_f1_features = f1_by_max_features[optimal_idx]

plt.axhline(y=optimal_f1_features, color='r', linestyle='--', linewidth=2, alpha=0.7)
plt.scatter([optimal_idx], [optimal_f1_features], color='r', s=200, zorder=5,
            label=f'Оптимум: max_features={optimal_max_features}, F1={optimal_f1_features:.3f}')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('max_features_vs_f1.png', dpi=150, bbox_inches='tight')
print(f"\nОптимальное значение max_features: {optimal_max_features}")
print(f"Максимальный F1-score: {optimal_f1_features:.4f}")
print("График сохранен в файл: 'max_features_vs_f1.png'")
plt.show()


print("\n" + "="*80)
print("ИТОГ")

print(f"\n1. СРАВНЕНИЕ МОДЕЛЕЙ:")
print(f"Логистическая регрессия: Accuracy = {logreg_results['accuracy']:.4f}, F1 = {logreg_results['f1']:.4f}")
print(f"Решающее дерево: Accuracy = {tree_results['accuracy']:.4f}, F1 = {tree_results['f1']:.4f}")

print(f"\n2. ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ДЕРЕВА:")
print(f"Глубина: {optimal_depth}")
print(f"max_features: {optimal_max_features}")
print(f"Максимальный F1-score: {optimal_f1:.4f}")

print(f"\n3. ВАЖНОСТЬ ПРИЗНАКОВ:")
for i, row in feature_importance.iterrows():
    print(f"   {i+1}. {row['Признак']}: {row['Важность']:.3f}")

print(f"\n4. КАЧЕСТВО ОПТИМАЛЬНОЙ МОДЕЛИ:")
print(f"   • ROC-AUC: {roc_auc:.4f}")
print(f"   • Average Precision: {avg_precision:.4f}")
