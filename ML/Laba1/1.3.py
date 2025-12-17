import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                         y_true[non_zero_mask])) * 100

def manual_linear_regression(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    b1 = numerator / denominator  
    b0 = y_mean - b1 * X_mean     
    
    return b0, b1

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"ОЦЕНКА МОДЕЛИ: {model_name}")
    print(f"{'='*60}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    print(f"\nИнтерпретация R²:")
    if r2 > 0.8:
        print(f"  Отличное качество (объясняет {r2*100:.1f}% дисперсии)")
    elif r2 > 0.6:
        print(f"  Хорошее качество (объясняет {r2*100:.1f}% дисперсии)")
    elif r2 > 0.4:
        print(f"  Удовлетворительное качество (объясняет {r2*100:.1f}% дисперсии)")
    else:
        print(f"  Низкое качество (объясняет только {r2*100:.1f}% дисперсии)")
    
    print(f"\nИнтерпретация MAPE:")
    if mape < 10:
        print(f"  Отличная точность (ошибка {mape:.1f}%)")
    elif mape < 20:
        print(f"  Хорошая точность (ошибка {mape:.1f}%)")
    elif mape < 30:
        print(f"  Удовлетворительная точность (ошибка {mape:.1f}%)")
    else:
        print(f"  Низкая точность (ошибка {mape:.1f}%)")
    
    return mae, r2, mape

def main():
    print("\n" + "="*60)
    print("ЛИНЕЙНАЯ РЕГРЕССИЯ НА НАБОРЕ ДАННЫХ DIABETES")
    print("="*60)
    
    diabetes = datasets.load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    
    correlations = df.corr()['target'].sort_values(ascending=False)
    selected_feature = correlations.drop('target').idxmax()
    
    print(f"\nВыбранный признак: '{selected_feature}'")
    print(f"Корреляция с target: {correlations[selected_feature]:.4f}")
    
    X = df[[selected_feature]].values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    

    manual_intercept, manual_coef = manual_linear_regression(
        X_train.flatten(), y_train
    )
    y_pred_manual = manual_intercept + manual_coef * X_test.flatten()

    mae_sk, r2_sk, mape_sk = evaluate_model(
        y_test, y_pred_sklearn, "Scikit-Learn"
    )
    
    mae_man, r2_man, mape_man = evaluate_model(
        y_test, y_pred_manual, "Собственная реализация"
    )
    

    print("\n" + "="*60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Metric': ['MAE', 'R²', 'MAPE (%)'],
        'Scikit-Learn': [f"{mae_sk:.4f}", f"{r2_sk:.4f}", f"{mape_sk:.2f}"],
        'Собственная реализация': [f"{mae_man:.4f}", f"{r2_man:.4f}", f"{mape_man:.2f}"],
        'Разница': [
            f"{abs(mae_sk - mae_man):.6f}",
            f"{abs(r2_sk - r2_man):.6f}",
            f"{abs(mape_sk - mape_man):.2f}"
        ]
    })
    
    print("\nСравнение метрик:")
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("АНАЛИЗ ОШИБОК (Scikit-Learn)")
    print("="*60)
    
    errors_sk = y_test - y_pred_sklearn
    errors_man = y_test - y_pred_manual
    
    error_stats = {
        'Метрика': ['Минимальная ошибка', 'Максимальная ошибка', 
                   'Средняя ошибка', 'Стандартное отклонение',
                   'Медианная ошибка'],
        'Scikit-Learn': [
            f"{errors_sk.min():.2f}",
            f"{errors_sk.max():.2f}",
            f"{errors_sk.mean():.2f}",
            f"{errors_sk.std():.2f}",
            f"{np.median(errors_sk):.2f}"
        ],
        'Собственная реализация': [
            f"{errors_man.min():.2f}",
            f"{errors_man.max():.2f}",
            f"{errors_man.mean():.2f}",
            f"{errors_man.std():.2f}",
            f"{np.median(errors_man):.2f}"
        ]
    }
    
    print(pd.DataFrame(error_stats).to_string(index=False))
    
    print("\nПроцент предсказаний с ошибкой менее:")
    for threshold in [50, 100, 150]:
        within_thresh_sk = np.sum(np.abs(errors_sk) < threshold) / len(errors_sk) * 100
        within_thresh_man = np.sum(np.abs(errors_man) < threshold) / len(errors_man) * 100
        print(f"  {threshold} единиц: Scikit-Learn - {within_thresh_sk:.1f}%, "
              f"Ручная - {within_thresh_man:.1f}%")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(y_test, y_pred_sklearn, alpha=0.5, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', linewidth=2)
    axes[0, 0].set_title(f'Scikit-Learn (R² = {r2_sk:.3f})')
    axes[0, 0].set_xlabel('Фактические значения')
    axes[0, 0].set_ylabel('Предсказанные значения')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(y_test, y_pred_manual, alpha=0.5, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', linewidth=2)
    axes[0, 1].set_title(f'Собственная реализация (R² = {r2_man:.3f})')
    axes[0, 1].set_xlabel('Фактические значения')
    axes[0, 1].set_ylabel('Предсказанные значения')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(errors_sk, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title(f'Распределение ошибок (MAE = {mae_sk:.2f})')
    axes[1, 0].set_xlabel('Ошибка')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(errors_man, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title(f'Распределение ошибок (MAE = {mae_man:.2f})')
    axes[1, 1].set_xlabel('Ошибка')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diabetes_metrics_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\n" + "="*60)
    print("ИТОГ")
    print(f"\n1. КАЧЕСТВО МОДЕЛИ:")
    print(f"R² = {r2_sk:.4f} - модель объясняет {r2_sk*100:.1f}% дисперсии")
    print(f"MAPE = {mape_sk:.1f}% - средняя ошибка {mape_sk:.1f}%")
    print(f"\n2. СРАВНЕНИЕ РЕАЛИЗАЦИЙ:")
    print(f"   - Разница в R²: {abs(r2_sk - r2_man):.10f}")
    print(f"   - Разница в MAE: {abs(mae_sk - mae_man):.10f}")
    
    print(f"   - Уравнение: y = {sklearn_model.intercept_:.2f} + "
          f"{sklearn_model.coef_[0]:.2f} * {selected_feature}")
    plt.show()

if __name__ == "__main__":
    main()
