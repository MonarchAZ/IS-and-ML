import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LinearRegressionAnalysis:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.X = None
        self.y = None
        self.X_column = None
        self.y_column = None
        self.b0 = None  
        self.b1 = None  
        self.predictions = None
        self.errors = None
        
    def select_columns(self):
        print("Доступные данные:")
        for i, col in enumerate(self.df.columns):
            print(f"{i+1}. {col}")
        
        while True:
            try:
                x_choice = int(input(f"\nВыбери номер столбца для X (1-{len(self.df.columns)}): ")) - 1
                if 0 <= x_choice < len(self.df.columns):
                    self.X_column = self.df.columns[x_choice]
                    break
                else:
                    print("Неверный номер. Попробуй ещё")
            except ValueError:
                print("Введи число")
        
        while True:
            try:
                y_choice = int(input(f"Выбери номер столбца для y (1-{len(self.df.columns)}): ")) - 1
                if 0 <= y_choice < len(self.df.columns) and y_choice != x_choice:
                    self.y_column = self.df.columns[y_choice]
                    break
                elif y_choice == x_choice:
                    print("Нельзя выбрать один и тот же столбец")
                else:
                    print("Неверный номер, попробуй ещё")
            except ValueError:
                print("Введи число")
        
        self.X = self.df[self.X_column].values.reshape(-1, 1)
        self.y = self.df[self.y_column].values
        
        print(f"\nВыбрано: X = '{self.X_column}', y = '{self.y_column}'")
    
    def calculate_statistics(self):
        print("\n" + "="*50)
        print("Статистическая информация")
        print("="*50)
        
        for col in [self.X_column, self.y_column]:
            data = self.df[col]
            print(f"\nСтолбец: {col}")
            print(f"  Количество: {len(data)}")
            print(f"  Минимум: {data.min():.2f}")
            print(f"  Максимум: {data.max():.2f}")
            print(f"  Среднее: {data.mean():.2f}")
            print(f"  Стандартное отклонение: {data.std():.2f}")
    
    def fit_linear_regression(self):
        X_mean = np.mean(self.X)
        y_mean = np.mean(self.y)
        
        numerator = np.sum((self.X - X_mean) * (self.y - y_mean))
        denominator = np.sum((self.X - X_mean) ** 2)
        
        self.b1 = numerator / denominator  
        self.b0 = y_mean - self.b1 * X_mean  
        
        self.predictions = self.b0 + self.b1 * self.X
        self.errors = self.y - self.predictions.flatten()
        
        print(f"Уравнение регрессии: y = {self.b0:.4f} + {self.b1:.4f} * x")
        print(f"Коэффициент детерминации (R²): {self.calculate_r2():.4f}")
        print(f"Сумма квадратов ошибок: {np.sum(self.errors**2):.4f}")
    
    def calculate_r2(self):
        ss_res = np.sum(self.errors ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def visualize_results(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        #Начальные данные
        axes[0].scatter(self.X, self.y, color='blue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel(self.X_column)
        axes[0].set_ylabel(self.y_column)
        axes[0].set_title('1. Исходные данные')
        axes[0].grid(True, alpha=0.3)
        
        #Регрессионная прямая
        axes[1].scatter(self.X, self.y, color='blue', alpha=0.7, edgecolor='black', label='Данные')
        x_range = np.array([self.X.min(), self.X.max()])
        y_range = self.b0 + self.b1 * x_range
        axes[1].plot(x_range, y_range, color='red', linewidth=3, label=f'y = {self.b0:.2f} + {self.b1:.2f}x')
        axes[1].set_xlabel(self.X_column)
        axes[1].set_ylabel(self.y_column)
        axes[1].set_title('2. Линейная регрессия')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        #Квадраты ошибок
        axes[2].scatter(self.X, self.y, color='blue', alpha=0.7, edgecolor='black')
        axes[2].plot(x_range, y_range, color='red', linewidth=2)
        
        for xi, yi, y_pred, error in zip(self.X.flatten(), self.y, self.predictions.flatten(), self.errors):
            left = min(xi, xi)
            bottom = min(yi, y_pred)
            height = abs(error)
            width = 0.2  
            
            rect = patches.Rectangle(
                (left - width/2, bottom), width, height,
                linewidth=1, edgecolor='orange', facecolor='yellow', alpha=0.3
            )
            axes[2].add_patch(rect)
            
            axes[2].plot([xi, xi], [yi, y_pred], 'g--', alpha=0.5, linewidth=0.8)
        
        axes[2].set_xlabel(self.X_column)
        axes[2].set_ylabel(self.y_column)
        axes[2].set_title('3. Квадраты ошибок')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        
        self.select_columns()
        self.calculate_statistics()
        self.fit_linear_regression()
        self.visualize_results()
        
        return self


def main():
    print("Анализ линейной регрессии")
    
    filepath = "student_scores.csv"
    
    try:
        analysis = LinearRegressionAnalysis(filepath)
        analysis.run_analysis()
        
    except FileNotFoundError:
        print(f"Файл '{filepath}' не найден")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()
