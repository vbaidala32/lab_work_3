import numpy as np
import tkinter as tk
from tkinter import messagebox


# Таблиця значень x_i та y_i
x = np.array([2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6])
y = np.array([3.5260, 3.7820, 3.9450, 4.0430, 4.1040, 4.1550, 4.2220, 4.3310, 4.5070, 4.7750, 5.1590, 5.6830])


# Розрахунок кінцевих різниць
def finite_differences(y_vals):
    n = len(y_vals)
    delta_y = np.zeros((n, n))
    delta_y[:, 0] = y_vals
    for j in range(1, n):
        for i in range(n - j):
            delta_y[i, j] = delta_y[i + 1, j - 1] - delta_y[i, j - 1]
    return delta_y


# Визначаємо, з якого вузла брати значення для інтерполяції
def choose_node(x_vals, x_target):
    n = len(x_vals)
    mid_index = n // 2
    if x_target < x_vals[mid_index]:  # Якщо у першій половині
        i = np.argmax(x_vals > x_target) - 1
        return i, 'first'
    else:  # Якщо у другій половині
        i = np.argmax(x_vals >= x_target)
        return i, 'second'


# Обчислення похідних для обох багаточленів Ньютона
def newton_derivatives(x_vals, y_vals, x_target, h):
    delta_y = finite_differences(y_vals)

    # Вибір вузла та багаточлена
    i, method = choose_node(x_vals, x_target)

    q = (x_target - x_vals[i]) / h

    # Перший інтерполяційний багаточлен Ньютона
    if method == 'first':
        # Перша похідна
        y1 = (1 / h) * (delta_y[i, 1] + (2 * q - 1) * delta_y[i, 2] / 2 +
                        (3 * q ** 2 - 6 * q + 2) * delta_y[i, 3] / 6 +
                        (4 * q ** 3 - 18 * q ** 2 + 22 * q - 6) * delta_y[i, 4] / 24 +
                        (5 * q ** 4 - 54 * q ** 3 + 96 * q ** 2 - 60 * q + 12) * delta_y[i, 5] / 120 +
                        (6 * q ** 5 - 120 * q ** 4 + 360 * q ** 3 - 360 * q ** 2 + 120 * q - 12) * delta_y[i, 6] / 720 +
                        (7 * q ** 6 - 252 * q ** 5 + 840 * q ** 4 - 840 * q ** 3 + 420 * q ** 2 - 60 * q + 24) * delta_y[i, 7] / 5040 +
                        (8 * q ** 7 - 672 * q ** 6 + 3360 * q ** 5 - 6720 * q ** 4 + 5040 * q ** 3 - 1200 * q ** 2 + 120 * q - 24) * delta_y[i, 8] / 40320 +
                        (9 * q ** 8 - 3024 * q ** 7 + 30240 * q ** 6 - 75600 * q ** 5 + 67200 * q ** 4 - 21600 * q ** 3 + 2880 * q ** 2 - 120 * q + 24) * delta_y[i, 9] / 362880 +
                        (10 * q ** 9 - 30240 * q ** 8 + 403200 * q ** 7 - 1008000 * q ** 6 + 907200 * q ** 5 - 302400 * q ** 4 + 43200 * q ** 3 - 2880 * q ** 2 + 720 * q - 120) * delta_y[i, 10] / 3628800)

        # Друга похідна
        y2 = (1 / h ** 2) * (delta_y[i, 2] + (q - 1) * delta_y[i, 3] +
                             (6 * q ** 2 - 18 * q + 11) * delta_y[i, 4] / 12 +
                             (12 * q ** 3 - 54 * q ** 2 + 66 * q - 24) * delta_y[i, 5] / 60 +
                             (60 * q ** 4 - 360 * q ** 3 + 540 * q ** 2 - 240 * q + 24) * delta_y[i, 6] / 720 +
                             (120 * q ** 5 - 720 * q ** 4 + 1080 * q ** 3 - 480 * q ** 2 + 48) * delta_y[i, 7] / 5040)

    elif method == 'second':
        # Перша похідна
        y1 = (1 / h) * (delta_y[i - 1, 1] + (2 * q + 1) * delta_y[i - 2, 2] / 2 +
                        (3 * q ** 2 + 6 * q + 2) * delta_y[i - 3, 3] / 6 +
                        (4 * q ** 3 + 18 * q ** 2 + 22 * q + 6) * delta_y[i - 4, 4] / 24 +
                        (5 * q ** 4 + 54 * q ** 3 + 96 * q ** 2 + 60 * q + 12) * delta_y[i - 5, 5] / 120 +
                        (6 * q ** 5 + 120 * q ** 4 + 360 * q ** 3 + 360 * q ** 2 + 120 * q + 12) * delta_y[i - 6, 6] / 720 +
                        (7 * q ** 6 + 252 * q ** 5 + 840 * q ** 4 + 840 * q ** 3 + 420 * q ** 2 + 60 * q + 24) * delta_y[i - 7, 7] / 5040 +
                        (8 * q ** 7 + 672 * q ** 6 + 3360 * q ** 5 + 6720 * q ** 4 + 5040 * q ** 3 + 1200 * q ** 2 + 120 * q + 24) * delta_y[i - 8, 8] / 40320 +
                        (9 * q ** 8 + 3024 * q ** 7 + 30240 * q ** 6 + 75600 * q ** 5 + 67200 * q ** 4 + 21600 * q ** 3 + 2880 * q ** 2 + 120 * q + 24) * delta_y[i - 9, 9] / 362880 +
                        (10 * q ** 9 + 30240 * q ** 8 + 403200 * q ** 7 + 1008000 * q ** 6 + 907200 * q ** 5 + 302400 * q ** 4 + 43200 * q ** 3 + 2880 * q ** 2 + 720 * q + 120) * delta_y[i - 10, 10] / 3628800)

        # Друга похідна
        y2 = (1 / h ** 2) * (delta_y[i - 2, 2] + (q + 1) * delta_y[i - 3, 3] +
                             (6 * q ** 2 + 18 * q + 11) * delta_y[i - 4, 4] / 12 +
                             (12 * q ** 3 + 54 * q ** 2 + 66 * q + 24) * delta_y[i - 5, 5] / 60 +
                             (60 * q ** 4 + 360 * q ** 3 + 540 * q ** 2 + 240 * q + 24) * delta_y[i - 6, 6] / 720)

    return y1, y2


# Функція для обчислення похідних і виведення результатів
def calculate_derivatives():
    try:
        x_target = float(entry_x_target.get())
        h = x[1] - x[0]
        y1, y2 = newton_derivatives(x, y, x_target, h)
        result_label.config(text=f"Перша похідна: {y1:.4f}\nДруга похідна: {y2:.4f}")
    except ValueError:
        messagebox.showerror("Помилка", "Будь ласка, введіть коректне число.")


# Функція для відображення таблиці з значеннями
def show_table():
    table_window = tk.Toplevel(root)
    table_window.title("Таблиця значень x, y та Δy")

    delta_y = finite_differences(y)

    columns = ["i", "x_i", "y_i"] + [f"Δ{j}y_i" for j in range(1, len(x))]

    for col, text in enumerate(columns):
        header = tk.Label(table_window, text=text, borderwidth=1, relief="solid", padx=5, pady=5)
        header.grid(row=0, column=col)

    for i in range(len(x)):
        tk.Label(table_window, text=f"{i}", borderwidth=1, relief="solid", padx=5, pady=5).grid(row=i + 1, column=0)
        tk.Label(table_window, text=f"{x[i]:.1f}", borderwidth=1, relief="solid", padx=5, pady=5).grid(row=i + 1, column=1)
        tk.Label(table_window, text=f"{y[i]:.4f}", borderwidth=1, relief="solid", padx=5, pady=5).grid(row=i + 1, column=2)
        for j in range(1, len(x) - i):
            tk.Label(table_window, text=f"{delta_y[i, j]:.4f}", borderwidth=1, relief="solid", padx=5, pady=5).grid(row=i + 1, column=j + 2)


# Головна функція для запуску програми
def main():
    global root, entry_x_target, result_label
    # Створення графічного інтерфейсу
    root = tk.Tk()
    root.title("Інтерполяція за методом Ньютона")

    frame = tk.Frame(root)
    frame.pack(pady=10)

    label_x_target = tk.Label(frame, text="Введіть значення x:")
    label_x_target.grid(row=0, column=0, padx=5, pady=5)

    entry_x_target = tk.Entry(frame)
    entry_x_target.grid(row=0, column=1, padx=5, pady=5)

    calculate_button = tk.Button(frame, text="Обчислити похідні", command=calculate_derivatives)
    calculate_button.grid(row=1, column=0, columnspan=2, pady=10)

    result_label = tk.Label(frame, text="")
    result_label.grid(row=2, column=0, columnspan=2, pady=10)

    # Відображення таблиці після запуску
    show_table()

    root.mainloop()


if __name__ == "__main__":
    main()
