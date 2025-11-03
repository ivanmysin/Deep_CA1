import numpy as np
import sympy as sp

PI = 3.141592653589793
# Определяем переменные
t = sp.symbols('t')
a = sp.symbols('a')
b = sp.symbols('b')
r = sp.symbols('r')
w_jump = sp.symbols('w_jump')
v_avg = sp.symbols('v_avg')

w_avg = sp.Function('w_avg')
dwdt = (a * (b * v_avg - w_avg(t)) + w_jump * r)
equation = sp.Eq(w_avg(t).diff(t), dwdt)

# # Решаем уравнение
solution = sp.dsolve(equation)
print(solution)
