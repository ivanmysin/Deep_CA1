import sympy as sp

PI = 3.141592653589793
# Определяем переменные
t = sp.symbols('t')
alpha = sp.symbols('alpha')
Delta_eta = sp.symbols('Delta_eta')
#v_avg = sp.symbols('v_avg')
g_syn_tot = sp.symbols('g_syn_tot')
I_ext = sp.symbols('I_ext')
w_avg = sp.symbols('w_avg')
Isyn = sp.symbols('Isyn')
r = sp.symbols('r')




# Определяем уравнение
#r = sp.Function('r')
# dr/dt = (Delta_eta / PI + 2 * rates * v_avg - (alpha + g_syn_tot) * rates)
# drdt = Delta_eta / PI + 2.0 * r(t) * v_avg - (alpha + g_syn_tot) * r(t)
# equation = sp.Eq(r(t).diff(t), drdt)

v_avg = sp.Function('v_avg')
# dv/dt = v_avg(t)**2 - alpha * v_avg - w_avg + self.I_ext + Isyn - (PI * rates) ** 2
dvdt =  v_avg(t)*v_avg(t) - alpha * v_avg(t) - w_avg + I_ext + Isyn - (PI * r)**2
equation = sp.Eq(v_avg(t).diff(t), dvdt)


# w_avg = w_avg + self.dts_non_dim * (self.a * (self.b * v_avg - w_avg) + self.w_jump * rates)

# Решаем уравнение
solution = sp.dsolve(equation)
print(solution)