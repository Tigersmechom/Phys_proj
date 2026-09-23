import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

G = 9.81
model = "quad"
mass = 1.0
k = 0.1
x0 = 0.0
y0 = 0.0
v0 = 30.0
grad = 45.0

# solution = solve_ivp(
#     equations,
#     [0, 100],             # [a, b] - время t от a до b
#     state0,               # начальное состояние
#     events=hit_ground,
#     max_step=0.01,        # максимальный step по времени (мб меньше)
#     method="RK45",        # вычисляет формулы 4ого и 5ого порядка, и если почти одинаковые, то шаг нормальный (иначе меняется шаг) (погрешность должна быть меньше atol + rtol*x)
#     rtol=1e-8,            # относительная погрешность (пусть x = 100, тогда rtol имеет главную роль)
#     atol=1e-10,           # абсолютная погрешность (если x = 0, то atol имеет главную роль)
#     args=(model, k, mass) # - эти доп аргументы допередаются в equations (!!! в hit_ground тоже передадутся)
# )


def simulate(model, mass, k, x0, y0, v0, grad):
    angle = np.radians(grad)
    vx0 = v0 * np.cos(angle)
    vy0 = v0 * np.sin(angle)
    state0 = [x0, y0, vx0, vy0]
    solution = solve_ivp(
        equations,
        [0, 100],
        state0,
        args=(model, k, mass),
        events=hit_ground,
        max_step=0.01,
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )
    return solution


angle = np.radians(grad)
vx0 = v0 * np.cos(angle)
vy0 = v0 * np.sin(angle)
state0 = [
    x0,
    y0,
    vx0,
    vy0
]

def equations(t, state, model, k, mass):
    x, y, vx, vy = state
    if model == "none":
        ax = 0
        ay = -G
    elif model == "line":
        ax = -k * vx / mass
        ay = -G - k * vy / mass
    elif model == "quad":
        V = np.sqrt(vx * vx + vy * vy)
        ax = -k * V * vx / mass
        ay = -G - k * V * vy / mass
    return [vx, vy, ax, ay]


def hit_ground(t, state, *args):
    return state[1]


hit_ground.direction = -1  # y был > 0, а стал < 0 (коснулись земли)
hit_ground.terminal = True # говорит, что если сработало, то стоп

if __name__ == "__main__":
    solution = simulate(
        model, mass, k,
        x0, y0, v0, grad
    )

    x = solution.y[0]
    y = solution.y[1]

    plt.plot(x, y)
    plt.xlabel("x, m")
    plt.ylabel("y, m")
    plt.grid()
    plt.show()

    flight_time = solution.t_events[0][0]
    final_state = solution.y_events[0][0]

    distance = final_state[0]
    max_height = np.max(solution.y[1])

    print("Время полёта:", flight_time, "с")        # T
    print("Дальность:", distance, "м")              # L
    print("Максимальная высота:", max_height, "м")  # H


########################################################################################################################################################

"""
Analitics:

(none):
T = 2 * v0 * sin(A) / g
L = v0^2 * sin(2A) / g
H = v0^2 * sin^2(A) / 2g

T_theory = 2 * v0 * np.sin(angle) / G
L_theory = v0**2 * np.sin(2 * angle) / G
H_theory = v0**2 * np.sin(angle)**2 / (2 * G)

  
print("\nТеория:")
print("Время полёта:", T_theory, "с")
print("Дальность:", L_theory, "м")
print("Максимальная высота:", H_theory, "м")

"""