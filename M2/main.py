import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

model = "hooke"     # "hooke" или "hertz"
m1 = 1.0
m2 = 2.0
r1 = 0.5
r2 = 0.5
k = 5e4             # Н/м для hooke, Н/м^1.5 для hertz
v1 = 3.0
v2 = 0.0
gap = 3.0           # расстояние между центрами в начале
b = 0.0             # прицельный параметр
t_max = 5.0

C = 3e8
V_LIMIT = 0.01 * C
MAX_DEFORM = 0.1
MAX_STEPS = 5e6


def contact_force(delta, k, model):
    if delta <= 0:
        return 0.0
    if model == "hooke":
        return k * delta
    return k * delta ** 1.5


def equations(t, state, m1, m2, r1, r2, k, model):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
    dx = x2 - x1
    dy = y2 - y1
    d = np.sqrt(dx**2 + dy**2)
    delta = r1 + r2 - d

    F = 0.0
    nx = 0.0
    ny = 0.0
    if delta > 0 and d > 0:
        nx = dx / d
        ny = dy / d
        F = contact_force(delta, k, model)

    ax1 = -F * nx / m1
    ay1 = -F * ny / m1
    ax2 = F * nx / m2
    ay2 = F * ny / m2
    return [vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2]


def touch(t, state, m1, m2, r1, r2, k, model):
    x1, y1, x2, y2 = state[0], state[1], state[4], state[5]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) - (r1 + r2)


def max_compression(v, m1, m2, k, model):
    mu = m1 * m2 / (m1 + m2)
    if model == "hooke":
        return v * np.sqrt(mu / k)
    return (5 * mu * v**2 / (4 * k)) ** 0.4


def contact_time(v, m1, m2, k, model):
    mu = m1 * m2 / (m1 + m2)
    if model == "hooke":
        return np.pi * np.sqrt(mu / k)
    if v == 0:
        return np.inf
    I = quad(lambda u: 1 / np.sqrt(1 - u**2.5), 0, 1)[0]
    return 2 * I * max_compression(v, m1, m2, k, model) / v


def check_params(state0, t_max, m1, m2, r1, r2, k, model):
    if model != "hooke" and model != "hertz":
        raise ValueError("неизвестная модель")

    if not np.all(np.isfinite(list(state0) + [t_max, m1, m2, r1, r2, k])):
        raise ValueError("есть nan или inf")

    if m1 <= 0 or m2 <= 0:
        raise ValueError("масса должна быть > 0")
    if r1 <= 0 or r2 <= 0:
        raise ValueError("радиус должен быть > 0")
    if k <= 0:
        raise ValueError("k должно быть > 0")
    if t_max <= 0:
        raise ValueError("t_max должно быть > 0")

    x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state0
    u1 = np.sqrt(vx1**2 + vy1**2)
    u2 = np.sqrt(vx2**2 + vy2**2)
    if u1 >= C or u2 >= C:
        raise ValueError("скорость больше скорости света")
    if u1 > V_LIMIT or u2 > V_LIMIT:
        raise ValueError("скорость слишком большая, механика Ньютона не работает")

    if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < r1 + r2:
        raise ValueError("шары перекрываются в начале")

    v_rel = np.sqrt((vx2 - vx1)**2 + (vy2 - vy1)**2)
    if max_compression(v_rel, m1, m2, k, model) > MAX_DEFORM * min(r1, r2):
        raise ValueError("слишком большое сжатие, шары слишком мягкие")

    max_step = min(contact_time(v_rel, m1, m2, k, model) / 50, t_max / 100)
    if t_max / max_step > MAX_STEPS:
        raise ValueError("слишком много шагов, уменьшите k или t_max")


def simulate(state0, t_max, m1, m2, r1, r2, k, model):
    check_params(state0, t_max, m1, m2, r1, r2, k, model)

    v_rel = np.sqrt((state0[6] - state0[2])**2 + (state0[7] - state0[3])**2)
    max_step = min(contact_time(v_rel, m1, m2, k, model) / 50, t_max / 100)

    solution = solve_ivp(
        equations,
        [0, t_max],
        state0,
        args=(m1, m2, r1, r2, k, model),
        events=touch,
        max_step=max_step,
        method="RK45",
        rtol=1e-9,
        atol=1e-11
    )
    return solution


def energy(state, m1, m2):
    vx1, vy1, vx2, vy2 = state[2], state[3], state[6], state[7]
    return m1 * (vx1**2 + vy1**2) / 2 + m2 * (vx2**2 + vy2**2) / 2


def potential(state, r1, r2, k, model):
    x1, y1, x2, y2 = state[0], state[1], state[4], state[5]
    delta = r1 + r2 - np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if delta <= 0:
        return 0.0
    if model == "hooke":
        return k * delta**2 / 2
    return 0.4 * k * delta**2.5


def momentum(state, m1, m2):
    px = m1 * state[2] + m2 * state[6]
    py = m1 * state[3] + m2 * state[7]
    return px, py


def analytic_head_on(v1, m1, m2, v2=0.0):
    v1p = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
    v2p = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
    return v1p, v2p


def analytic_oblique(v1, b, m1, m2, R):
    # шар 2 стоит, импульс передаётся вдоль линии центров
    n = np.array([np.sqrt(R**2 - b**2), -b]) / R
    v2p = 2 * m1 / (m1 + m2) * v1 * n[0] * n
    v1p = np.array([v1, 0.0]) - m2 / m1 * v2p
    return v1p, v2p


if __name__ == "__main__":
    state0 = [0.0, b, v1, 0.0, gap, 0.0, v2, 0.0]

    try:
        check_params(state0, t_max, m1, m2, r1, r2, k, model)
    except ValueError as e:
        print("Ошибка:", e)
        sys.exit(1)

    sol = simulate(state0, t_max, m1, m2, r1, r2, k, model)

    if b == 0:
        plt.plot(sol.t, sol.y[0], label="ball 1")
        plt.plot(sol.t, sol.y[4], label="ball 2")
        plt.xlabel("t, s")
        plt.ylabel("x, m")
    else:
        plt.plot(sol.y[0], sol.y[1], label="ball 1")
        plt.plot(sol.y[4], sol.y[5], label="ball 2")
        plt.xlabel("x, m")
        plt.ylabel("y, m")
        plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.show()

    final = sol.y[:, -1]
    print("Численно: v1' =", final[2:4], "v2' =", final[6:8])
    if b == 0:
        print("Теория: v1', v2' =", analytic_head_on(v1, m1, m2, v2))

    if len(sol.t_events[0]) >= 2:
        print("Время контакта:", sol.t_events[0][1] - sol.t_events[0][0], "с")

    print("Энергия: было", energy(state0, m1, m2), "стало", energy(final, m1, m2))
    print("Импульс: было", *momentum(state0, m1, m2), "стало", *momentum(final, m1, m2))
