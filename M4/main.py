import sys
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# параметры
shape = "ball"        # ball, hollow_ball, cylinder, hoop (тонкостенный цилиндр)
m = 0.17              # масса, кг (бильярдный шар)
R = 0.0286            # радиус, м
mu = 0.2              # коэффициент трения скольжения
alpha_deg = 0.0       # угол наклона, градусы
g = 9.81
# x - вниз по склону, y - поперёк склона, z - нормаль вверх; качение вдоль x: wy = vx / R
x0, y0 = 0.0, 0.0     # м
vx0, vy0 = 2.0, 0.0   # м/с
wx0, wy0 = 40.0, -30.0  # рад/с
t_max = 2.0           # с

C = 3e8
KT = 4.1e-21          # kT при комнатной температуре, Дж
RHO_AIR = 1.2         # плотность воздуха, кг/м^3
ETA_AIR = 1.8e-5      # вязкость воздуха, Па*с
MU_MAX = 2.0
R_MIN = 1e-4          # м
V_MAX = 30.0          # предел оценки скорости, м/с
WR_MAX = 20.0         # макс. окружная скорость R*w, м/с
T_MAX = 3600.0        # не больше часа движения
MAX_NFEV = 100_000    # предел числа вычислений правой части
MAX_TIME = 5.0        # не больше 5 секунд на один расчёт
N_POINTS = 1000
RTOL = 1e-11

# I = k * m * R^2
K = {
    "ball": 2 / 5,
    "hollow_ball": 2 / 3,
    "cylinder": 1 / 2,
    "hoop": 1.0,
}


def slip(y, R):
    # скорость точки касания
    vx, vy, wx, wy = y[2:6]
    return vx - R * wy, vy + R * wx


def can_roll(k, mu, alpha):
    # нужное для качения трение покоя m*g*sin(a)*k/(1+k) не больше mu*m*g*cos(a)
    return np.sin(alpha) * k / (1 + k) <= mu * np.cos(alpha)


def rhs_slide(t, y, k, m, R, mu, alpha, g, u_eps):
    # y = [x, y, vx, vy, wx, wy, Q], Q - тепло от трения
    ux, uy = slip(y, R)
    us = np.hypot(ux, uy)
    if us > u_eps:
        ex, ey = ux / us, uy / us
    else:
        # при u = 0 проскальзывание начнётся вниз по склону
        ex, ey = 1.0, 0.0
    f = mu * g * np.cos(alpha)        # сила трения на единицу массы
    return [
        y[2], y[3],
        g * np.sin(alpha) - f * ex, -f * ey,
        -f * ey / (k * R), f * ex / (k * R),
        m * f * us,
    ]


def rhs_roll(t, y, k, m, R, mu, alpha, g, u_eps):
    a = g * np.sin(alpha) / (1 + k)
    return [y[2], y[3], a, 0.0, 0.0, a / R, 0.0]


def slip_stop(t, y, k, m, R, mu, alpha, g, u_eps):
    return np.hypot(*slip(y, R)) - u_eps


slip_stop.terminal = True
slip_stop.direction = -1


def to_rolling(y, k, R):
    # убираем остаточное проскальзывание, сохраняя момент импульса относительно точки касания
    y = y.copy()
    vx = (y[2] + k * R * y[5]) / (1 + k)
    vy = (y[3] - k * R * y[4]) / (1 + k)
    y[2:6] = vx, vy, -vy / R, vx / R
    return y


def max_speed(state0, t_max, R, alpha_deg, g):
    # оценка сверху скорости за время расчёта
    v0 = np.hypot(state0[2], state0[3])
    wR = R * np.hypot(state0[4], state0[5])
    return v0 + wR + g * np.sin(np.radians(alpha_deg)) * t_max


def check_params(state0, t_max, shape, m, R, mu, alpha_deg, g):
    if np.shape(state0) != (6,):
        raise ValueError("нужно 6 начальных значений: x, y, vx, vy, wx, wy")
    if shape not in K:
        raise ValueError(f"неизвестная форма {shape}")
    values = list(state0) + [t_max, m, R, mu, alpha_deg, g]
    if not np.all(np.isfinite(values)):
        raise ValueError("есть nan или inf")
    if min(m, R, g, t_max) <= 0:
        raise ValueError("m, R, g и t_max должны быть больше нуля")
    if not 1e-6 <= t_max <= T_MAX:
        raise ValueError(f"время расчёта должно быть от 1e-6 до {T_MAX:g} с")
    if m > 100 or R > 1 or not 0.01 <= g <= 100:
        raise ValueError("для этой модели нужны m <= 100 кг, R <= 1 м и g от 0.01 до 100 м/с^2")
    if not 0 <= mu <= MU_MAX:
        raise ValueError(f"mu должен быть от 0 до {MU_MAX}")
    if not 0 <= alpha_deg < 90:
        raise ValueError("угол должен быть от 0 до 90 градусов")
    if shape in ("cylinder", "hoop") and (state0[3] != 0 or state0[4] != 0):
        raise ValueError("цилиндр катится только вдоль x, нужно vy = 0 и wx = 0")
    if R < R_MIN:
        raise ValueError("тело слишком маленькое, прилипание сравнимо с весом")
    if RHO_AIR * 4 / 3 * np.pi * R**3 > 0.1 * m:
        raise ValueError("тело слишком лёгкое, архимедова сила больше 10% веса")

    v = max_speed(state0, t_max, R, alpha_deg, g)
    wR = R * np.hypot(state0[4], state0[5])
    if max(v, wR) >= C:
        raise ValueError("скорость больше скорости света")
    if wR > WR_MAX:
        raise ValueError(f"R*w больше {WR_MAX} м/с, слишком быстрое вращение для этой модели")
    if v > V_MAX:
        raise ValueError(f"оценка скорости {v:.3g} м/с больше допустимых {V_MAX:g} м/с")
    # Стокс + лобовое сопротивление, Cx = 0.5
    drag = 6 * np.pi * ETA_AIR * R * v + 0.25 * RHO_AIR * np.pi * R**2 * v**2
    if drag > 0.1 * m * g:
        raise ValueError(f"скорость дойдёт до {v:.3g} м/с, сопротивление воздуха больше 10% веса")
    if 0 < m * v**2 < 100 * KT:
        raise ValueError("скорость сравнима со скоростью теплового движения")


def simulate(state0, t_max, shape, m, R, mu, alpha_deg, g, rtol=RTOL):
    check_params(state0, t_max, shape, m, R, mu, alpha_deg, g)
    if not np.isfinite(rtol) or not 1e-13 <= rtol <= 1e-5:
        raise ValueError("rtol должен быть от 1e-13 до 1e-5")
    k, alpha = K[shape], np.radians(alpha_deg)
    rolling_possible = can_roll(k, mu, alpha)
    v = max_speed(state0, t_max, R, alpha_deg, g) or 1.0
    # погрешности в масштабах задачи, чтобы точность не зависела от величины скоростей
    atol = rtol * np.array([v * t_max, v * t_max, v, v, v / R, v / R, m * v**2])
    u_eps = 1e3 * rtol * v        # порог нулевого проскальзывания, с запасом над погрешностью
    args = (k, m, R, mu, alpha, g, u_eps)

    t, y = 0.0, np.append(np.array(state0, dtype=float), 0.0)
    rolling = rolling_possible and np.hypot(*slip(y, R)) <= u_eps
    times, states, modes, t_stick = [], [], [], []
    start = perf_counter()
    nfev = 0

    def equations(t, y, *args):
        nonlocal nfev
        nfev += 1
        if nfev > MAX_NFEV:
            raise RuntimeError("слишком много вычислений, проверьте параметры и точность")
        if perf_counter() - start > MAX_TIME:
            raise RuntimeError(f"расчёт занял больше {MAX_TIME:g} с, проверьте параметры и точность")
        rhs = rhs_roll if rolling else rhs_slide
        return rhs(t, y, *args)

    while t < t_max:
        if rolling:
            y = to_rolling(y, k, R)
        sol = solve_ivp(
            equations, [t, t_max], y,
            args=args, events=None if rolling else slip_stop,
            max_step=t_max / N_POINTS, rtol=rtol, atol=atol,
        )
        if sol.status == -1:
            raise RuntimeError(sol.message)
        if sol.t[-1] <= t:
            raise RuntimeError("не удалось продвинуться по времени, проверьте параметры и точность")
        skip = 1 if times else 0
        times.append(sol.t[skip:])
        states.append(sol.y[:, skip:])
        modes.append(np.full(len(sol.t) - skip, rolling))
        t, y = sol.t[-1], sol.y[:, -1]
        if sol.status == 1:       # проскальзывание исчезло
            t_stick.append(t)
            y = to_rolling(y, k, R)
            rolling = rolling_possible  # иначе проскальзывание меняет знак
    return np.concatenate(times), np.hstack(states), np.concatenate(modes), t_stick


def energy(y, shape, m, R, alpha_deg, g):
    I = K[shape] * m * R**2
    return (0.5 * m * (y[2]**2 + y[3]**2) + 0.5 * I * (y[4]**2 + y[5]**2)
            - m * g * np.sin(np.radians(alpha_deg)) * y[0])


def ang_momentum(y, shape, m, R):
    # момент импульса относительно точки на плоскости, компоненты вдоль плоскости
    I = K[shape] * m * R**2
    return -m * R * y[3] + I * y[4], m * R * y[2] + I * y[5]


def theory_1d(t, v0, w0, shape, R, mu, alpha_deg, g):
    # движение вдоль x: скольжение с постоянными ускорениями до u = 0, потом качение
    # (или скольжение вниз, если угол больше критического)
    k, alpha = K[shape], np.radians(alpha_deg)
    ag, af = g * np.sin(alpha), mu * g * np.cos(alpha)
    rolls = can_roll(k, mu, alpha)

    def accel(s):   # s - знак проскальзывания, 0 - качение
        if s == 0:
            a = ag / (1 + k)
            return a, a / R
        return ag - s * af, s * af / (k * R)

    u0 = v0 - R * w0
    if u0 != 0:
        s1 = np.sign(u0)
    else:
        s1 = 0 if rolls else 1
    a1, e1 = accel(s1)
    a2, e2 = accel(0 if rolls else 1)
    du = a1 - R * e1
    t1 = -u0 / du if s1 * du < 0 else np.inf

    t = np.asarray(t, dtype=float)
    ta, tb = np.minimum(t, t1), np.maximum(t - t1, 0.0)
    va, wa = v0 + a1 * ta, w0 + e1 * ta
    x = v0 * ta + a1 * ta**2 / 2 + va * tb + a2 * tb**2 / 2
    return x, va + a2 * tb, wa + e2 * tb, t1


def main():
    state0 = [x0, y0, vx0, vy0, wx0, wy0]
    try:
        t, y, roll, t_stick = simulate(state0, t_max, shape, m, R, mu, alpha_deg, g)
    except ValueError as e:
        print("Недопустимые параметры:", e)
        sys.exit(1)
    except RuntimeError as e:
        print("Расчёт остановлен:", e)
        sys.exit(1)

    k = K[shape]
    print("Проскальзывание исчезло в t =", ", ".join(f"{s:.6f}" for s in t_stick) or "-")
    print("Конечная скорость", y[2:4, -1], "м/с, угловая", y[4:6, -1], "рад/с")
    if roll[-1]:
        # при качении скорость следует из закона изменения момента импульса
        vx = (vx0 + k * R * wy0 + g * np.sin(np.radians(alpha_deg)) * t[-1]) / (1 + k)
        print("Теория:", np.array([vx, (vy0 - k * R * wx0) / (1 + k)]))
    E = energy(y, shape, m, R, alpha_deg, g)
    print("Энергия в начале", E[0], "в конце", E[-1], "тепло", y[6, -1],
          "ошибка баланса", E[-1] + y[6, -1] - E[0])

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    ax[0].set_title("Траектория движения")
    ax[0].plot(y[0], y[1], "b", label="Качение без скольжения")
    ax[0].plot(y[0][~roll], y[1][~roll], "r", lw=3, label="Скольжение")
    ax[0].set_xlabel("Смещение вдоль склона, м")
    ax[0].set_ylabel("Смещение поперёк склона, м")
    ax[0].axis("equal")
    ax[1].set_title("Скорости движения и вращения")
    ax[1].plot(t, y[2], label="Скорость центра вдоль склона")
    ax[1].plot(t, R * y[5], "--", label="Окружная скорость вдоль склона")
    ax[1].plot(t, y[3], label="Скорость центра поперёк склона")
    ax[1].plot(t, -R * y[4], "--", label="Окружная скорость поперёк склона")
    ax[1].set_xlabel("Время, с")
    ax[1].set_ylabel("Скорость, м/с")
    ax[2].set_title("Баланс энергии")
    ax[2].plot(t, E, label="Механическая энергия")
    ax[2].plot(t, y[6], label="Выделившееся тепло")
    ax[2].plot(t, E + y[6], "k--", label="Энергия с учётом тепла")
    ax[2].set_xlabel("Время, с")
    ax[2].set_ylabel("Энергия, Дж")
    for a in ax:
        a.grid()
        a.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
