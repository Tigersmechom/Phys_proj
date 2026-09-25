import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import simulate, contact_force, contact_time, max_compression, analytic_head_on, analytic_oblique, energy, potential

R = 0.5
GAP = 2.0


def run(v1, m1, m2, k, model, b=0.0):
    state0 = [0.0, b, v1, 0.0, GAP, 0.0, 0.0, 0.0]
    t_max = (GAP - 2 * R) / v1 + 0.3
    return simulate(state0, t_max, m1, m2, R, R, k, model)


def get_delta(sol):
    return 2 * R - np.sqrt((sol.y[4] - sol.y[0])**2 + (sol.y[5] - sol.y[1])**2)


# 1. сила во время удара
print("1. Сила, v = 3, m1 = m2 = 1")
for model, k in [("hooke", 5e4), ("hertz", 5e5)]:
    sol = run(3.0, 1.0, 1.0, k, model)
    t0 = sol.t_events[0][0]
    t1 = sol.t_events[0][1]
    delta = get_delta(sol)
    F = np.array([contact_force(d, k, model) for d in delta])
    mask = (sol.t > t0 - 0.002) & (sol.t < t1 + 0.002)
    plt.plot((sol.t[mask] - t0) * 1000, F[mask], label=model)
    print(model, "t_c =", round((t1 - t0) * 1000, 3), "мс, F_max =", round(F.max(), 1), "Н, delta_max =", round(delta.max() * 1000, 3), "мм")
plt.xlabel("t, ms")
plt.ylabel("F, N")
plt.legend()
plt.grid()
plt.savefig("results/1_force.png")
plt.close()


# 2. время контакта от скорости
print("\n2. Время контакта (мс): v, hooke, теория, hertz, теория")
speeds = [0.3, 1.0, 3.0, 8.0]
times = {"hooke": [], "hertz": []}
for v in speeds:
    line = [v]
    for model, k in [("hooke", 5e4), ("hertz", 5e5)]:
        sol = run(v, 1.0, 1.0, k, model)
        t_c = sol.t_events[0][1] - sol.t_events[0][0]
        times[model].append(t_c)
        line += [round(t_c * 1000, 3), round(contact_time(v, 1.0, 1.0, k, model) * 1000, 3)]
    print(*line)
for model in times:
    slope = np.polyfit(np.log(speeds), np.log(times[model]), 1)[0]
    print("наклон", model, round(slope, 3))
    plt.loglog(speeds, np.array(times[model]) * 1000, "o-", label=model)
plt.xlabel("v, m/s")
plt.ylabel("t_c, ms")
plt.legend()
plt.grid()
plt.savefig("results/2_contact_time.png")
plt.close()


# 3. лобовой удар
print("\n3. Лобовой удар, v1 = 3: m2/m1, v1' числ, v1' теор, v2' числ, v2' теор")
ratios = [0.1, 0.5, 1.0, 2.0, 10.0]
num1 = []
num2 = []
for q in ratios:
    sol = run(3.0, 1.0, q, 5e4, "hooke")
    th1, th2 = analytic_head_on(3.0, 1.0, q)
    num1.append(sol.y[2, -1])
    num2.append(sol.y[6, -1])
    print(q, round(sol.y[2, -1], 4), round(th1, 4), round(sol.y[6, -1], 4), round(th2, 4))
q = np.linspace(0.1, 10, 200)
plt.semilogx(q, analytic_head_on(3.0, 1.0, q)[0], label="v1' theory")
plt.semilogx(q, analytic_head_on(3.0, 1.0, q)[1], label="v2' theory")
plt.semilogx(ratios, num1, "o", label="v1'")
plt.semilogx(ratios, num2, "o", label="v2'")
plt.xlabel("m2 / m1")
plt.ylabel("v', m/s")
plt.legend()
plt.grid()
plt.savefig("results/3_head_on.png")
plt.close()


# 4. косой удар
print("\n4. Косой удар: b/R, m2/m1, угол шара 2, теория, угол между шарами")
bs = [0.1, 0.3, 0.5, 0.7, 0.9]
for q in [1.0, 2.0]:
    angles = []
    for bR in bs:
        sol = run(3.0, 1.0, q, 5e5, "hooke", b=bR)
        u1 = sol.y[2:4, -1]
        u2 = sol.y[6:8, -1]
        a2 = np.degrees(np.arctan2(u2[1], u2[0]))
        between = np.degrees(np.arccos(np.dot(u1, u2) / np.linalg.norm(u1) / np.linalg.norm(u2)))
        angles.append(a2)
        print(bR, q, round(a2, 2), round(-np.degrees(np.arcsin(bR)), 2), round(between, 2))
    plt.plot(bs, angles, "o", label="m2/m1 = " + str(q))
x = np.linspace(0, 0.95, 100)
plt.plot(x, -np.degrees(np.arcsin(x)), label="theory")
plt.xlabel("b / R")
plt.ylabel("angle, deg")
plt.legend()
plt.grid()
plt.savefig("results/4_oblique.png")
plt.close()


# 5. зависимость от k
print("\n5. b/R = 0.6, m1 = m2 = 1: k, ошибка v2', delta_max/R")
ks = [1e4, 1e5, 1e6, 1e7]
errors = []
_, v2p = analytic_oblique(3.0, 0.6, 1.0, 1.0, 2 * R)
for k in ks:
    sol = run(3.0, 1.0, 1.0, k, "hooke", b=0.6)
    errors.append(np.linalg.norm(sol.y[6:8, -1] - v2p))
    print(k, round(errors[-1], 5), round(max_compression(3.0, 1.0, 1.0, k, "hooke") / (2 * R), 5))
print("наклон", round(np.polyfit(np.log(ks), np.log(errors), 1)[0], 3))
plt.loglog(ks, errors, "o-")
plt.xlabel("k, N/m")
plt.ylabel("error v2', m/s")
plt.grid()
plt.savefig("results/5_convergence.png")
plt.close()


# 6. энергия
print("\n6. Энергия, v1 = 3, m1 = 1, m2 = 1.5, b = 0.3")
for model, k in [("hooke", 5e4), ("hertz", 5e5)]:
    sol = run(3.0, 1.0, 1.5, k, model, b=0.3)
    E0 = energy(sol.y[:, 0], 1.0, 1.5)
    kin = np.array([energy(s, 1.0, 1.5) for s in sol.y.T])
    pot = np.array([potential(s, R, R, k, model) for s in sol.y.T])
    err = abs(kin + pot - E0) / E0
    print(model, "макс ошибка", err.max(), ", мин кинетическая", round(kin.min() / E0 * 100, 1), "%")
    plt.plot(sol.t, err, label=model)
plt.xlabel("t, s")
plt.ylabel("|E - E0| / E0")
plt.legend()
plt.grid()
plt.savefig("results/6_energy.png")
plt.close()
