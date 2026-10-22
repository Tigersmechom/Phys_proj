from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .main import simulate, theory_1d, energy, ang_momentum, K

OUT = Path(__file__).parent / "docs" / "figures"
M, R, MU, G = 0.17, 0.0286, 0.2, 9.81      # бильярдный шар на сукне


def save(name, note=""):
    fig = plt.gcf()
    for a in fig.axes:
        a.grid()
        a.legend(fontsize=9)
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.12 if note else 0, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / name, dpi=120)
    plt.close()


def cue_height():
    print("1. Удар кием на высоте h над центром, v0 = 2 м/с")
    plt.figure(figsize=(9, 6))
    for h_ratio, label in (
        (-0.6, "Ниже центра на 60 % радиуса"),
        (0.0, "В центр шара"),
        (0.4, "Выше центра на 40 % радиуса"),
        (0.8, "Выше центра на 80 % радиуса"),
    ):
        w0 = 2.0 * h_ratio / (K["ball"] * R)  # h_ratio = h/R
        t, y, _, ts = simulate([0, 0, 2.0, 0, 0, w0], 1.0, "ball", M, R, MU, 0.0, G)
        _, v, _, t1 = theory_1d(t, 2.0, w0, "ball", R, MU, 0.0, G)
        print(f"   h/R = {h_ratio:4}: скольжение {ts[0] if ts else 0:.6f} с (теория {t1 if ts else 0:.6f}),"
              f" v = {y[2, -1]:.6f} м/с (теория {v[-1]:.6f})")
        line, = plt.plot(t, y[2], label=label)
        plt.plot(t, R * y[5], "--", color=line.get_color())
        plt.plot(t[::40], v[::40], "k.", ms=3)
    plt.title("Удар кием на разной высоте\nНачальная скорость центра — 2 м/с")
    plt.xlabel("Время, с")
    plt.ylabel("Скорость, м/с")
    save("1_cue_height.png",
         "Сплошная линия — скорость центра, штриховая — окружная скорость при вращении.\n"
         "Чёрные точки — теоретическая скорость центра.")


def different_bodies():
    print("\n2. Разные тела, удар в центр, v0 = 2 м/с")
    for shape, k in K.items():
        _, y, _, ts = simulate([0, 0, 2.0, 0, 0, 0], 1.0, shape, M, R, MU, 0.0, G)
        print(f"   {shape:12s} скольжение {ts[0]:.6f} с, v = {y[2, -1]:.6f} м/с (теория {2 / (1 + k):.6f}),"
              f" в тепло {y[6, -1] / (2 * M):.6f} энергии (теория {k / (1 + k):.6f})")


def incline_angle():
    print("\n3. Ускорение от угла наклона, старт с места")
    plt.figure(figsize=(9, 6))
    for shape, color, label in (
        ("ball", "C0", "Сплошной шар"),
        ("hoop", "C3", "Тонкостенный цилиндр"),
    ):
        k = K[shape]
        alpha_c = np.degrees(np.arctan(MU * (1 + k) / k))
        angles = np.arange(2.0, 81.0, 3.0)
        a_num = []
        for angle in angles:
            t, y, _, _ = simulate([0, 0, 0, 0, 0, 0], 1.0, shape, M, R, MU, angle, G)
            a_num.append(y[2, -1] / t[-1])
        a_num = np.array(a_num) / G
        sin_a = np.sin(np.radians(angles))
        cos_a = np.cos(np.radians(angles))
        a_th = np.where(angles <= alpha_c, sin_a / (1 + k), sin_a - MU * cos_a)
        print(f"   {shape}: критический угол {alpha_c:.2f} град, "
              f"макс. отличие a/g от теории {np.max(np.abs(a_num - a_th)):.1e}")
        plt.plot(angles, a_th, color=color, label=f"{label}: теория")
        plt.plot(angles, a_num, "o", color=color, ms=4, label=f"{label}: расчёт")
        plt.axvline(alpha_c, color=color, ls=":")
    plt.title("Скатывание из состояния покоя")
    plt.xlabel("Угол наклона плоскости, градусы")
    plt.ylabel("Ускорение в долях\nускорения свободного падения")
    save("2_incline_angle.png",
         "Вертикальные пунктирные линии — предельные углы качения без скольжения.\n"
         "Цвет линии соответствует форме тела.")


def uphill():
    print("\n4. Бросок вверх по склону, v0 = -3 м/с, угол 10 град")
    plt.figure(figsize=(9, 6.5))
    for wR, label in (
        (0.0, "Без начального вращения"),
        (-3.0, "Качение с самого начала"),
        (3.0, "Вращение в сторону спуска, 3 м/с"),
        (-5.0, "Вращение в сторону подъёма, 5 м/с"),
    ):
        t, y, _, ts = simulate([0, 0, -3.0, 0, 0, wR / R], 3.0, "ball", M, R, MU, 10.0, G)
        x, v, _, t1 = theory_1d(t, -3.0, wR / R, "ball", R, MU, 10.0, G)
        print(f"   R*w0 = {wR:4}: поднялся на {-y[0].min():.6f} м (теория {-x.min():.6f}),"
              f" переход к качению {ts[0] if ts else 0:.6f} с (теория {t1 if ts else 0:.6f})")
        line, = plt.plot(t, y[2], label=label)
        plt.plot(t, R * y[5], "--", color=line.get_color())
        plt.plot(t[::50], v[::50], "k.", ms=3)
    t, y, _, ts = simulate([0, 0, -3.0, 0, 0, 0], 1.2, "ball", M, R, 0.1, 45.0, G)
    _, v, _, t1 = theory_1d(t, -3.0, 0.0, "ball", R, 0.1, 45.0, G)
    print(f"   угол 45 град, mu = 0.1: проскальзывание сменило знак в {ts[0]:.6f} с (теория {t1:.6f}),"
          f" макс. отличие v от теории {np.max(np.abs(y[2] - v)):.1e} м/с")
    plt.title("Бросок вверх по склону: начальная скорость — 3 м/с, наклон — 10°")
    plt.xlabel("Время, с")
    plt.ylabel("Скорость вдоль склона, м/с")
    save("3_uphill.png",
         "Сплошная линия — скорость центра, штриховая — окружная скорость; чёрные точки — теория.\n"
         "Отрицательная скорость — движение вверх, положительная — вниз.\n"
         "В легенде указана начальная окружная скорость при вращении.")


def across_slope():
    print("\n5. Бросок поперёк склона, угол 20 град, v0 = (0, 2) м/с")
    _, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    for wxR, wyR, label in (
        (-2.0, 0.0, "Качение с самого начала"),
        (0.0, 0.0, "Без начального вращения"),
        (2.0, 0.0, "Вращение против начального движения"),
        (0.0, -2.0, "Вращение в сторону подъёма"),
    ):
        t, y, roll, _ = simulate([0, 0, 0.0, 2.0, wxR / R, wyR / R], 1.2, "ball", M, R, MU, 20.0, G)
        E = energy(y, "ball", M, R, 20.0, G)
        Lx, Ly = ang_momentum(y, "ball", M, R)
        dE = np.abs(E + y[6] - E[0])
        dL = np.abs(Ly - Ly[0] - M * G * R * np.sin(np.radians(20.0)) * t)
        print(f"   R*w0 = ({wxR}, {wyR}): ошибка E + Q {dE.max():.1e} Дж, Lx {np.ptp(Lx):.1e},"
              f" Ly {dL.max():.1e} (сам L ~ {np.abs(Ly).max():.1e})")
        line, = ax[0].plot(y[1], -y[0], lw=1, label=label)
        ax[0].plot(y[1][~roll], -y[0][~roll], lw=3, color=line.get_color())
        if wxR == 2.0:
            ax[1].semilogy(t, dE / E[0] + 1e-17, label="Баланс энергии")
            ax[1].semilogy(t, dL / np.abs(Ly).max() + 1e-17, label="Изменение момента импульса")
    ax[0].set_title("Бросок поперёк склона с наклоном 20°")
    ax[0].set_xlabel("Смещение поперёк склона, м")
    ax[0].set_ylabel("Смещение вверх по склону, м")
    ax[0].axis("equal")
    ax[1].set_title("Погрешности расчёта\nпри вращении против начального движения")
    ax[1].set_xlabel("Время, с")
    ax[1].set_ylabel("Относительная погрешность")
    save("4_across_slope.png",
         "Начальная скорость центра — 2 м/с; окружная скорость вращающихся шаров — 2 м/с.\n"
         "Толстые участки траекторий — скольжение, тонкие — качение без скольжения.")


def accuracy():
    print("\n6. Точность от rtol (случай 5, R*w0 = (2, 0))")
    state0 = [0, 0, 0.0, 2.0, 2.0 / R, 0.0]
    _, y_ref, _, ts_ref = simulate(state0, 1.2, "ball", M, R, MU, 20.0, G, rtol=1e-13)
    for rtol in (1e-5, 1e-8, 1e-11):
        t, y, _, ts = simulate(state0, 1.2, "ball", M, R, MU, 20.0, G, rtol=rtol)
        dt = abs(ts[0] - ts_ref[0])
        dv = np.abs(y[2:4, -1] - y_ref[2:4, -1]).max()
        print(f"   rtol = {rtol:.0e}: ошибка момента перехода {dt:.1e} с,"
              f" конечной скорости {dv:.1e} м/с, точек {len(t)}")


def main():
    cue_height()
    different_bodies()
    incline_angle()
    uphill()
    across_slope()
    accuracy()


if __name__ == "__main__":
    main()
