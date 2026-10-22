from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest

from ..main import simulate, check_params, theory_1d, energy, ang_momentum, K, C
from .. import main as model

M, R, MU, G = 0.17, 0.0286, 0.2, 9.81      # бильярдный шар на сукне


@pytest.mark.parametrize("shape, v0, wR, mu, alpha, t_max", [
    ("ball", 2.0, 0.0, MU, 0.0, 1.0),          # удар в центр
    ("hollow_ball", 2.0, 0.0, MU, 0.0, 1.0),
    ("cylinder", 2.0, 0.0, MU, 0.0, 1.0),
    ("hoop", 2.0, 0.0, MU, 0.0, 1.0),
    ("ball", 2.0, -3.0, MU, 0.0, 1.0),         # удар ниже центра
    ("ball", 2.0, 2.0, MU, 0.0, 1.0),          # сразу катится
    ("ball", 0.0, -2.0, MU, 0.0, 1.0),         # оттяжка
    ("ball", 0.0, 0.0, MU, 15.0, 1.5),         # скатывание с места
    ("hoop", 0.0, 0.0, MU, 15.0, 1.5),
    ("ball", 0.0, 0.0, MU, 50.0, 1.0),         # угол больше критического
    ("ball", -3.0, 0.0, MU, 10.0, 3.0),        # бросок вверх по склону
    ("hoop", -3.0, 3.0, MU, 10.0, 3.0),
    ("ball", -3.0, -5.0, MU, 10.0, 3.0),
    ("ball", -3.0, 0.0, 0.1, 45.0, 1.2),       # проскальзывание меняет знак
])
def test_vs_theory(shape, v0, wR, mu, alpha, t_max):
    t, y, roll, t_stick = simulate([0, 0, v0, 0, 0, wR / R], t_max, shape, M, R, mu, alpha, G)
    x, v, w, t1 = theory_1d(t, v0, wR / R, shape, R, mu, alpha, G)
    assert np.max(np.abs(y[0] - x)) < 1e-7
    assert np.max(np.abs(y[2] - v)) < 1e-7
    assert np.max(np.abs(R * y[5] - R * w)) < 1e-7
    if np.isinf(t1):
        assert t_stick == []
    else:
        assert abs(t_stick[0] - t1) / t1 < 1e-6


@pytest.mark.parametrize("shape", list(K))
def test_center_hit(shape):
    # после скольжения v = v0/(1+k), в тепло уходит k/(1+k) энергии
    k, v0 = K[shape], 2.0
    t, y, roll, t_stick = simulate([0, 0, v0, 0, 0, 0], 1.0, shape, M, R, MU, 0.0, G)
    assert abs(y[2, -1] - v0 / (1 + k)) < 1e-9
    assert abs(y[6, -1] - M * v0**2 / 2 * k / (1 + k)) < 1e-9
    assert abs(t_stick[0] - v0 / (MU * G * (1 + 1 / k))) < 1e-7


@pytest.mark.parametrize("shape", list(K))
def test_critical_angle(shape):
    # tg(a) = mu*(1+k)/k: чуть меньше - катится, чуть больше - скользит
    k = K[shape]
    alpha_c = np.degrees(np.arctan(MU * (1 + k) / k))
    _, y, roll, _ = simulate([0, 0, 0, 0, 0, 0], 1.0, shape, M, R, MU, alpha_c - 0.01, G)
    assert np.all(roll) and y[6, -1] == 0
    _, y, roll, _ = simulate([0, 0, 0, 0, 0, 0], 1.0, shape, M, R, MU, alpha_c + 0.01, G)
    assert not np.any(roll) and y[6, -1] > 0


@pytest.mark.parametrize("shape, state0, mu, alpha", [
    ("ball", [0, 0, 2.0, 0, 40.0, -30.0], 0.2, 0.0),
    ("hollow_ball", [0, 0, 1.0, -1.0, 20.0, 60.0], 0.3, 0.0),
    ("ball", [0, 0, -1.0, 2.0, 30.0, 10.0], 0.2, 20.0),
    ("ball", [0, 0, 0.5, 1.0, 0.0, 0.0], 0.1, 40.0),
])
def test_conservation(shape, state0, mu, alpha):
    # E + Q = const, Lx = const, Ly = Ly0 + m*g*R*sin(a)*t
    t, y, roll, t_stick = simulate(state0, 1.2, shape, M, R, mu, alpha, G)
    E = energy(y, shape, M, R, alpha, G)
    assert np.max(np.abs(E + y[6] - E[0])) < 1e-12
    Lx, Ly = ang_momentum(y, shape, M, R)
    assert np.max(np.abs(Lx - Lx[0])) < 1e-14
    assert np.max(np.abs(Ly - Ly[0] - M * G * R * np.sin(np.radians(alpha)) * t)) < 1e-14


@pytest.mark.parametrize("v0", [1e-7, 1e-3, 10.0])
def test_small_and_large_speeds(v0):
    # относительная точность не зависит от величины скорости
    t, y, roll, t_stick = simulate([0, 0, v0, 0, 0, 0], 2.0, "ball", M, R, MU, 0.0, G)
    t1 = v0 / (MU * G * 3.5)
    assert abs(t_stick[0] - t1) / t1 < 1e-6
    assert abs(y[2, -1] - v0 / 1.4) / v0 < 1e-12


def test_long_run_is_fast():
    start = perf_counter()
    t, y, roll, t_stick = simulate([0, 0, 2.0, 0, 0, 0], 3600.0, "ball", M, R, MU, 0.0, G)
    assert perf_counter() - start < 5
    assert len(t) < 3000


GOOD = dict(state0=[0, 0, 2.0, 0, 0, 0], t_max=2.0, shape="ball",
            m=M, R=R, mu=MU, alpha_deg=0.0, g=G)


def test_good_params_pass():
    check_params(**GOOD)


@pytest.mark.parametrize("changes", [
    dict(state0=[0, 0, C, 0, 0, 0]),                 # скорость света
    dict(state0=[0, 0, 0, 0, 0, 2 * C / R]),
    dict(state0=[0, 0, 50.0, 0, 0, 0]),              # сопротивление воздуха
    dict(alpha_deg=60.0, t_max=100.0),               # разгонится по склону
    dict(m=3e-8, R=1e-4, state0=[0, 0, 1.0, 0, 0, 0]),   # вязкое сопротивление
    dict(state0=[0, 0, 1e-10, 0, 0, 0]),             # тепловое движение
    dict(state0=[0, 0, 2.0, 0, 0, 5000.0]),          # слишком быстрое вращение
    dict(R=5e-5),                                    # слишком маленькое
    dict(m=0.001, R=0.1),                            # слишком лёгкое
    dict(state0=[0, 0, np.nan, 0, 0, 0]),
    dict(shape="cube"),
    dict(shape="cylinder", state0=[0, 0, 2.0, 1.0, 0, 0]),
    dict(m=0.0),
    dict(mu=-0.1),
    dict(mu=5.0),
    dict(alpha_deg=90.0),
    dict(g=0.0),
    dict(t_max=0.0),
    dict(t_max=1e-8),
    dict(t_max=3601.0),
    dict(m=101.0),
    dict(R=1.01, m=100.0),
    dict(g=0.001),
    dict(g=101.0),
    dict(state0=[0, 0, 2.0]),
    dict(state0=[[0, 0, 2.0, 0, 0, 0]]),
])
def test_bad_params(changes):
    with pytest.raises(ValueError):
        check_params(**{**GOOD, **changes})


@pytest.mark.parametrize("changes", [
    dict(state0=[0, 0, 22.0, 22.0, 0, 0]),         # модуль скорости, а не одна компонента
    dict(state0=[0, 0, 20.0, 0, 0, 15.0 / R]),    # запас на раскрутку
    dict(alpha_deg=30.0, t_max=10.0),             # разгон за время расчёта
])
def test_speed_limit_for_heavy_body(changes):
    # для тяжёлого тела проверка воздуха пропустила бы эти скорости
    with pytest.raises(ValueError, match="оценка скорости"):
        check_params(**{**GOOD, "m": 100.0, **changes})


def test_rotation_limit_for_heavy_body():
    with pytest.raises(ValueError, match="слишком быстрое вращение"):
        check_params(**{**GOOD, "m": 100.0, "state0": [0, 0, 0, 0, 0, 21.0 / R]})


def test_speed_limit_boundary():
    check_params(**{**GOOD, "m": 100.0, "state0": [0, 0, 20.0, 0, 0, 10.0 / R]})


def test_short_run():
    t, y, _, _ = simulate(**{**GOOD, "t_max": 1e-6})
    assert t[-1] == 1e-6
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("rtol", [0.0, -1e-8, 1e-15, 1e-3, np.nan, np.inf])
def test_bad_accuracy(rtol):
    with pytest.raises(ValueError, match="rtol"):
        simulate(**GOOD, rtol=rtol)


def test_calculation_limit(monkeypatch):
    monkeypatch.setattr(model, "MAX_NFEV", 10)
    with pytest.raises(RuntimeError, match="слишком много вычислений"):
        simulate(**GOOD)


def test_time_limit(monkeypatch):
    clock = iter([0.0, model.MAX_TIME + 1])
    monkeypatch.setattr(model, "perf_counter", lambda: next(clock))
    with pytest.raises(RuntimeError, match="расчёт занял больше"):
        simulate(**GOOD)


def test_no_progress(monkeypatch):
    def stuck_solver(rhs, interval, y, **kwargs):
        return SimpleNamespace(status=1, t=np.array([interval[0]]), y=y[:, None])

    monkeypatch.setattr(model, "solve_ivp", stuck_solver)
    with pytest.raises(RuntimeError, match="не удалось продвинуться по времени"):
        simulate(**GOOD)


def test_main_reports_calculation_limit(monkeypatch, capsys):
    monkeypatch.setattr(model, "MAX_NFEV", 10)
    with pytest.raises(SystemExit) as error:
        model.main()
    assert error.value.code == 1
    assert "Расчёт остановлен: слишком много вычислений" in capsys.readouterr().out
