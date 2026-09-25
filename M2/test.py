import numpy as np
import pytest
from main import simulate, check_params, analytic_head_on, analytic_oblique, energy, potential, momentum, C


def test_head_on_equal_mass():
    # равные массы -> обмен скоростями
    sol = simulate([0, 0, 3.0, 0, 3.0, 0, 0, 0], 5.0, 1.0, 1.0, 0.5, 0.5, 5e4, "hooke")
    assert abs(sol.y[2, -1]) < 1e-3
    assert abs(sol.y[6, -1] - 3.0) < 1e-3


def test_head_on_unequal_mass():
    sol = simulate([0, 0, 3.0, 0, 3.0, 0, 0, 0], 5.0, 1.0, 2.0, 0.5, 0.5, 5e4, "hooke")
    v1p, v2p = analytic_head_on(3.0, 1.0, 2.0)
    assert abs(sol.y[2, -1] - v1p) < 1e-3
    assert abs(sol.y[6, -1] - v2p) < 1e-3


def test_head_on_hertz():
    sol = simulate([0, 0, 3.0, 0, 3.0, 0, 0, 0], 5.0, 1.0, 1.0, 0.5, 0.5, 5e5, "hertz")
    assert abs(sol.y[2, -1]) < 1e-3
    assert abs(sol.y[6, -1] - 3.0) < 1e-3


def test_both_moving():
    sol = simulate([0, 0, 2.0, 0, 2.0, 0, -1.5, 0], 3.0, 1.0, 3.0, 0.5, 0.3, 5e5, "hooke")
    v1p, v2p = analytic_head_on(2.0, 1.0, 3.0, -1.5)
    assert abs(sol.y[2, -1] - v1p) < 1e-3
    assert abs(sol.y[6, -1] - v2p) < 1e-3


def test_energy_momentum():
    state0 = [0, 0.4, 3.0, 0, 3.0, 0, 0, 0]
    sol = simulate(state0, 5.0, 1.0, 1.7, 0.5, 0.5, 5e4, "hooke")
    final = sol.y[:, -1]
    E0 = energy(state0, 1.0, 1.7)
    assert abs(energy(final, 1.0, 1.7) - E0) / E0 < 1e-3
    p0 = momentum(state0, 1.0, 1.7)
    p1 = momentum(final, 1.0, 1.7)
    assert abs(p1[0] - p0[0]) < 1e-3
    assert abs(p1[1] - p0[1]) < 1e-3


def test_energy_during_contact():
    state0 = [0, 0.3, 3.0, 0, 3.0, 0, 0, 0]
    sol = simulate(state0, 3.0, 1.0, 1.5, 0.5, 0.5, 5e4, "hertz")
    E0 = energy(state0, 1.0, 1.5)
    for s in sol.y.T:
        E = energy(s, 1.0, 1.5) + potential(s, 0.5, 0.5, 5e4, "hertz")
        assert abs(E - E0) / E0 < 1e-5


def test_oblique_90_degrees():
    # равные массы -> разлетаются под прямым углом
    sol = simulate([0, 0.5, 3.0, 0, 3.0, 0, 0, 0], 3.0, 1.0, 1.0, 0.5, 0.5, 5e5, "hooke")
    u1 = sol.y[2:4, -1]
    u2 = sol.y[6:8, -1]
    assert abs(np.dot(u1, u2)) < 1e-5


def test_oblique_vs_theory():
    sol = simulate([0, 0.7, 3.0, 0, 3.0, 0, 0, 0], 3.0, 2.0, 1.0, 0.4, 0.6, 5e5, "hooke")
    v1p, v2p = analytic_oblique(3.0, 0.7, 2.0, 1.0, 1.0)
    assert np.linalg.norm(sol.y[2:4, -1] - v1p) < 0.05
    assert np.linalg.norm(sol.y[6:8, -1] - v2p) < 0.05


def test_miss():
    sol = simulate([0, 1.2, 3.0, 0, 3.0, 0, 0, 0], 3.0, 1.0, 1.0, 0.5, 0.5, 5e4, "hooke")
    assert len(sol.t_events[0]) == 0
    assert abs(sol.y[2, -1] - 3.0) < 1e-9
    assert abs(sol.y[6, -1]) < 1e-9


def test_contact_time_hooke():
    sol = simulate([0, 0, 3.0, 0, 3.0, 0, 0, 0], 3.0, 1.0, 2.0, 0.5, 0.5, 5e4, "hooke")
    mu = 2.0 / 3.0
    t_c = sol.t_events[0][1] - sol.t_events[0][0]
    assert abs(t_c - np.pi * np.sqrt(mu / 5e4)) < 1e-6


def test_contact_time_hertz():
    # t_c ~ v^(-1/5)
    sol1 = simulate([0, 0, 1.0, 0, 3.0, 0, 0, 0], 4.0, 1.0, 1.0, 0.5, 0.5, 5e5, "hertz")
    sol2 = simulate([0, 0, 2.0, 0, 3.0, 0, 0, 0], 4.0, 1.0, 1.0, 0.5, 0.5, 5e5, "hertz")
    t1 = sol1.t_events[0][1] - sol1.t_events[0][0]
    t2 = sol2.t_events[0][1] - sol2.t_events[0][0]
    assert abs(t2 / t1 - 2 ** (-0.2)) < 1e-3


def test_bigger_k_is_closer_to_theory():
    _, v2p = analytic_oblique(3.0, 0.6, 1.0, 1.0, 1.0)
    errors = []
    for k in [1e4, 1e5, 1e6]:
        sol = simulate([0, 0.6, 3.0, 0, 3.0, 0, 0, 0], 3.0, 1.0, 1.0, 0.5, 0.5, k, "hooke")
        errors.append(np.linalg.norm(sol.y[6:8, -1] - v2p))
    assert errors[0] > errors[1] > errors[2]


def test_bad_speed():
    with pytest.raises(ValueError):
        check_params([0, 0, C, 0, 3.0, 0, 0, 0], 5.0, 1.0, 1.0, 0.5, 0.5, 5e4, "hooke")
    with pytest.raises(ValueError):
        check_params([0, 0, 0.5 * C, 0, 3.0, 0, 0, 0], 5.0, 1.0, 1.0, 0.5, 0.5, 5e4, "hooke")


def test_overlap():
    with pytest.raises(ValueError):
        check_params([0, 0, 3.0, 0, 0.5, 0, 0, 0], 5.0, 1.0, 1.0, 0.5, 0.5, 5e4, "hooke")


def test_bad_mass_and_k():
    with pytest.raises(ValueError):
        check_params([0, 0, 3.0, 0, 3.0, 0, 0, 0], 5.0, -1.0, 1.0, 0.5, 0.5, 5e4, "hooke")
    with pytest.raises(ValueError):
        check_params([0, 0, 3.0, 0, 3.0, 0, 0, 0], 5.0, 1.0, 1.0, 0.5, 0.5, 10.0, "hooke")
    with pytest.raises(ValueError):
        check_params([0, 0, 3.0, 0, 3.0, 0, 0, 0], 5.0, 1.0, 1.0, 0.5, 0.5, 5e4, "abc")
