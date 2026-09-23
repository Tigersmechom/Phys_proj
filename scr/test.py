import sys
import os
import numpy as np

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

from M1.scr.main import simulate, G


def test_flight_time_without_air():
    v0 = 30.0
    grad = 45.0
    solution = simulate(
        "none",
        1.0,
        0.0,
        0.0,
        0.0,
        v0,
        grad
    )
    T_numeric = solution.t_events[0][0]
    angle = np.radians(grad)
    T_theory = 2 * v0 * np.sin(angle) / G
    assert abs(T_numeric - T_theory) < 1e-6


def test_distance_without_air():
    v0 = 30.0
    grad = 45.0
    solution = simulate(
        "none",
        1.0,
        0.0,
        0.0,
        0.0,
        v0,
        grad
    )
    L_numeric = solution.y_events[0][0][0]
    angle = np.radians(grad)
    L_theory = v0**2 * np.sin(2 * angle) / G
    assert abs(L_numeric - L_theory) < 1e-6


def test_max_height_without_air():
    v0 = 30.0
    grad = 45.0
    solution = simulate(
        "none",
        1.0,
        0.0,
        0.0,
        0.0,
        v0,
        grad
    )
    H_numeric = np.max(solution.y[1])
    angle = np.radians(grad)
    H_theory = (
        v0**2
        * np.sin(angle)**2
        / (2 * G)
    )
    assert abs(H_numeric - H_theory) < 1e-3

test_distance_without_air()
test_flight_time_without_air()
test_max_height_without_air()