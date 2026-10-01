"""Unit tests for Population container and PortFilter elitism guardrails."""

import numpy as np

from leo.core.population import Population, port_and_filter


def test_population_sorting_and_truncation() -> None:
    x = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5], [3.0, 3.0]])
    y = np.array([10.0, 20.0, 5.0, 30.0])

    pop = Population(x=x, y=y)
    assert len(pop) == 4
    assert pop.best_y == 5.0
    np.testing.assert_array_equal(pop.best_x, np.array([0.5, 0.5]))

    sorted_pop = pop.sorted()
    np.testing.assert_array_equal(sorted_pop.y, np.array([5.0, 10.0, 20.0, 30.0]))

    truncated = pop.truncate(2)
    assert len(truncated) == 2
    np.testing.assert_array_equal(truncated.y, np.array([5.0, 10.0]))


def test_port_and_filter() -> None:
    # Explore pool: sorted y = [1.0, 3.0, 5.0]
    x_exp = np.array([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5]])
    y_exp = np.array([1.0, 3.0, 5.0])
    explore = Population(x=x_exp, y=y_exp)

    # Exploit pool: sorted y = [2.0, 4.0, 9.0]
    x_expl = np.array([[0.2, 0.2], [0.4, 0.4], [0.9, 0.9]])
    y_expl = np.array([2.0, 4.0, 9.0])
    exploit = Population(x=x_expl, y=y_expl)

    # Port top 1 from explore to exploit: should replace y=9.0 with y=1.0
    _, updated_exploit = port_and_filter(explore, exploit, num_port=1)

    assert updated_exploit.size == 3
    # The worst candidate (index -1) should now have values from explore's top (0.1, 0.1) and loss 1.0
    np.testing.assert_array_equal(updated_exploit.x[-1], np.array([0.1, 0.1]))
    assert updated_exploit.y[-1] == 1.0
