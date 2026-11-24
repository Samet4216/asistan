"""Unit tests for SymbolicMathEngine (Phase 6).
"""

from personal_ai.tools.math_tools import SymbolicMathEngine


def test_differentiate_symbolic():
    """Verify exact symbolic differentiation and LaTeX generation."""
    res = SymbolicMathEngine.differentiate("3*x**2 + 5*sin(x)", variable="x")
    assert res.success is True
    assert "6*x + 5*cos(x)" in res.result_text
    assert "\\frac{d}{dx}" in res.latex


def test_integrate_symbolic():
    """Verify symbolic integration."""
    res = SymbolicMathEngine.integrate("2*x + exp(x)", variable="x")
    assert res.success is True
    assert "x**2 + exp(x)" in res.result_text
    assert "\\int" in res.latex


def test_eigenvalues_computation():
    """Verify matrix eigenvalue computation."""
    # Diagonal matrix with eigenvalues 2 and 5
    A = [[2.0, 0.0], [0.0, 5.0]]
    res = SymbolicMathEngine.compute_eigen(A)
    assert res.success is True
    assert "2" in res.result_text
    assert "5" in res.result_text


def test_solve_linear_system():
    """Verify Ax = b solution."""
    # 2x + y = 5, x + 3y = 5 -> x=2, y=1
    A = [[2.0, 1.0], [1.0, 3.0]]
    b = [5.0, 5.0]
    res = SymbolicMathEngine.solve_linear_system(A, b)
    assert res.success is True
    assert "2" in res.result_text
    assert "1" in res.result_text
