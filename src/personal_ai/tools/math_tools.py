"""Symbolic Mathematics & Numerical Tools (Phase 6).

Provides deterministic symbolic calculus, linear algebra, and differential equations
via SymPy with formatted LaTeX output.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import sympy as sp


@dataclass
class MathToolResult:
    """Represents output of a symbolic calculation with LaTeX rendering."""
    success: bool
    operation: str
    result_text: str
    latex: str
    error: Optional[str] = None


class SymbolicMathEngine:
    """Performs exact mathematical computations for the Mathematics Agent."""

    @staticmethod
    def differentiate(expression: str, variable: str = "x") -> MathToolResult:
        """Computes exact derivative of an expression."""
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)
            deriv = sp.diff(expr, var)
            return MathToolResult(
                success=True,
                operation=f"d/d{variable}({expression})",
                result_text=str(deriv),
                latex=f"\\frac{{d}}{{d{variable}}}\\left({sp.latex(expr)}\\right) = {sp.latex(deriv)}",
            )
        except Exception as e:
            return MathToolResult(success=False, operation="diff", result_text="", latex="", error=str(e))

    @staticmethod
    def integrate(expression: str, variable: str = "x") -> MathToolResult:
        """Computes indefinite or definite integral."""
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression)
            integral = sp.integrate(expr, var)
            return MathToolResult(
                success=True,
                operation=f"integrate({expression}, {variable})",
                result_text=str(integral),
                latex=f"\\int {sp.latex(expr)}\\,d{variable} = {sp.latex(integral)} + C",
            )
        except Exception as e:
            return MathToolResult(success=False, operation="integrate", result_text="", latex="", error=str(e))

    @staticmethod
    def compute_eigen(matrix_rows: List[List[float]]) -> MathToolResult:
        """Computes eigenvalues and eigenvectors of a square matrix."""
        try:
            mat = sp.Matrix(matrix_rows)
            eigenvals = mat.eigenvals()
            eigen_summary = [f"λ = {sp.latex(val)} (katlılık: {mult})" for val, mult in eigenvals.items()]
            return MathToolResult(
                success=True,
                operation="eigenvalues",
                result_text=str(eigenvals),
                latex=", \\quad ".join(eigen_summary),
            )
        except Exception as e:
            return MathToolResult(success=False, operation="eigen", result_text="", latex="", error=str(e))

    @staticmethod
    def solve_linear_system(matrix_A: List[List[float]], vector_b: List[float]) -> MathToolResult:
        """Solves matrix equation Ax = b."""
        try:
            A = sp.Matrix(matrix_A)
            b = sp.Matrix(vector_b)
            sol = A.LUsolve(b)
            return MathToolResult(
                success=True,
                operation="solve_linear_system",
                result_text=str(sol),
                latex=f"x = {sp.latex(sol)}",
            )
        except Exception as e:
            return MathToolResult(success=False, operation="solve_linear_system", result_text="", latex="", error=str(e))
