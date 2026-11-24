"""Unit tests for CodeValidator (Phase 6).
"""

from personal_ai.tools.code_tools import CodeValidator


def test_validate_valid_python_code():
    """Verify correct AST parsing and architecture extraction."""
    py_code = """
import math
from typing import List

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def update(self, error: float, dt: float) -> float:
        return self.kp * error
"""
    res = CodeValidator.validate_python(py_code)
    assert res.valid is True
    assert "PIDController" in res.classes_defined
    assert "__init__" in res.functions_defined
    assert "update" in res.functions_defined
    assert "math" in res.imports_found


def test_validate_invalid_python_code():
    """Verify detection of syntax errors with line localization."""
    broken_code = """
def compute():
    x = [1, 2, 3
    return x
"""
    res = CodeValidator.validate_python(broken_code)
    assert res.valid is False
    assert res.error_message is not None
    assert "SyntaxError" in res.error_message


def test_validate_cpp_balanced():
    """Verify balanced C++ delimiters."""
    cpp_code = """
#include <iostream>
#include <memory>

int main() {
    auto ptr = std::make_unique<int>(42);
    if (*ptr > 0) {
        std::cout << *ptr << std::endl;
    }
    return 0;
}
"""
    res = CodeValidator.validate_cpp(cpp_code)
    assert res.valid is True


def test_validate_cpp_unbalanced():
    """Verify unclosed delimiter in C++."""
    broken_cpp = """
int main() {
    if (x > 0) {
        return 1;
    // Missing closing brace
}
"""
    res = CodeValidator.validate_cpp(broken_cpp)
    assert res.valid is False
    assert "Unclosed" in res.error_message or "Mismatched" in res.error_message
