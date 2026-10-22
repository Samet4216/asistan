"""
Tensor Dry-Run Sandbox Module (Faz 10).
Performs AST whitelist security checks and dry-runs PyTorch nn.Module snippets on CPU.
Verifies forward tensor shapes, backpropagation gradient flow, and parameter count
without touching GPU VRAM.
"""

import ast
import inspect
import sys
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Allowed top-level modules
ALLOWED_MODULES = {"torch", "torch.nn", "torch.nn.functional", "math"}

# Disallowed dangerous attribute lookups
FORBIDDEN_ATTRIBUTES = {
    "__subclasses__", "__globals__", "__code__", "__builtins__",
    "__class__", "__bases__", "__import__"
}

# Disallowed dangerous builtin functions
FORBIDDEN_CALLS = {
    "eval", "exec", "open", "__import__", "compile",
    "globals", "locals", "getattr", "setattr", "delattr", "system"
}

import builtins

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    base = name.split(".")[0]
    if base not in {"torch", "math"}:
        raise ImportError(f"Security: Import of '{name}' is forbidden in sandbox.")
    return __import__(name, globals, locals, fromlist, level)


# Safe builtins whitelist
SAFE_BUILTINS = {
    "__import__": safe_import,
    "__build_class__": builtins.__build_class__,
    "range": range,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "zip": zip,
    "enumerate": enumerate,
    "super": super,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "print": lambda *args, **kwargs: None,  # Mute prints in sandbox
    "Exception": Exception,
    "ValueError": ValueError,
    "RuntimeError": RuntimeError
}


class SecurityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod not in {"torch", "math"}:
                self.errors.append(f"Security violation: Import of '{alias.name}' is forbidden.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if not node.module or (not node.module.startswith("torch") and node.module != "math"):
            self.errors.append(f"Security violation: 'from {node.module} import ...' is forbidden.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.errors.append(f"Security violation: Access to '{node.attr}' is forbidden.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self.errors.append(f"Security violation: Call to '{node.func.id}()' is forbidden.")
        self.generic_visit(node)


def validate_ast(code_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validates Python source code against the strict security AST whitelist.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

    visitor = SecurityVisitor()
    visitor.visit(tree)

    if visitor.errors:
        return False, "; ".join(visitor.errors)

    return True, None


def dry_run_module(
    code_str: str,
    input_shape: Tuple[int, ...] = (2, 64, 32, 32),
    module_name: Optional[str] = None,
    init_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes PyTorch module dry-run on CPU:
    1. Validates AST security.
    2. Compiles in sandbox.
    3. Instantiates module.
    4. Runs forward pass with dummy tensor.
    5. Runs backward pass to verify autograd graph.
    6. Measures parameter count.
    """
    # 1. AST Whitelist Verification
    is_safe, sec_err = validate_ast(code_str)
    if not is_safe:
        return {
            "passed": False,
            "error_stage": "security",
            "error_message": sec_err
        }

    # 2. Prepare Sandbox Execution Environment
    safe_globals = {
        "__name__": "__tensor_sandbox__",
        "__builtins__": SAFE_BUILTINS,
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math
    }
    safe_locals: Dict[str, Any] = {}

    try:
        compiled_code = compile(code_str, "<tensor_sandbox>", "exec")
        exec(compiled_code, safe_globals, safe_locals)
    except Exception as e:
        return {
            "passed": False,
            "error_stage": "compilation",
            "error_message": f"Compilation failed: {type(e).__name__}: {e}"
        }

    # 3. Locate target nn.Module
    target_cls = None
    if module_name and module_name in safe_locals:
        target_cls = safe_locals[module_name]
    else:
        for val in safe_locals.values():
            if inspect.isclass(val) and issubclass(val, nn.Module) and val is not nn.Module:
                target_cls = val
                break

    if target_cls is None:
        return {
            "passed": False,
            "error_stage": "discovery",
            "error_message": "No torch.nn.Module subclass found in the provided code."
        }

    # 4. Instantiate Module with args
    instantiation_kwargs = init_kwargs or {}
    try:
        # Check constructor params
        sig = inspect.signature(target_cls.__init__)
        params = sig.parameters
        # Auto-supply common DL channel parameters if not provided
        kwargs = {}
        for p_name, p in params.items():
            if p_name in ("self", "args", "kwargs"):
                continue
            if p_name in instantiation_kwargs:
                kwargs[p_name] = instantiation_kwargs[p_name]
            elif p.default is not inspect.Parameter.empty:
                continue
            elif "channel" in p_name.lower() or "dim" in p_name.lower():
                kwargs[p_name] = input_shape[1] if len(input_shape) > 1 else 64
            elif "num_classes" in p_name.lower():
                kwargs[p_name] = 10
            else:
                kwargs[p_name] = 64

        model = target_cls(**kwargs).cpu()
        model.train()
    except Exception as e:
        return {
            "passed": False,
            "error_stage": "instantiation",
            "error_message": f"Failed to instantiate {target_cls.__name__}: {type(e).__name__}: {e}"
        }

    # 5. Measure Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 6. Forward Pass
    try:
        dummy_input = torch.randn(*input_shape, device="cpu", requires_grad=True)
        out = model(dummy_input)
    except Exception as e:
        return {
            "passed": False,
            "error_stage": "forward",
            "error_message": f"Forward pass failed on input shape {list(input_shape)}: {type(e).__name__}: {e}",
            "param_count": total_params
        }

    # Capture output shape
    if isinstance(out, torch.Tensor):
        out_shape = list(out.shape)
    elif isinstance(out, (tuple, list)):
        out_shape = [list(o.shape) if isinstance(o, torch.Tensor) else type(o).__name__ for o in out]
    else:
        out_shape = str(type(out))

    # 7. Backward Pass (Autograd Graph Verification)
    try:
        loss = out.sum() if isinstance(out, torch.Tensor) else sum(o.sum() for o in out if isinstance(o, torch.Tensor))
        loss.backward()
        backward_passed = True
    except Exception as e:
        return {
            "passed": False,
            "error_stage": "backward",
            "error_message": f"Backward pass failed: {type(e).__name__}: {e}",
            "input_shape": list(input_shape),
            "output_shape": out_shape,
            "param_count": total_params
        }

    return {
        "passed": True,
        "module_name": target_cls.__name__,
        "input_shape": list(input_shape),
        "output_shape": out_shape,
        "backward_passed": backward_passed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": "cpu"
    }


if __name__ == "__main__":
    print("[*] Testing answer/shape_check.py...")

    # Test 1: Coordinate Attention Implementation
    coord_att_code = """
import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out
"""

    res1 = dry_run_module(coord_att_code, input_shape=(2, 64, 32, 32))
    print("\n[+] Test 1 (CoordAtt) Dry-Run Result:")
    for k, v in res1.items():
        print(f"  * {k}: {v}")

    # Test 2: Malicious Code Blocked by AST
    malicious_code = """
import os
import torch.nn as nn

class AttackModule(nn.Module):
    def forward(self, x):
        os.system("echo hacked")
        return x
"""
    res2 = dry_run_module(malicious_code)
    print("\n[+] Test 2 (Malicious Code Blocked) Result:")
    for k, v in res2.items():
        print(f"  * {k}: {v}")

    # Test 3: Dimension Mismatch caught cleanly
    broken_code = """
import torch
import torch.nn as nn

class BrokenModule(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.fc = nn.Linear(32, 128)  # expects 32, input will have 64

    def forward(self, x):
        return self.fc(x)
"""
    res3 = dry_run_module(broken_code, input_shape=(2, 64))
    print("\n[+] Test 3 (Broken Shape Caught) Result:")
    for k, v in res3.items():
        print(f"  * {k}: {v}")
