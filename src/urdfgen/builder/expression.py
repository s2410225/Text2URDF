"""A tiny arithmetic evaluator for parametric module ``affects`` expressions.

Module definitions are static data, but they come from YAML files that a user
edits, so the expressions in ``parameters[].affects[].expr`` are still input.
``eval`` is not used: the expression is parsed with :mod:`ast` and only a
fixed set of node types is accepted. The single free variable is ``value``.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping

VARIABLE_NAME = "value"


class ExpressionError(ValueError):
    """Raised when an expression is malformed or uses a forbidden construct."""


_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def evaluate(expr: str, variables: Mapping[str, float]) -> float:
    """Evaluate ``expr`` over ``variables``, allowing only arithmetic."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        msg = f"cannot parse expression {expr!r}: {exc.msg}"
        raise ExpressionError(msg) from exc
    return _eval_node(tree.body, variables, expr)


def _eval_node(node: ast.expr, variables: Mapping[str, float], expr: str) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            msg = f"expression {expr!r}: only numeric constants are allowed"
            raise ExpressionError(msg)
        return float(node.value)

    if isinstance(node, ast.Name):
        try:
            return float(variables[node.id])
        except KeyError as exc:
            known = ", ".join(sorted(variables)) or "(none)"
            msg = f"expression {expr!r}: unknown variable {node.id!r} (known: {known})"
            raise ExpressionError(msg) from exc

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            msg = f"expression {expr!r}: operator {type(node.op).__name__} is not allowed"
            raise ExpressionError(msg)
        left = _eval_node(node.left, variables, expr)
        right = _eval_node(node.right, variables, expr)
        try:
            return float(op(left, right))
        except (ArithmeticError, ValueError) as exc:
            msg = f"expression {expr!r}: {exc}"
            raise ExpressionError(msg) from exc

    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            msg = f"expression {expr!r}: unary {type(node.op).__name__} is not allowed"
            raise ExpressionError(msg)
        return float(unary(_eval_node(node.operand, variables, expr)))

    if isinstance(node, ast.Call):
        return _eval_call(node, variables, expr)

    msg = f"expression {expr!r}: {type(node).__name__} is not allowed"
    raise ExpressionError(msg)


def _eval_call(node: ast.Call, variables: Mapping[str, float], expr: str) -> float:
    if not isinstance(node.func, ast.Name) or node.keywords:
        msg = f"expression {expr!r}: only plain calls to {sorted(_FUNCTIONS)} are allowed"
        raise ExpressionError(msg)
    func = _FUNCTIONS.get(node.func.id)
    if func is None or len(node.args) != 1:
        msg = f"expression {expr!r}: unknown or misused function {node.func.id!r}"
        raise ExpressionError(msg)
    try:
        return float(func(_eval_node(node.args[0], variables, expr)))
    except (ArithmeticError, ValueError) as exc:
        msg = f"expression {expr!r}: {exc}"
        raise ExpressionError(msg) from exc
