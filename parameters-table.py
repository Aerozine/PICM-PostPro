#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MISSING = r"\multicolumn{1}{c}{$-$}"

EXCLUDED_TOP_LEVEL = {
    "filename",
    "folder",
    "freeSurface",
    "kernelOrder",
    "max_cfl",
    "method",
    "particleInteraction",
    "ppcx",
    "ppcy",
    "refill",
    "sampling_rate",
    "solver",
    "smoke",
    "surfaceTension",
}

EXCLUDED_PREFIXES = ("write_",)

SHAPE_KEY_ORDER = (
    "condition",
    "val",
    "x",
    "y",
    "cx",
    "cy",
    "r",
    "core_r",
    "omega",
    "x1",
    "y1",
    "x2",
    "y2",
    "left_x",
    "right_x",
    "bottom_y",
    "top_y",
    "tube_width",
    "left_level",
    "right_level",
    "wall",
)

SKIP_SHAPE_KEYS = {"confine"}
GRID_INDEX_KEYS = {
    "bottom_y",
    "core_r",
    "cx",
    "cy",
    "left_level",
    "left_x",
    "r",
    "right_level",
    "right_x",
    "top_y",
    "tube_width",
    "wall",
    "x",
    "x1",
    "x2",
    "y",
    "y1",
    "y2",
}

KEY_LABELS = {
    "val": r"\mathrm{value}",
    "x": r"x_c",
    "y": r"y_c",
    "cx": r"x_c",
    "cy": r"y_c",
    "r": r"R",
    "core_r": r"R_c",
    "omega": r"\omega",
    "x1": r"x_1",
    "y1": r"y_1",
    "x2": r"x_2",
    "y2": r"y_2",
    "left_x": r"x_L",
    "right_x": r"x_R",
    "bottom_y": r"y_b",
    "top_y": r"y_t",
    "tube_width": r"w",
    "left_level": r"h_L^0",
    "right_level": r"h_R^0",
    "wall": r"w_s",
}

BLOCK_LABELS = {
    "fluid": "Liquid",
    "solid": "Solid",
    "velocityu": "u condition",
    "velocityv": "v condition",
    "velocity": "Velocity condition",
    "rankine_vortex": "Rankine vortex",
}

SHAPE_LABELS = {
    "rectangle": "rectangle",
    "cylinder": "cylinder",
    "u_tube": "U-tube",
}


@dataclass
class Row:
    section: str
    parameter: str
    value: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print a LaTeX table, in the appendix-style vertical layout, "
            "with method-agnostic parameters needed to reproduce a simulation JSON."
        )
    )
    parser.add_argument("json", type=Path, help="Input simulation JSON.")
    parser.add_argument(
        "--caption",
        default="Simulation parameters required for reproduction.",
        help="LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:simulation-parameters",
        help="LaTeX label.",
    )
    parser.add_argument(
        "--no-table-env",
        action="store_true",
        help="Print only the tabular block, without table/caption/label.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use a compact two-panel layout. By default, the script prints the single-column layout shown in the appendix table.",
    )
    return parser.parse_args()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_number(value: Any) -> str:
    if isinstance(value, bool):
        return r"\texttt{true}" if value else r"\texttt{false}"
    if isinstance(value, int):
        return f"${value}$"
    if isinstance(value, float):
        if not math.isfinite(value):
            return MISSING
        if abs(value - round(value)) < 1e-12:
            return f"${int(round(value))}$"
        abs_value = abs(value)
        if abs_value != 0.0 and (abs_value < 1e-3 or abs_value >= 1e4):
            mantissa, exponent = f"{value:.3e}".split("e")
            mantissa = f"{float(mantissa):.3g}"
            exponent = str(int(exponent))
            if mantissa == "1":
                return rf"$10^{{{exponent}}}$"
            if mantissa == "-1":
                return rf"$-10^{{{exponent}}}$"
            return rf"${mantissa}\times 10^{{{exponent}}}$"
        return f"${value:.6g}$"
    return latex_escape(str(value))


def plain_number(value: Any) -> str:
    if isinstance(value, bool):
        return r"\mathrm{true}" if value else r"\mathrm{false}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-12:
            return str(int(round(value)))
        return f"{value:.6g}"
    return r"\mathrm{" + latex_escape(str(value)) + "}"


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def safe_eval_expr(expr: str, variables: dict[str, float]) -> float | int | str:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return float(variables[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
        raise ValueError(expr)

    try:
        result = eval_node(tree)
    except Exception:
        return expr
    if abs(result - round(result)) < 1e-12:
        return int(round(result))
    return result


def variables_from_config(config: dict[str, Any]) -> dict[str, float]:
    variables: dict[str, float] = {}
    for key, value in config.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            variables[key] = float(value)
    return variables


def resolve_value(value: Any, variables: dict[str, float]) -> Any:
    if isinstance(value, str):
        return safe_eval_expr(value, variables)
    return value


def resolve_shape_value(key: str, value: Any, variables: dict[str, float]) -> Any:
    resolved = resolve_value(value, variables)
    if key in GRID_INDEX_KEYS and isinstance(resolved, (int, float)):
        return int(resolved)
    return resolved


def value_or_missing(config: dict[str, Any], key: str) -> str:
    if key not in config:
        return MISSING
    return latex_number(config[key])


def physics_value(config: dict[str, Any], key: str, enabled: bool = True) -> str:
    if not enabled or key not in config:
        return MISSING
    return latex_number(config[key])


def make_core_rows(config: dict[str, Any]) -> list[Row]:
    nx = config.get("nx")
    ny = config.get("ny")
    dx = config.get("dx")
    dy = config.get("dy")
    dt = config.get("dt")
    nt = config.get("nt")

    rows = [
        Row("Grid", r"$N_x \times N_y$", (
            rf"${int(nx)}\times {int(ny)}$"
            if isinstance(nx, (int, float)) and isinstance(ny, (int, float))
            else MISSING
        )),
        Row("Grid", r"$\Delta x \times \Delta y$ [m]", (
            rf"{latex_number(float(dx))} $\times$ {latex_number(float(dy))}"
            if isinstance(dx, (int, float)) and isinstance(dy, (int, float))
            else MISSING
        )),
        Row("Grid", r"$L_x \times L_y$ [m]", (
            rf"{latex_number(float(nx) * float(dx))} $\times$ {latex_number(float(ny) * float(dy))}"
            if all(isinstance(v, (int, float)) for v in (nx, ny, dx, dy))
            else MISSING
        )),
        Row("Time", r"$\Delta t$ [s]", latex_number(dt) if isinstance(dt, (int, float)) else MISSING),
        Row("Time", r"$N_t$", latex_number(nt) if isinstance(nt, (int, float)) else MISSING),
        Row("Time", r"$T_{\mathrm{end}}$ [s]", (
            latex_number(float(nt) * float(dt))
            if isinstance(nt, (int, float)) and isinstance(dt, (int, float))
            else MISSING
        )),
        Row("Physics", r"$\rho$ [kg\,m$^{-3}$]", value_or_missing(config, "density")),
        Row("Physics", r"$g$ [m\,s$^{-2}$]", value_or_missing(config, "gravity")),
    ]

    surface_tension = truthy(config.get("surfaceTension", "gamma" in config))
    viscosity_key = next(
        (key for key in ("viscosity", "mu", "dynamicViscosity") if key in config),
        "",
    )
    rows.extend(
        [
            Row("Physics", r"$\gamma$ [N\,m$^{-1}$]", physics_value(config, "gamma", surface_tension)),
            Row("Physics", r"$\theta_c$ [deg]", physics_value(
                config,
                "physicalContactAngleDegrees",
                surface_tension,
            )),
            Row("Physics", r"$\mu$ [Pa\,s]", value_or_missing(config, viscosity_key) if viscosity_key else MISSING),
        ]
    )
    return rows


def ordered_items(data: dict[str, Any]) -> list[tuple[str, Any]]:
    keys = [key for key in SHAPE_KEY_ORDER if key in data]
    keys.extend(key for key in data if key not in keys)
    return [(key, data[key]) for key in keys]


def format_shape_value(data: dict[str, Any], variables: dict[str, float]) -> str:
    pieces = []
    for key, raw_value in ordered_items(data):
        if key in SKIP_SHAPE_KEYS:
            continue
        if key == "condition":
            continue
        value = resolve_shape_value(key, raw_value, variables)
        label = KEY_LABELS.get(key, r"\mathrm{" + latex_escape(key) + "}")
        pieces.append(rf"{label}={plain_number(value)}")
    if not pieces:
        return MISSING
    return "$" + r",\ ".join(pieces) + "$"


def condition_suffix(data: dict[str, Any]) -> str:
    condition = data.get("condition")
    if isinstance(condition, str) and condition:
        return f" ({latex_escape(condition)})"
    return ""


def shape_label(shape: str, index: int, total: int) -> str:
    base = SHAPE_LABELS.get(shape, latex_escape(shape.replace("_", " ")))
    return f"{base} {index}" if total > 1 else base


def rows_from_shape_collection(
    section: str,
    block: Any,
    variables: dict[str, float],
) -> list[Row]:
    rows: list[Row] = []
    if not isinstance(block, dict):
        return rows

    for shape, raw_shapes in block.items():
        shapes = raw_shapes if isinstance(raw_shapes, list) else [raw_shapes]
        if not all(isinstance(item, dict) for item in shapes):
            continue
        total = len(shapes)
        for idx, shape_data in enumerate(shapes, start=1):
            parameter = shape_label(shape, idx, total) + condition_suffix(shape_data)
            rows.append(Row(section, parameter, format_shape_value(shape_data, variables)))
    return rows


def rows_from_scalar_block(
    section: str,
    block: dict[str, Any],
    variables: dict[str, float],
) -> list[Row]:
    return [Row(section, "parameters" + condition_suffix(block), format_shape_value(block, variables))]


def make_geometry_rows(config: dict[str, Any]) -> list[Row]:
    variables = variables_from_config(config)
    rows: list[Row] = []

    for block_key in ("fluid", "solid"):
        block_rows = rows_from_shape_collection(
            BLOCK_LABELS[block_key],
            config.get(block_key),
            variables,
        )
        rows.extend(block_rows or [Row(BLOCK_LABELS[block_key], "geometry", MISSING)])

    for key, value in config.items():
        if key in {"fluid", "solid"} or key in EXCLUDED_TOP_LEVEL:
            continue
        if key.startswith(EXCLUDED_PREFIXES):
            continue
        if not isinstance(value, dict):
            continue

        section = BLOCK_LABELS.get(key, latex_escape(key.replace("_", " ")))
        block_rows = rows_from_shape_collection(section, value, variables)
        if block_rows:
            rows.extend(block_rows)
        else:
            rows.extend(rows_from_scalar_block(section, value, variables))

    return rows


def row_lines(rows: list[Row]) -> list[str]:
    lines = []
    current_section = ""
    for row in rows:
        section = latex_escape(row.section) if row.section != current_section else ""
        current_section = row.section
        lines.append(
            rf"{section} & {row.parameter} & {row.value} \\"
        )
    return lines


def compact_row_cells(row: Row | None, previous_section: str) -> tuple[list[str], str]:
    if row is None:
        return [r"\multicolumn{3}{c}{}"], previous_section
    section = latex_escape(row.section) if row.section != previous_section else ""
    return [section, row.parameter, row.value], row.section


def compact_row_lines(left_rows: list[Row], right_rows: list[Row]) -> list[str]:
    n_lines = max(len(left_rows), len(right_rows))
    lines = []
    left_section = ""
    right_section = ""

    for index in range(n_lines):
        left = left_rows[index] if index < len(left_rows) else None
        left_cells, left_section = compact_row_cells(left, left_section)
        right = right_rows[index] if index < len(right_rows) else None
        right_cells, right_section = compact_row_cells(right, right_section)
        lines.append(" & ".join(left_cells + right_cells) + r" \\")

    return lines


def render_table(
    rows: list[Row],
    right_rows: list[Row],
    caption: str,
    label: str,
    include_table_env: bool,
    compact: bool,
) -> str:
    lines: list[str] = []
    if include_table_env:
        lines.extend(
            [
                r"\begin{table}[htbp]",
                r"\centering",
                r"\tiny" if compact else r"\small",
            ]
        )

    lines.extend([r"\setlength{\tabcolsep}{10pt}", r"\renewcommand{\arraystretch}{1.05}"])
    if compact:
        lines.extend(
            [
                r"\begin{tabularx}{\linewidth}{@{}llX@{\hspace{0.75em}}llX@{}}",
                r"\toprule",
                (
                    r"\textbf{Block} & \textbf{Parameter} & \textbf{Value} & "
                    r"\textbf{Block} & \textbf{Parameter} & \textbf{Value} \\"
                ),
                r"\midrule",
            ]
        )
        lines.extend(compact_row_lines(rows, right_rows))
        lines.extend([r"\bottomrule", r"\end{tabularx}"])
    else:
        all_rows = rows + right_rows
        lines.extend(
            [
                r"\begin{tabular}{@{}lll@{}}",
                r"\toprule",
                r"\textbf{Block} & \textbf{Parameter} & \textbf{Value} \\",
                r"\midrule",
            ]
        )
        lines.extend(row_lines(all_rows))
        lines.extend([r"\bottomrule", r"\end{tabular}"])

    if include_table_env:
        lines.extend(
            [
                rf"\caption{{{latex_escape(caption)}}}",
                rf"\label{{{latex_escape(label)}}}",
                r"\end{table}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with args.json.open() as f:
        config = json.load(f)

    core_rows = make_core_rows(config)
    geometry_rows = make_geometry_rows(config)
    if args.compact:
        left_rows = [row for row in core_rows if row.section != "Physics"]
        right_rows = geometry_rows + [row for row in core_rows if row.section == "Physics"]
    else:
        left_rows = core_rows
        right_rows = geometry_rows

    print(
        render_table(
            left_rows,
            right_rows,
            args.caption,
            args.label,
            not args.no_table_env,
            args.compact,
        )
    )


if __name__ == "__main__":
    main()
