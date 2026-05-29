"""AlphaForge-style RL factor mining via evolutionary search.

Generates random formulaic alpha expressions (operator trees over
Alpha158 base features), evaluates them by RankIC against forward
returns, and evolves the population across generations.

Usage:
    python -c "
    import sys; sys.path.insert(0, '.')
    from models.alpha_forge import mine_factors
    results = mine_factors(n_generations=3, population_size=20, top_k=5)
    print(f'Top {len(results)} factors found')
    for r in results:
        print(f'  IC={r[\"rank_ic\"]:.4f} ICIR={r[\"icir\"]:.2f} | {r[\"expression\"]}')
    "
"""
from __future__ import annotations

import copy
import logging
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# Suppress noisy numpy/scipy warnings during mass evaluation
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*ConstantInputWarning.*")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Operator Library
# ---------------------------------------------------------------------------

# Operators are split by arity for tree construction.
# Unary operators take a single Series; binary take two.
# Window operators also carry a default window parameter.

# ---------------------------------------------------------------------------
# 1a. Operator Library (expanded from 14→38 ops, inspired by QuantaAlpha's 54)
# ---------------------------------------------------------------------------

# === Unary operators (cross-sectional, per date) ===
UNARY_OPS: dict[str, dict[str, Any]] = {
    "rank": {
        "fn": lambda x: x.groupby(level=0).rank(pct=True),
        "desc": "rank({x})",
    },
    "neg": {
        "fn": lambda x: -x,
        "desc": "-({x})",
    },
    "abs": {
        "fn": lambda x: x.abs(),
        "desc": "abs({x})",
    },
    "log": {
        "fn": lambda x: np.log1p(x.abs()) * np.sign(x),
        "desc": "log({x})",
    },
    "sign": {
        "fn": lambda x: np.sign(x),
        "desc": "sign({x})",
    },
    # QuantaAlpha-inspired additions
    "inv": {
        "fn": lambda x: 1.0 / x.replace(0, np.nan),
        "desc": "inv({x})",
    },
    "square": {
        "fn": lambda x: x ** 2,
        "desc": "square({x})",
    },
    "sqrt": {
        "fn": lambda x: np.sqrt(x.abs()),
        "desc": "sqrt({x})",
    },
    "cs_zscore": {
        "fn": lambda x: x.groupby(level=0).transform(
            lambda g: (g - g.mean()) / (g.std() + 1e-8)
        ),
        "desc": "cs_zscore({x})",
    },
    "cs_demean": {
        "fn": lambda x: x.groupby(level=0).transform(lambda g: g - g.mean()),
        "desc": "cs_demean({x})",
    },
}

# === Window (time-series) operators ===
WINDOW_OPS: dict[str, dict[str, Any]] = {
    "delta": {
        "fn": lambda x, d: x.groupby(level=1).diff(d),
        "windows": [1, 3, 5, 10, 20],
        "desc": "delta({x},{d})",
    },
    "ts_mean": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(1, d // 2)).mean()
        ),
        "windows": [5, 10, 20],
        "desc": "ts_mean({x},{d})",
    },
    "ts_std": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(1, d // 2)).std()
        ),
        "windows": [5, 10, 20],
        "desc": "ts_std({x},{d})",
    },
    "ts_max": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(1, d // 2)).max()
        ),
        "windows": [5, 10, 20],
        "desc": "ts_max({x},{d})",
    },
    "ts_min": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(1, d // 2)).min()
        ),
        "windows": [5, 10, 20],
        "desc": "ts_min({x},{d})",
    },
    # QuantaAlpha-inspired additions
    "ts_rank": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(1, d // 2)).apply(
                lambda s: pd.Series(s).rank(pct=True).iloc[-1], raw=False
            )
        ),
        "windows": [5, 10, 20],
        "desc": "ts_rank({x},{d})",
    },
    "ts_skew": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(3, d // 2)).skew()
        ),
        "windows": [10, 20],
        "desc": "ts_skew({x},{d})",
    },
    "ts_kurt": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(4, d // 2)).kurt()
        ),
        "windows": [20],
        "desc": "ts_kurt({x},{d})",
    },
    "ts_decay": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(1, d // 2)).apply(
                lambda s: np.average(s, weights=np.arange(1, len(s) + 1)), raw=True
            )
        ),
        "windows": [5, 10, 20],
        "desc": "ts_decay({x},{d})",
    },
    "ts_delay": {
        "fn": lambda x, d: x.groupby(level=1).shift(d),
        "windows": [1, 3, 5, 10],
        "desc": "ts_delay({x},{d})",
    },
    "ts_pctchange": {
        "fn": lambda x, d: x.groupby(level=1).pct_change(d),
        "windows": [1, 3, 5, 10, 20],
        "desc": "ts_pctchange({x},{d})",
    },
    "ts_sum": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(1, d // 2)).sum()
        ),
        "windows": [5, 10, 20],
        "desc": "ts_sum({x},{d})",
    },
    "ts_median": {
        "fn": lambda x, d: x.groupby(level=1).transform(
            lambda g: g.rolling(d, min_periods=max(1, d // 2)).median()
        ),
        "windows": [5, 10, 20],
        "desc": "ts_median({x},{d})",
    },
}

# === Binary operators ===
BINARY_OPS: dict[str, dict[str, Any]] = {
    "add": {
        "fn": lambda a, b: a + b,
        "desc": "({a}+{b})",
    },
    "sub": {
        "fn": lambda a, b: a - b,
        "desc": "({a}-{b})",
    },
    "mul": {
        "fn": lambda a, b: a * b,
        "desc": "({a}*{b})",
    },
    "div": {
        "fn": lambda a, b: a / b.replace(0, np.nan),
        "desc": "({a}/{b})",
    },
    # QuantaAlpha-inspired additions
    "max2": {
        "fn": lambda a, b: pd.DataFrame({"a": a, "b": b}).max(axis=1),
        "desc": "max({a},{b})",
    },
    "min2": {
        "fn": lambda a, b: pd.DataFrame({"a": a, "b": b}).min(axis=1),
        "desc": "min({a},{b})",
    },
    "gt": {
        "fn": lambda a, b: (a > b).astype(float),
        "desc": "gt({a},{b})",
    },
    "lt": {
        "fn": lambda a, b: (a < b).astype(float),
        "desc": "lt({a},{b})",
    },
}

# ---------------------------------------------------------------------------
# 1b. AST Constraints (inspired by QuantaAlpha's semantic consistency)
# ---------------------------------------------------------------------------

MAX_DEPTH = 4             # prevent runaway nesting
MAX_NODES = 15            # limit total expression complexity
MAX_SAME_OP_CHAIN = 2     # prevent rank(rank(rank(...)))

# Banned compositions: applying these ops in sequence is meaningless
BANNED_CHAINS = {
    ("rank", "rank"),
    ("abs", "abs"),
    ("sign", "sign"),
    ("neg", "neg"),
    ("log", "log"),       # log(log(x)) produces extreme NaN/values
    ("sqrt", "sqrt"),
    ("cs_zscore", "cs_zscore"),
    ("cs_zscore", "rank"),  # both normalize → redundant
    ("rank", "cs_zscore"),
}

def _count_nodes(expr) -> int:
    """Count total nodes in expression tree."""
    if not expr.children:
        return 1
    return 1 + sum(_count_nodes(c) for c in expr.children)

def _check_banned_chain(expr) -> bool:
    """Return True if expression contains a banned operator chain."""
    if expr.node_type in ("unary", "window") and expr.children:
        child = expr.children[0]
        if child.node_type in ("unary", "window"):
            pair = (expr.operator, child.operator)
            if pair in BANNED_CHAINS:
                return True
    for c in expr.children:
        if _check_banned_chain(c):
            return True
    return False

def validate_expression(expr) -> bool:
    """Validate expression against AST constraints. Returns True if valid."""
    if expr.depth() > MAX_DEPTH:
        return False
    if _count_nodes(expr) > MAX_NODES:
        return False
    if _check_banned_chain(expr):
        return False
    return True

# Base features — Alpha158 columns available in the cache.
# We pick a representative subset that covers price, volume, momentum, etc.
BASE_FEATURES = [
    "KMID", "KLEN", "OPEN0", "HIGH0", "LOW0",
    "ROC5", "ROC10", "ROC20",
    "MA5", "MA10", "MA20",
    "STD5", "STD10", "STD20",
    "RANK5", "RANK10", "RANK20",
    "RSV5", "RSV10", "RSV20",
    "CORR5", "CORR10", "CORR20",
    "CNTP5", "CNTP10", "CNTP20",
    "VMA5", "VMA10", "VMA20",
    "VSTD5", "VSTD10", "VSTD20",
    "WVMA5", "WVMA10", "WVMA20",
]

# ---------------------------------------------------------------------------
# 2. Expression Tree
# ---------------------------------------------------------------------------

@dataclass
class AlphaExpression:
    """A composable alpha expression tree node."""

    node_type: str  # "leaf", "unary", "window", "binary"
    operator: str | None = None  # operator name
    feature: str | None = None  # leaf feature name
    window: int | None = None  # window parameter for window ops
    children: list["AlphaExpression"] = field(default_factory=list)

    # ---- evaluation ----

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate the expression tree on a (datetime, instrument) DataFrame."""
        if self.node_type == "leaf":
            return data[self.feature].copy()

        if self.node_type == "unary":
            child_val = self.children[0].evaluate(data)
            fn = UNARY_OPS[self.operator]["fn"]
            return fn(child_val)

        if self.node_type == "window":
            child_val = self.children[0].evaluate(data)
            fn = WINDOW_OPS[self.operator]["fn"]
            return fn(child_val, self.window)

        if self.node_type == "binary":
            left = self.children[0].evaluate(data)
            right = self.children[1].evaluate(data)
            fn = BINARY_OPS[self.operator]["fn"]
            return fn(left, right)

        raise ValueError(f"Unknown node_type: {self.node_type}")

    # ---- string representation ----

    def to_string(self) -> str:
        if self.node_type == "leaf":
            return self.feature

        if self.node_type == "unary":
            template = UNARY_OPS[self.operator]["desc"]
            return template.format(x=self.children[0].to_string())

        if self.node_type == "window":
            template = WINDOW_OPS[self.operator]["desc"]
            return template.format(
                x=self.children[0].to_string(), d=self.window
            )

        if self.node_type == "binary":
            template = BINARY_OPS[self.operator]["desc"]
            return template.format(
                a=self.children[0].to_string(),
                b=self.children[1].to_string(),
            )

        return "???"

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def __repr__(self) -> str:
        return f"AlphaExpr({self.to_string()})"


# ---------------------------------------------------------------------------
# 3. Random Expression Generation
# ---------------------------------------------------------------------------

def _make_leaf() -> AlphaExpression:
    return AlphaExpression(
        node_type="leaf", feature=random.choice(BASE_FEATURES)
    )


def random_expression(max_depth: int = 3) -> AlphaExpression:
    """Generate a random expression tree up to *max_depth* levels.

    Applies AST constraints (QuantaAlpha-inspired):
    - Validates against banned chains, max depth/nodes
    - Retries up to 10 times if constraints violated
    """
    for _ in range(10):
        expr = _random_expression_inner(max_depth)
        if validate_expression(expr):
            return expr
    # Fallback: simple expression
    return _make_leaf()


def _random_expression_inner(max_depth: int) -> AlphaExpression:
    """Internal random expression generator (no validation)."""
    if max_depth <= 0:
        return _make_leaf()

    r = random.random()

    if r < 0.25:
        return _make_leaf()

    if r < 0.50:
        op = random.choice(list(UNARY_OPS))
        child = _random_expression_inner(max_depth - 1)
        return AlphaExpression(
            node_type="unary", operator=op, children=[child]
        )

    if r < 0.75:
        op = random.choice(list(WINDOW_OPS))
        win = random.choice(WINDOW_OPS[op]["windows"])
        child = _random_expression_inner(max_depth - 1)
        return AlphaExpression(
            node_type="window", operator=op, window=win, children=[child]
        )

    op = random.choice(list(BINARY_OPS))
    left = _random_expression_inner(max_depth - 1)
    right = _random_expression_inner(max_depth - 1)
    return AlphaExpression(
        node_type="binary", operator=op, children=[left, right]
    )


# ---------------------------------------------------------------------------
# 4. Mutation
# ---------------------------------------------------------------------------

def mutate(expr: AlphaExpression, p: float = 0.3) -> AlphaExpression:
    """Return a mutated copy of *expr* that satisfies AST constraints.

    With probability *p* at each node, apply one of:
      - swap leaf feature
      - change operator (same arity)
      - change window parameter
      - wrap in a new unary/window op

    Retries up to 5 times if mutation violates constraints.
    """
    for _ in range(5):
        candidate = copy.deepcopy(expr)
        candidate = _mutate_node(candidate, p)
        if validate_expression(candidate):
            return candidate
    return copy.deepcopy(expr)  # return original if can't find valid mutation


def _mutate_node(node: AlphaExpression, p: float) -> AlphaExpression:
    if random.random() < p:
        return _apply_mutation(node)

    # recurse into children
    node.children = [_mutate_node(c, p) for c in node.children]
    return node


def _apply_mutation(node: AlphaExpression) -> AlphaExpression:
    roll = random.random()

    if node.node_type == "leaf":
        if roll < 0.5:
            # swap feature
            node.feature = random.choice(BASE_FEATURES)
        else:
            # wrap in unary/window
            if random.random() < 0.5:
                op = random.choice(list(UNARY_OPS))
                return AlphaExpression(
                    node_type="unary", operator=op, children=[node]
                )
            else:
                op = random.choice(list(WINDOW_OPS))
                win = random.choice(WINDOW_OPS[op]["windows"])
                return AlphaExpression(
                    node_type="window", operator=op, window=win, children=[node]
                )

    elif node.node_type == "unary":
        node.operator = random.choice(list(UNARY_OPS))

    elif node.node_type == "window":
        if roll < 0.5:
            # change window
            node.window = random.choice(WINDOW_OPS[node.operator]["windows"])
        else:
            node.operator = random.choice(list(WINDOW_OPS))
            node.window = random.choice(WINDOW_OPS[node.operator]["windows"])

    elif node.node_type == "binary":
        node.operator = random.choice(list(BINARY_OPS))

    return node


# ---------------------------------------------------------------------------
# 5. Crossover
# ---------------------------------------------------------------------------

def crossover(
    parent_a: AlphaExpression, parent_b: AlphaExpression
) -> AlphaExpression:
    """Produce a child by swapping a random subtree from parent_b into parent_a."""
    child = copy.deepcopy(parent_a)
    donor_subtree = copy.deepcopy(_random_subtree(parent_b))

    # Replace a random leaf or subtree in child
    _replace_random_subtree(child, donor_subtree)
    return child


def _random_subtree(node: AlphaExpression) -> AlphaExpression:
    """Pick a random subtree (possibly the root)."""
    nodes = _collect_nodes(node)
    return random.choice(nodes)


def _collect_nodes(node: AlphaExpression) -> list[AlphaExpression]:
    result = [node]
    for c in node.children:
        result.extend(_collect_nodes(c))
    return result


def _replace_random_subtree(
    node: AlphaExpression, replacement: AlphaExpression
) -> bool:
    """Replace a random child of *node* with *replacement*. Returns True if done."""
    if not node.children:
        return False

    # 50% chance to replace here, else recurse
    if random.random() < 0.5 or all(
        not c.children for c in node.children
    ):
        idx = random.randrange(len(node.children))
        node.children[idx] = replacement
        return True

    random.shuffle(node.children)
    for c in node.children:
        if _replace_random_subtree(c, replacement):
            return True

    # fallback: replace first child
    node.children[0] = replacement
    return True


# ---------------------------------------------------------------------------
# 6. Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate_expression(
    expr: AlphaExpression,
    data: pd.DataFrame,
    forward_returns: pd.Series,
) -> dict | None:
    """Evaluate an expression and return IC metrics.

    Returns None if the factor is degenerate (all NaN, zero variance, etc.).
    """
    try:
        factor = expr.evaluate(data)
    except Exception:
        return None

    # Clean up infinities
    factor = factor.replace([np.inf, -np.inf], np.nan)

    # Coverage check — need at least 30% non-NaN
    valid_mask = factor.notna() & forward_returns.notna()
    coverage = valid_mask.mean()
    if coverage < 0.10:
        return None

    # Compute daily RankIC (Spearman correlation per day)
    combined = pd.DataFrame({"factor": factor, "ret": forward_returns})
    combined = combined.dropna()

    if len(combined) < 100:
        return None

    daily_ic = (
        combined.groupby(level=0)
        .apply(lambda g: g["factor"].corr(g["ret"], method="spearman") if len(g) > 10 else np.nan)
    )
    daily_ic = daily_ic.dropna()

    if len(daily_ic) < 5:
        return None

    rank_ic = daily_ic.mean()
    ic_std = daily_ic.std()
    icir = rank_ic / ic_std if ic_std > 1e-8 else 0.0

    # Autocorrelation of the factor (proxy for turnover)
    try:
        autocorr = (
            factor.groupby(level=1)
            .apply(lambda g: g.autocorr(1))
            .mean()
        )
    except Exception:
        autocorr = np.nan

    return {
        "rank_ic": rank_ic,
        "icir": icir,
        "ic_std": ic_std,
        "coverage": coverage,
        "autocorr": autocorr,
        "n_days": len(daily_ic),
        "expression": expr.to_string(),
        "expr_obj": expr,
    }


# ---------------------------------------------------------------------------
# 7. Evolutionary Search
# ---------------------------------------------------------------------------

def mine_factors(
    n_generations: int = 10,
    population_size: int = 50,
    top_k: int = 10,
    max_depth: int = 3,
    n_months: int = 3,
    cache_path: str | Path | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Run evolutionary alpha mining.

    Args:
        n_generations: Number of evolution generations.
        population_size: Expressions per generation.
        top_k: Keep top-K survivors each generation.
        max_depth: Maximum expression tree depth.
        n_months: How many months of recent data to use.
        cache_path: Path to feature cache parquet. Defaults to champion cache.
        seed: Random seed for reproducibility.

    Returns:
        List of top factor dictionaries sorted by |rank_ic| descending.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ---- Load data ----
    t0 = time.time()
    if cache_path is None:
        project_root = Path(__file__).resolve().parents[1]
        cache_path = (
            project_root / "data" / "storage"
            / "feature_cache_174_holder_regime_ma.parquet"
        )

    logger.info(f"Loading cache from {cache_path}")
    needed_cols = list(set(BASE_FEATURES)) + ["__label_5d"]
    # Only read columns we actually need
    data = pd.read_parquet(cache_path, columns=needed_cols)

    # Slice to recent N months
    dates = data.index.get_level_values(0).unique().sort_values()
    if n_months > 0:
        cutoff = dates[-1] - pd.DateOffset(months=n_months)
        data = data.loc[data.index.get_level_values(0) >= cutoff]
        dates = data.index.get_level_values(0).unique().sort_values()

    forward_returns = data.pop("__label_5d")
    logger.info(
        f"Data loaded: {data.shape[0]:,} rows, {len(dates)} days, "
        f"{data.shape[1]} features ({time.time()-t0:.1f}s)"
    )

    # ---- Initial population ----
    population: list[AlphaExpression] = [
        random_expression(max_depth=max_depth) for _ in range(population_size)
    ]

    all_results: list[dict] = []

    for gen in range(n_generations):
        t1 = time.time()
        gen_results: list[dict] = []

        for expr in population:
            # Limit tree depth to avoid blowup
            if expr.depth() > 5:
                continue
            result = evaluate_expression(expr, data, forward_returns)
            if result is not None:
                gen_results.append(result)

        # Combine with historical best
        all_results.extend(gen_results)

        # De-duplicate by expression string, keep best |IC|
        seen: dict[str, dict] = {}
        for r in all_results:
            key = r["expression"]
            if key not in seen or abs(r["rank_ic"]) > abs(seen[key]["rank_ic"]):
                seen[key] = r
        all_results = list(seen.values())

        # Sort by absolute rank IC
        all_results.sort(key=lambda r: abs(r["rank_ic"]), reverse=True)

        # Top-K survivors
        survivors = all_results[:top_k]
        survivor_exprs = [r["expr_obj"] for r in survivors]

        best_ic = survivors[0]["rank_ic"] if survivors else 0.0
        n_valid = len(gen_results)
        logger.info(
            f"Gen {gen+1}/{n_generations}: "
            f"{n_valid}/{len(population)} valid | "
            f"best |IC|={abs(best_ic):.4f} | "
            f"{time.time()-t1:.1f}s"
        )

        if gen < n_generations - 1:
            # Build next generation
            next_pop: list[AlphaExpression] = []

            # Elites survive unchanged
            next_pop.extend(copy.deepcopy(survivor_exprs))

            # Fill rest with mutations and crossovers
            while len(next_pop) < population_size:
                r = random.random()
                if r < 0.3 and len(survivor_exprs) >= 2:
                    # crossover
                    a, b = random.sample(survivor_exprs, 2)
                    child = crossover(a, b)
                    next_pop.append(child)
                elif r < 0.7 and survivor_exprs:
                    # mutate survivor
                    parent = random.choice(survivor_exprs)
                    child = mutate(parent, p=0.3)
                    next_pop.append(child)
                else:
                    # fresh random
                    next_pop.append(random_expression(max_depth=max_depth))

            population = next_pop

    # Final output: strip expr_obj (not serializable)
    output = []
    for r in all_results[:top_k]:
        out = {k: v for k, v in r.items() if k != "expr_obj"}
        output.append(out)

    elapsed = time.time() - t0
    logger.info(f"Mining complete: {len(output)} factors in {elapsed:.1f}s")
    return output


# ---------------------------------------------------------------------------
# 8. CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    results = mine_factors(n_generations=5, population_size=30, top_k=10, seed=42)
    print(f"\n{'='*70}")
    print(f"Top {len(results)} mined alpha factors")
    print(f"{'='*70}")
    for i, r in enumerate(results, 1):
        print(
            f"  {i:2d}. IC={r['rank_ic']:+.4f}  ICIR={r['icir']:+.2f}  "
            f"cov={r['coverage']:.1%}  autocorr={r['autocorr']:.2f}  "
            f"| {r['expression']}"
        )
