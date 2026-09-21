"""Configuration-driven experiments (project_spec.md section 12).

A run is fully described by one TOML file (configs/v1.toml) parsed into
frozen dataclasses. `config_hash` is a stable fingerprint of the parsed
values, recorded in every run manifest so "which settings produced this
result?" always has a definite answer.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "v1.toml"


@dataclass(frozen=True)
class WalkForwardConfig:
    first_test_year: int = 2019
    holdout_start: str = "2025-01-01"
    embargo_days: int = 5


@dataclass(frozen=True)
class ModelConfig:
    names: tuple[str, ...] = ("mean", "reversal_5d", "ols", "ridge", "hgb")
    ridge_alpha: float = 1000.0
    hgb_max_depth: int = 3
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 200
    hgb_min_samples_leaf: int = 500
    hgb_l2_regularization: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 10_000_000.0
    rebalance_every: int = 5
    quantile: float = 0.10


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 1.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 2.0
    borrow_bps_annual: float = 50.0
    adv_participation_cap: float = 0.05

    def scaled(self, multiplier: float) -> CostConfig:
        """All cost components times `multiplier` (cost-sensitivity runs).
        The ADV participation cap is a liquidity limit, not a cost, so it is unchanged."""
        return CostConfig(
            commission_bps=self.commission_bps * multiplier,
            half_spread_bps=self.half_spread_bps * multiplier,
            slippage_bps=self.slippage_bps * multiplier,
            borrow_bps_annual=self.borrow_bps_annual * multiplier,
            adv_participation_cap=self.adv_participation_cap,
        )


@dataclass(frozen=True)
class RiskConfig:
    risk_aversion: float = 5.0
    max_abs_weight: float = 0.10
    max_gross: float = 2.0
    net_tolerance: float = 0.02
    sector_net_limit: float = 0.10
    beta_limit: float = 0.05
    target_vol_annual: float = 0.10
    turnover_limit: float = 0.60
    cov_lookback: int = 60


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "v1"
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def config_hash(cfg: ExperimentConfig) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True, default=list)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _build(cls, raw: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()}
    return cls(**kwargs)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return ExperimentConfig(
        name=raw.get("name", "v1"),
        walk_forward=_build(WalkForwardConfig, raw.get("walk_forward", {})),
        models=_build(ModelConfig, raw.get("models", {})),
        portfolio=_build(PortfolioConfig, raw.get("portfolio", {})),
        costs=_build(CostConfig, raw.get("costs", {})),
        risk=_build(RiskConfig, raw.get("risk", {})),
    )
