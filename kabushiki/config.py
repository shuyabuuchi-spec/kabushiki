"""Configuration loading, saving, stock universes, and multi-pattern definitions."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.json"
USER_CONFIG_PATH = PROJECT_ROOT / "data" / "user_config.json"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "kabushiki.db"
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "dashboard_snapshot.json"
WEB_SNAPSHOT_PATH = PROJECT_ROOT / "web" / "data" / "dashboard_snapshot.json"

SUPPORTED_INTERVALS = ("5m", "10m", "15m", "30m", "60m")

MODE_UNIT_SIZE = {
    "lot1": 1,
    "lot100": 100,
}

MODE_LABELS = {
    "lot1": "1株単位（単元未満株・かぶミニ想定）",
    "lot100": "100株単位（単元株・10万円以内銘柄）",
}


@dataclass
class StockInfo:
    symbol: str
    name: str
    sector: str = "その他"


@dataclass
class SimulationPattern:
    id: str
    name: str
    description: str = ""
    enabled: bool = True
    initial_capital: float = 100000.0
    unit_size: int = 1
    watchlist_key: str = "lot1"
    interval: str = "15m"
    max_positions: int = 3
    max_allocation_pct: float = 0.4
    fee_rate: float = 0.0
    slippage_bps: float = 5.0
    buy_score_threshold: float = 66.0
    strong_buy_threshold: float = 76.0
    sell_score_threshold: float = 30.0
    take_profit_pct: float = 0.035
    stop_loss_pct: float = 0.02
    trailing_activation_pct: float = 0.015
    trailing_stop_pct: float = 0.01
    close_before_eod: bool = False

    def __post_init__(self) -> None:
        if self.interval not in SUPPORTED_INTERVALS:
            self.interval = "15m"
        self.unit_size = 100 if int(self.unit_size) >= 100 else 1
        self.initial_capital = max(1000.0, float(self.initial_capital))
        if self.watchlist_key not in ("lot1", "lot100"):
            self.watchlist_key = "lot100" if self.unit_size >= 100 else "lot1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_strategy_config(self, watchlists: dict[str, list[dict[str, str]]]) -> "AppConfig":
        """Build an AppConfig view representing this specific pattern's rules."""
        return AppConfig(
            initial_capital=self.initial_capital,
            interval=self.interval,
            default_mode=self.watchlist_key,
            max_positions=self.max_positions,
            max_allocation_pct=self.max_allocation_pct,
            fee_rate=self.fee_rate,
            slippage_bps=self.slippage_bps,
            buy_score_threshold=self.buy_score_threshold,
            strong_buy_threshold=self.strong_buy_threshold,
            sell_score_threshold=self.sell_score_threshold,
            take_profit_pct=self.take_profit_pct,
            stop_loss_pct=self.stop_loss_pct,
            trailing_activation_pct=self.trailing_activation_pct,
            trailing_stop_pct=self.trailing_stop_pct,
            close_before_eod=self.close_before_eod,
            watchlists=watchlists,
        )


def slugify_pattern_id(raw: str, existing_ids: set[str] | None = None) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip().lower()).strip("_")
    if not cleaned:
        cleaned = "pattern"
    if not cleaned.startswith("p_"):
        cleaned = f"p_{cleaned}"
    cleaned = cleaned[:40]
    existing = existing_ids or set()
    if cleaned not in existing:
        return cleaned
    idx = 2
    while f"{cleaned}_{idx}" in existing:
        idx += 1
    return f"{cleaned}_{idx}"


@dataclass
class AppConfig:
    initial_capital: float = 100000.0
    interval: str = "15m"
    default_mode: str = "lot1"
    max_positions: int = 3
    max_allocation_pct: float = 0.4
    fee_rate: float = 0.0
    slippage_bps: float = 5.0
    buy_score_threshold: float = 66.0
    strong_buy_threshold: float = 76.0
    sell_score_threshold: float = 30.0
    take_profit_pct: float = 0.035
    stop_loss_pct: float = 0.02
    trailing_activation_pct: float = 0.015
    trailing_stop_pct: float = 0.01
    close_before_eod: bool = False
    patterns: list[SimulationPattern] = field(default_factory=list)
    watchlists: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: list[SimulationPattern] = []
        valid_p_fields = set(SimulationPattern.__dataclass_fields__.keys())
        for item in self.patterns:
            if isinstance(item, SimulationPattern):
                normalized.append(item)
            elif isinstance(item, dict):
                filtered = {k: v for k, v in item.items() if k in valid_p_fields}
                normalized.append(SimulationPattern(**filtered))
        self.patterns = normalized

        if not self.patterns:
            # Ensure at least default lot1 & lot100 patterns exist
            self.patterns = [
                SimulationPattern(
                    id="p_100k_15m_balanced",
                    name="標準バランス 10万円（1株・15分足）",
                    initial_capital=self.initial_capital,
                    unit_size=1,
                    watchlist_key="lot1",
                    interval=self.interval,
                    max_positions=self.max_positions,
                    max_allocation_pct=self.max_allocation_pct,
                    fee_rate=self.fee_rate,
                    slippage_bps=self.slippage_bps,
                    buy_score_threshold=self.buy_score_threshold,
                    strong_buy_threshold=self.strong_buy_threshold,
                    sell_score_threshold=self.sell_score_threshold,
                    take_profit_pct=self.take_profit_pct,
                    stop_loss_pct=self.stop_loss_pct,
                    trailing_activation_pct=self.trailing_activation_pct,
                    trailing_stop_pct=self.trailing_stop_pct,
                    close_before_eod=self.close_before_eod,
                ),
                SimulationPattern(
                    id="p_100k_15m_lot100",
                    name="単元株 10万円（100株・15分足）",
                    initial_capital=self.initial_capital,
                    unit_size=100,
                    watchlist_key="lot100",
                    interval=self.interval,
                    max_positions=self.max_positions,
                    max_allocation_pct=max(self.max_allocation_pct, 0.5),
                    fee_rate=self.fee_rate,
                    slippage_bps=self.slippage_bps,
                    buy_score_threshold=self.buy_score_threshold,
                    strong_buy_threshold=self.strong_buy_threshold,
                    sell_score_threshold=self.sell_score_threshold,
                    take_profit_pct=self.take_profit_pct,
                    stop_loss_pct=self.stop_loss_pct,
                    trailing_activation_pct=self.trailing_activation_pct,
                    trailing_stop_pct=self.trailing_stop_pct,
                    close_before_eod=self.close_before_eod,
                ),
            ]

    def get_watchlist(self, key: str) -> list[StockInfo]:
        items = self.watchlists.get(key) or self.watchlists.get("lot1", [])
        return [
            StockInfo(
                symbol=item["symbol"],
                name=item.get("name", item["symbol"]),
                sector=item.get("sector", "その他"),
            )
            for item in items
        ]

    def get_pattern(self, pattern_id: str) -> SimulationPattern | None:
        for p in self.patterns:
            if p.id == pattern_id:
                return p
        # Backward compatibility alias for legacy "lot1" / "lot100"
        if pattern_id == "lot1":
            for p in self.patterns:
                if p.unit_size == 1:
                    return p
        if pattern_id == "lot100":
            for p in self.patterns:
                if p.unit_size >= 100:
                    return p
        return None

    def get_enabled_patterns(self) -> list[SimulationPattern]:
        return [p for p in self.patterns if p.enabled]

    def symbol_meta_map(self) -> dict[str, StockInfo]:
        mapping: dict[str, StockInfo] = {}
        for mode_items in self.watchlists.values():
            for item in mode_items:
                sym = item["symbol"]
                if sym not in mapping:
                    mapping[sym] = StockInfo(
                        symbol=sym,
                        name=item.get("name", sym),
                        sector=item.get("sector", "その他"),
                    )
        return mapping

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["patterns"] = [p.to_dict() for p in self.patterns]
        return d


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from default_config.json overlaid with user_config.json."""
    base_data: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        base_data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    target_path = config_path or USER_CONFIG_PATH
    if target_path.exists():
        user_data = json.loads(target_path.read_text(encoding="utf-8"))
        merged = copy.deepcopy(base_data)
        for key, value in user_data.items():
            if key == "watchlists" and isinstance(value, dict):
                merged.setdefault("watchlists", {}).update(value)
            else:
                merged[key] = value
        base_data = merged

    valid_fields = set(AppConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in base_data.items() if k in valid_fields}
    cfg = AppConfig(**filtered)
    if cfg.interval not in SUPPORTED_INTERVALS:
        cfg.interval = "15m"
    if cfg.default_mode not in MODE_UNIT_SIZE:
        cfg.default_mode = "lot1"
    return cfg


def save_config(cfg: AppConfig, config_path: Path | None = None) -> Path:
    """Persist user configuration overrides (including parallel patterns) to JSON."""
    target_path = config_path or USER_CONFIG_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_path
