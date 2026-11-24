"""FastAPI web server providing multi-pattern CRUD, parallel scan/backtest APIs, and dashboard UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    AppConfig,
    SimulationPattern,
    PROJECT_ROOT,
    SUPPORTED_INTERVALS,
    MODE_UNIT_SIZE,
    load_config,
    save_config,
    slugify_pattern_id,
)
from .market_data import MarketDataProvider, tse_market_status
from .simulator import TradingSimulator
from .storage import Storage

WEB_DIR = PROJECT_ROOT / "web"


class ScanRequest(BaseModel):
    mode: str = Field(default="all", description="all | pattern_id | lot1 | lot100")
    pattern_id: str | None = Field(default=None)
    catchup: bool = Field(default=True, description="PC OFF期間の欠損バーを自動補完する")
    interval: str | None = Field(default=None)
    days: int = Field(default=5, ge=1, le=55)


class BacktestRequest(BaseModel):
    compare_all: bool = Field(default=False, description="全有効パターンを並行比較測定")
    pattern_id: str | None = Field(default=None)
    mode: str = Field(default="lot1")
    interval: str | None = Field(default=None)
    days: int = Field(default=10, ge=1, le=55)
    initial_capital: float | None = Field(default=None, ge=1000.0, le=1000000000.0)


class ResetRequest(BaseModel):
    mode: str = Field(default="all")
    pattern_id: str | None = Field(default=None)
    initial_capital: float | None = Field(default=None, ge=1000.0)


class PatternCreateRequest(BaseModel):
    id: str | None = None
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(default="", max_length=200)
    enabled: bool = Field(default=True)
    strategy_type: str = Field(default="ml_walkforward")
    initial_capital: float = Field(default=100000.0, ge=1000.0, le=1000000000.0)
    unit_size: int = Field(default=1)
    watchlist_key: str = Field(default="lot1")
    interval: str = Field(default="30m")
    max_positions: int = Field(default=3, ge=1, le=20)
    max_allocation_pct: float = Field(default=0.45, ge=0.05, le=1.0)
    fee_rate: float = Field(default=0.001, ge=0.0, le=0.05)
    tax_rate: float = Field(default=0.20315, ge=0.0, le=0.5)
    slippage_bps: float = Field(default=5.0, ge=0.0, le=200.0)
    buy_score_threshold: float = Field(default=68.0, ge=30.0, le=95.0)
    strong_buy_threshold: float = Field(default=78.0, ge=40.0, le=99.0)
    sell_score_threshold: float = Field(default=25.0, ge=5.0, le=65.0)
    take_profit_pct: float = Field(default=0.045, ge=0.003, le=0.5)
    stop_loss_pct: float = Field(default=0.03, ge=0.003, le=0.5)
    trailing_activation_pct: float = Field(default=0.02, ge=0.002, le=0.5)
    trailing_stop_pct: float = Field(default=0.015, ge=0.002, le=0.5)
    close_before_eod: bool = Field(default=False)
    ml_min_pred_return: float = Field(default=0.008, ge=0.001, le=0.1)
    ml_horizon_bars: int = Field(default=8, ge=2, le=40)
    max_hold_bars: int = Field(default=12, ge=0, le=100)
    cooldown_bars: int = Field(default=6, ge=0, le=50)


class PatternUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    enabled: bool | None = None
    strategy_type: str | None = None
    initial_capital: float | None = Field(default=None, ge=1000.0, le=1000000000.0)
    unit_size: int | None = None
    watchlist_key: str | None = None
    interval: str | None = None
    max_positions: int | None = Field(default=None, ge=1, le=20)
    max_allocation_pct: float | None = Field(default=None, ge=0.05, le=1.0)
    fee_rate: float | None = Field(default=None, ge=0.0, le=0.05)
    tax_rate: float | None = Field(default=None, ge=0.0, le=0.5)
    slippage_bps: float | None = Field(default=None, ge=0.0, le=200.0)
    buy_score_threshold: float | None = Field(default=None, ge=30.0, le=95.0)
    strong_buy_threshold: float | None = Field(default=None, ge=40.0, le=99.0)
    sell_score_threshold: float | None = Field(default=None, ge=5.0, le=65.0)
    take_profit_pct: float | None = Field(default=None, ge=0.003, le=0.5)
    stop_loss_pct: float | None = Field(default=None, ge=0.003, le=0.5)
    trailing_activation_pct: float | None = Field(default=None, ge=0.002, le=0.5)
    trailing_stop_pct: float | None = Field(default=None, ge=0.002, le=0.5)
    close_before_eod: bool | None = None
    ml_min_pred_return: float | None = Field(default=None, ge=0.001, le=0.1)
    ml_horizon_bars: int | None = Field(default=None, ge=2, le=40)
    max_hold_bars: int | None = Field(default=None, ge=0, le=100)
    cooldown_bars: int | None = Field(default=None, ge=0, le=50)
    reset_portfolio: bool = Field(default=False, description="初期資金変更時などにポートフォリオを初期化する")


class ConfigUpdateRequest(BaseModel):
    initial_capital: float | None = None
    interval: str | None = None
    default_mode: str | None = None
    max_positions: int | None = None
    max_allocation_pct: float | None = None
    fee_rate: float | None = None
    slippage_bps: float | None = None
    buy_score_threshold: float | None = None
    strong_buy_threshold: float | None = None
    sell_score_threshold: float | None = None
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    trailing_activation_pct: float | None = None
    trailing_stop_pct: float | None = None
    close_before_eod: bool | None = None
    watchlists: dict[str, list[dict[str, str]]] | None = None


def create_app(
    db_path: Path | None = None,
    config_path: Path | None = None,
    provider: MarketDataProvider | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Kabushiki - 日本株監視・売買推定・複数パターン並行シミュレータ",
        version="2.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    storage = Storage(db_path=db_path)
    market_provider = provider or MarketDataProvider()

    def _get_cfg() -> AppConfig:
        return load_config(config_path)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "market_status": tse_market_status()}

    @app.get("/api/dashboard")
    def get_dashboard() -> dict[str, Any]:
        cfg = _get_cfg()
        return storage.export_dashboard_snapshot(cfg)

    @app.post("/api/ml/collect")
    def collect_ml_data() -> dict[str, Any]:
        cfg = _get_cfg()
        sim = TradingSimulator(cfg, storage, provider=market_provider)
        res = sim.collect_ml_data(intervals=("15m", "30m", "60m"), period_days=55)
        snapshot = storage.export_dashboard_snapshot(cfg)
        return {"ml": res, "dashboard": snapshot}

    # --- Multi-Pattern Management CRUD ---
    @app.get("/api/patterns")
    def list_patterns() -> dict[str, Any]:
        cfg = _get_cfg()
        return {"patterns": [p.to_dict() for p in cfg.patterns]}

    @app.post("/api/patterns")
    def create_pattern(req: PatternCreateRequest) -> dict[str, Any]:
        cfg = _get_cfg()
        if req.interval not in SUPPORTED_INTERVALS:
            raise HTTPException(status_code=400, detail=f"interval must be one of {SUPPORTED_INTERVALS}")
        if req.unit_size not in (1, 100):
            raise HTTPException(status_code=400, detail="unit_size must be 1 or 100")

        existing_ids = {p.id for p in cfg.patterns}
        new_id = slugify_pattern_id(req.id or req.name, existing_ids)
        data = req.model_dump()
        data["id"] = new_id
        pattern = SimulationPattern(**data)

        cfg.patterns.append(pattern)
        save_config(cfg, config_path)
        storage.ensure_portfolio(pattern.id, initial_capital=pattern.initial_capital, unit_size=pattern.unit_size)
        snapshot = storage.export_dashboard_snapshot(cfg)
        return {"pattern": pattern.to_dict(), "dashboard": snapshot}

    @app.put("/api/patterns/{pattern_id}")
    def update_pattern(pattern_id: str, req: PatternUpdateRequest) -> dict[str, Any]:
        cfg = _get_cfg()
        pattern = next((p for p in cfg.patterns if p.id == pattern_id), None)
        if pattern is None:
            raise HTTPException(status_code=404, detail=f"Pattern not found: {pattern_id}")

        updates = req.model_dump(exclude_none=True)
        should_reset = bool(updates.pop("reset_portfolio", False))
        old_capital = pattern.initial_capital

        for key, val in updates.items():
            if key == "interval" and val not in SUPPORTED_INTERVALS:
                raise HTTPException(status_code=400, detail=f"Unsupported interval: {val}")
            if key == "unit_size" and val not in (1, 100):
                raise HTTPException(status_code=400, detail="unit_size must be 1 or 100")
            setattr(pattern, key, val)
        pattern.__post_init__()

        save_config(cfg, config_path)

        # If reset requested OR portfolio has zero trades and capital changed, apply new capital cleanly
        existing_trades = storage.get_trades(pattern.id, limit=1)
        if should_reset or (not existing_trades and pattern.initial_capital != old_capital):
            storage.reset_portfolio(pattern.id, initial_capital=pattern.initial_capital, unit_size=pattern.unit_size)
        else:
            storage.ensure_portfolio(pattern.id, initial_capital=pattern.initial_capital, unit_size=pattern.unit_size)

        snapshot = storage.export_dashboard_snapshot(cfg)
        return {"pattern": pattern.to_dict(), "dashboard": snapshot}

    @app.delete("/api/patterns/{pattern_id}")
    def delete_pattern(pattern_id: str) -> dict[str, Any]:
        cfg = _get_cfg()
        if len(cfg.patterns) <= 1:
            raise HTTPException(status_code=400, detail="最低1つのシミュレーションパターンを残す必要があります")
        before = len(cfg.patterns)
        cfg.patterns = [p for p in cfg.patterns if p.id != pattern_id]
        if len(cfg.patterns) == before:
            raise HTTPException(status_code=404, detail=f"Pattern not found: {pattern_id}")

        save_config(cfg, config_path)
        storage.delete_portfolio(pattern_id)
        snapshot = storage.export_dashboard_snapshot(cfg)
        return {"deleted": pattern_id, "dashboard": snapshot}

    @app.post("/api/patterns/{pattern_id}/reset")
    def reset_pattern(pattern_id: str, req: ResetRequest | None = None) -> dict[str, Any]:
        cfg = _get_cfg()
        pattern = next((p for p in cfg.patterns if p.id == pattern_id), None)
        if pattern is None:
            raise HTTPException(status_code=404, detail=f"Pattern not found: {pattern_id}")

        new_cap = req.initial_capital if (req and req.initial_capital is not None) else pattern.initial_capital
        if new_cap != pattern.initial_capital:
            pattern.initial_capital = float(new_cap)
            save_config(cfg, config_path)

        storage.reset_portfolio(pattern.id, initial_capital=pattern.initial_capital, unit_size=pattern.unit_size)
        snapshot = storage.export_dashboard_snapshot(cfg)
        return {"pattern_id": pattern.id, "initial_capital": pattern.initial_capital, "dashboard": snapshot}

    # --- Parallel Scan / Catchup ---
    @app.post("/api/scan")
    def run_scan(req: ScanRequest) -> dict[str, Any]:
        cfg = _get_cfg()
        if req.interval and req.interval not in SUPPORTED_INTERVALS:
            raise HTTPException(status_code=400, detail=f"interval must be one of {SUPPORTED_INTERVALS}")

        sim = TradingSimulator(cfg, storage, provider=market_provider)
        target = req.pattern_id or req.mode

        if target == "all":
            if req.catchup:
                results = sim.run_catchup_all_patterns(lookback_days=req.days, override_interval=req.interval)
            else:
                results = [
                    sim.run_step(mode=p.id, interval=req.interval or p.interval)
                    for p in cfg.get_enabled_patterns()
                ]
        else:
            pattern = cfg.get_pattern(target)
            if pattern is None:
                raise HTTPException(status_code=404, detail=f"Pattern not found: {target}")
            if req.catchup:
                results = [sim.run_catchup_pattern(pattern, interval=req.interval, lookback_days=req.days)]
            else:
                results = [sim.run_step(mode=pattern.id, interval=req.interval or pattern.interval)]

        snapshot = storage.export_dashboard_snapshot(cfg)
        return {"results": results, "dashboard": snapshot}

    # --- Single or Parallel Compare Real-Data Backtest ---
    @app.post("/api/backtest")
    def run_backtest(req: BacktestRequest) -> dict[str, Any]:
        cfg = _get_cfg()
        if req.interval and req.interval not in SUPPORTED_INTERVALS:
            raise HTTPException(status_code=400, detail=f"interval must be one of {SUPPORTED_INTERVALS}")

        sim = TradingSimulator(cfg, storage, provider=market_provider)
        if req.compare_all:
            res = sim.run_backtest_compare(
                days=req.days,
                override_capital=req.initial_capital,
                override_interval=req.interval,
            )
            if "error" in res:
                raise HTTPException(status_code=502, detail=res["error"])
            return res

        target = req.pattern_id or req.mode
        pattern = cfg.get_pattern(target)
        if pattern is None:
            raise HTTPException(status_code=404, detail=f"Pattern not found: {target}")

        res = sim.run_backtest_pattern(
            pattern=pattern,
            days=req.days,
            override_interval=req.interval,
            override_capital=req.initial_capital,
        )
        if "error" in res:
            raise HTTPException(status_code=502, detail=res["error"])
        return res

    @app.post("/api/reset")
    def reset_sim(req: ResetRequest) -> dict[str, Any]:
        cfg = _get_cfg()
        target = req.pattern_id or req.mode
        if target == "all":
            reset_ids = []
            for p in cfg.patterns:
                cap = req.initial_capital if req.initial_capital is not None else p.initial_capital
                storage.reset_portfolio(p.id, initial_capital=cap, unit_size=p.unit_size)
                reset_ids.append(p.id)
        else:
            p = cfg.get_pattern(target)
            if p is None:
                raise HTTPException(status_code=404, detail=f"Pattern not found: {target}")
            cap = req.initial_capital if req.initial_capital is not None else p.initial_capital
            storage.reset_portfolio(p.id, initial_capital=cap, unit_size=p.unit_size)
            reset_ids = [p.id]

        snapshot = storage.export_dashboard_snapshot(cfg)
        return {"reset_modes": reset_ids, "dashboard": snapshot}

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return _get_cfg().to_dict()

    @app.put("/api/config")
    def update_config(req: ConfigUpdateRequest) -> dict[str, Any]:
        cfg = _get_cfg()
        updates = req.model_dump(exclude_none=True)
        for key, val in updates.items():
            if key == "interval" and val not in SUPPORTED_INTERVALS:
                raise HTTPException(status_code=400, detail=f"Unsupported interval: {val}")
            if key == "default_mode" and val not in MODE_UNIT_SIZE:
                raise HTTPException(status_code=400, detail=f"Unsupported mode: {val}")
            setattr(cfg, key, val)
        save_config(cfg, config_path)
        storage.export_dashboard_snapshot(cfg)
        return cfg.to_dict()

    if WEB_DIR.exists():
        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("kabushiki.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
