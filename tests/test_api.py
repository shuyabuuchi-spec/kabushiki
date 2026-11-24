"""API tests for the multi-pattern FastAPI dashboard backend."""

import asyncio
from pathlib import Path
import httpx

from kabushiki.server import create_app
from tests.test_simulator import _synthetic_wave_bars


class FakeMarketProvider:
    def fetch_symbol(self, symbol: str, interval: str = "15m", period_days: int = 10):
        base = 400.0 if symbol in ("9432.T", "4689.T", "9434.T", "8410.T") else 2500.0
        return _synthetic_wave_bars(base, n_bars=70, phase=float(abs(hash(symbol)) % 7))

    def fetch_many(self, symbols, interval: str = "15m", period_days: int = 10):
        return {s: self.fetch_symbol(s, interval=interval, period_days=period_days) for s in symbols}


def test_dashboard_multi_pattern_crud_scan_and_compare_api(tmp_path: Path):
    app = create_app(
        db_path=tmp_path / "api.db",
        config_path=tmp_path / "user_config.json",
        provider=FakeMarketProvider(),
    )

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Health & static UI
            r_health = await client.get("/api/health")
            assert r_health.status_code == 200
            assert r_health.json()["status"] == "ok"

            r_home = await client.get("/")
            assert r_home.status_code == 200
            assert "Kabushiki" in r_home.text

            # Initial dashboard has 4 default parallel patterns
            r_dash = await client.get("/api/dashboard")
            assert r_dash.status_code == 200
            dash = r_dash.json()
            assert len(dash["pattern_order"]) == 4

            # Create a custom 5th pattern with 300,000 JPY and 10m interval
            r_create = await client.post(
                "/api/patterns",
                json={
                    "name": "カスタム30万円・10分足",
                    "initial_capital": 300000,
                    "interval": "10m",
                    "unit_size": 1,
                    "watchlist_key": "lot1",
                    "buy_score_threshold": 62,
                    "take_profit_pct": 0.04,
                    "stop_loss_pct": 0.02,
                    "close_before_eod": True,
                },
            )
            assert r_create.status_code == 200
            created = r_create.json()["pattern"]
            new_pid = created["id"]
            assert created["initial_capital"] == 300000.0
            assert created["interval"] == "10m"
            assert len(r_create.json()["dashboard"]["pattern_order"]) == 5

            # Run parallel scan across ALL enabled patterns
            r_scan = await client.post("/api/scan", json={"mode": "all", "catchup": True, "days": 5})
            assert r_scan.status_code == 200
            scan_data = r_scan.json()
            assert len(scan_data["results"]) == 5
            assert scan_data["dashboard"]["patterns"][new_pid]["initial_capital"] == 300000.0

            # Run parallel compare backtest across ALL enabled patterns
            r_cmp = await client.post(
                "/api/backtest",
                json={"compare_all": True, "days": 7},
            )
            assert r_cmp.status_code == 200
            cmp_data = r_cmp.json()
            assert cmp_data["compare_mode"] is True
            assert cmp_data["pattern_count"] == 5
            assert len(cmp_data["results"]) == 5

            # Update pattern initial capital to 500,000 JPY & reset
            r_upd = await client.put(
                f"/api/patterns/{new_pid}",
                json={"initial_capital": 500000, "interval": "30m", "reset_portfolio": True},
            )
            assert r_upd.status_code == 200
            assert r_upd.json()["pattern"]["initial_capital"] == 500000.0
            assert r_upd.json()["dashboard"]["patterns"][new_pid]["total_equity"] == 500000.0

            # Reset endpoint with custom capital
            r_reset = await client.post(
                f"/api/patterns/{new_pid}/reset",
                json={"initial_capital": 250000},
            )
            assert r_reset.status_code == 200
            assert r_reset.json()["dashboard"]["patterns"][new_pid]["total_equity"] == 250000.0

            # Delete custom pattern
            r_del = await client.delete(f"/api/patterns/{new_pid}")
            assert r_del.status_code == 200
            assert new_pid not in r_del.json()["dashboard"]["patterns"]
            assert len(r_del.json()["dashboard"]["pattern_order"]) == 4

    asyncio.run(_run())
