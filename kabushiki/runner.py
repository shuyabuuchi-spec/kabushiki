"""CLI runner for multi-pattern parallel monitoring, PC-OFF catch-up, backtesting, and daemon."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_mod

from .config import (
    load_config,
    SUPPORTED_INTERVALS,
    DEFAULT_DB_PATH,
    DEFAULT_SNAPSHOT_PATH,
    WEB_SNAPSHOT_PATH,
    PROJECT_ROOT,
)
from .crypto_utils import encrypt_file, decrypt_file
from .market_data import is_tse_market_open, tse_market_status
from .simulator import TradingSimulator
from .storage import Storage

ENC_DB_PATH = PROJECT_ROOT / "data" / "kabushiki.db.enc"
ENC_WEB_SNAPSHOT_PATH = PROJECT_ROOT / "web" / "data" / "dashboard_snapshot.enc.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kabushiki",
        description="日本株リアルタイム監視・売買タイミング推定・複数パターン並行シミュレータ",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # catchup (recommended for GitHub Actions & PC boot)
    p_catch = sub.add_parser("catchup", help="全有効パターンを並行でPC OFF補完＋最新売買シミュレート")
    p_catch.add_argument("--pattern", default="all", help="パターンID または all（全有効パターン）")
    p_catch.add_argument("--mode", default=None, help="互換オプション（--pattern と同義）")
    p_catch.add_argument("--interval", choices=list(SUPPORTED_INTERVALS), default=None)
    p_catch.add_argument("--days", type=int, default=7, help="最大遡及日数")
    p_catch.add_argument("--only-when-open", action="store_true", help="東証取引時間外はスキップ")

    # step
    p_step = sub.add_parser("step", help="最新バー1本のみ監視・シミュレーション実行")
    p_step.add_argument("--pattern", default="all")
    p_step.add_argument("--interval", choices=list(SUPPORTED_INTERVALS), default=None)

    # backtest
    p_bt = sub.add_parser("backtest", help="実データ（5m/10m/15m/30m/60m足）で運用結果を測定・並行比較")
    p_bt.add_argument("--pattern", default=None, help="単一パターンID（未指定かつ--mode未指定時は--compare-all）")
    p_bt.add_argument("--mode", default=None, help="互換用: lot1 / lot100")
    p_bt.add_argument("--compare-all", action="store_true", help="全有効パターンを並行比較測定")
    p_bt.add_argument("--interval", choices=list(SUPPORTED_INTERVALS), default=None)
    p_bt.add_argument("--days", type=int, default=14)
    p_bt.add_argument("--capital", type=float, default=None, help="初期資金を上書き（円）")
    p_bt.add_argument("--json", action="store_true", help="JSON形式で出力")

    # daemon
    p_daemon = sub.add_parser("daemon", help="ローカルPC上で全有効パターンの自動監視ループを起動")
    p_daemon.add_argument("--interval-minutes", type=int, default=10)

    # status / patterns
    sub.add_parser("status", help="全パターンの現在のシミュレーション資産状況を一覧表示")
    sub.add_parser("patterns", help="登録されているシミュレーションパターン一覧を表示")
    sub.add_parser("export-snapshot", help="ダッシュボード用静的JSONスナップショットを再出力")

    # encrypt-state / decrypt-state (uses DASHBOARD_PASSPHRASE env var)
    p_enc = sub.add_parser("encrypt-state", help="DBとダッシュボードJSONをAES-256-GCMで暗号化")
    p_enc.add_argument("--remove-plaintext", action="store_true", help="暗号化後に平文ファイルを削除")
    sub.add_parser("decrypt-state", help="暗号化済みDB(kabushiki.db.enc)を復号")

    # reset
    p_reset = sub.add_parser("reset", help="シミュレーション資産を初期資金にリセット")
    p_reset.add_argument("--pattern", default="all", help="パターンID または all")
    p_reset.add_argument("--mode", default=None, help="互換用: lot1 / lot100 / all")
    p_reset.add_argument("--capital", type=float, default=None)

    args = parser.parse_args(argv)
    cfg = load_config()
    storage = Storage()
    sim = TradingSimulator(cfg, storage)

    if args.command == "catchup":
        if args.only_when_open and not is_tse_market_open(include_buffer_minutes=15):
            status = tse_market_status()
            print(json.dumps({"skipped": True, "market_status": status}, ensure_ascii=False))
            return 0
        target = args.mode or args.pattern
        if target == "all":
            results = sim.run_catchup_all_patterns(lookback_days=args.days, override_interval=args.interval)
        else:
            p = cfg.get_pattern(target)
            if p is None:
                print(f"Pattern not found: {target}", file=sys.stderr)
                return 1
            results = [sim.run_catchup_pattern(p, interval=args.interval, lookback_days=args.days)]
        print(json.dumps({"results": results, "market_status": tse_market_status()}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "step":
        targets = cfg.get_enabled_patterns() if args.pattern == "all" else [cfg.get_pattern(args.pattern)]
        targets = [p for p in targets if p is not None]
        results = [sim.run_step(mode=p.id, interval=args.interval or p.interval) for p in targets]
        print(json.dumps({"results": results, "market_status": tse_market_status()}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "backtest":
        if args.compare_all or (args.pattern is None and args.mode is None):
            res = sim.run_backtest_compare(
                days=args.days,
                override_capital=args.capital,
                override_interval=args.interval,
            )
            if args.json:
                print(json.dumps(res, ensure_ascii=False, indent=2))
                return 0
            if "error" in res:
                print(f"Error: {res['error']}", file=sys.stderr)
                return 1
            print(f"=== 全パターン並行 実データ比較結果（直近{args.days}日間） ===")
            for rank, item in enumerate(res["results"], 1):
                bm = f" / 日経 {item['benchmark_nikkei225_pct']:+.2f}%" if item.get("benchmark_nikkei225_pct") is not None else ""
                print(
                    f"#{rank} {item['pattern_name']} [{item['interval']} / {item['unit_size']}株] "
                    f"¥{item['initial_capital']:,.0f} → ¥{item['final_equity']:,.0f} "
                    f"({item['net_pnl']:+,.0f}円 / {item['return_pct']:+.2f}%{bm}) | "
                    f"取引{item['total_trades']}回 勝率{item['win_rate']:.1f}% PF{item['profit_factor']} 最大DD-{item['max_drawdown_pct']:.2f}%"
                )
            return 0

        target_id = args.pattern or args.mode or "lot1"
        p = cfg.get_pattern(target_id)
        if p is None:
            print(f"Pattern not found: {target_id}", file=sys.stderr)
            return 1
        res = sim.run_backtest_pattern(
            pattern=p,
            days=args.days,
            override_interval=args.interval,
            override_capital=args.capital,
        )
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            if "error" in res:
                print(f"Error: {res['error']}", file=sys.stderr)
                return 1
            print(f"=== 実データシミュレーション結果 ({res['pattern_name']}) ===")
            print(f"期間      : {res['start_time']} ～ {res['end_time']} ({res['bars_simulated']}本 / {res['interval']}足)")
            print(f"元手      : ¥{res['initial_capital']:,.0f}")
            print(f"最終資産  : ¥{res['final_equity']:,.0f} ({res['net_pnl']:+,.0f}円 / {res['return_pct']:+.2f}%)")
            print(f"確定損益  : ¥{res['realized_pnl']:+,.0f} / 含み損益: ¥{res['unrealized_pnl']:+,.0f}")
            print(f"取引回数  : {res['total_trades']}回 (決済 {res['closed_trades']}回 / 勝率 {res['win_rate']:.1f}% / PF {res['profit_factor']})")
            print(f"最大DD    : -{res['max_drawdown_pct']:.2f}%")
            if res.get("benchmark_nikkei225_pct") is not None:
                print(f"日経平均  : {res['benchmark_nikkei225_pct']:+.2f}% (同期間)")
        return 0

    if args.command == "daemon":
        sleep_sec = max(60, args.interval_minutes * 60)
        print(f"Kabushiki 並行監視デーモン開始: {args.interval_minutes}分間隔 / 有効パターン数={len(cfg.get_enabled_patterns())}")
        while True:
            try:
                if is_tse_market_open(include_buffer_minutes=10):
                    cfg_fresh = load_config()
                    sim_fresh = TradingSimulator(cfg_fresh, storage)
                    res = sim_fresh.run_catchup_all_patterns()
                    print(json.dumps(res, ensure_ascii=False))
                else:
                    print(f"[待機中] {tse_market_status()['session']}")
            except KeyboardInterrupt:
                print("停止しました。")
                return 0
            except Exception as exc:
                print(f"エラー: {exc}", file=sys.stderr)
            time_mod.sleep(sleep_sec)

    if args.command == "status":
        snap = storage.export_dashboard_snapshot(cfg)
        for pid in snap["pattern_order"]:
            d = snap["patterns"][pid]
            state_txt = "稼働" if d["enabled"] else "停止"
            print(
                f"[{state_txt}] {d['mode_label']} ({d['interval']}/{d['unit_size']}株) | "
                f"元手¥{d['initial_capital']:,.0f} → 資産¥{d['total_equity']:,.0f} "
                f"({d['net_pnl']:+,.0f}円 / {d['return_pct']:+.2f}%) | "
                f"保有{len(d['positions'])}銘柄 | 取引{d['trade_count']}回 | 勝率{d['win_rate']}%"
            )
        return 0

    if args.command == "patterns":
        for p in cfg.patterns:
            print(
                f"{p.id}: {p.name} | enabled={p.enabled} | capital=¥{p.initial_capital:,.0f} | "
                f"interval={p.interval} | unit={p.unit_size} | buy>={p.buy_score_threshold} sell<={p.sell_score_threshold}"
            )
        return 0

    if args.command == "export-snapshot":
        storage.export_dashboard_snapshot(cfg)
        print("Snapshot exported.")
        return 0

    if args.command == "encrypt-state":
        passphrase = os.environ.get("DASHBOARD_PASSPHRASE", "").strip()
        if not passphrase:
            print("Error: DASHBOARD_PASSPHRASE environment variable is not set.", file=sys.stderr)
            return 1
        if not DEFAULT_SNAPSHOT_PATH.exists():
            storage.export_dashboard_snapshot(cfg)
        if DEFAULT_SNAPSHOT_PATH.exists():
            encrypt_file(DEFAULT_SNAPSHOT_PATH, ENC_WEB_SNAPSHOT_PATH, passphrase)
        if DEFAULT_DB_PATH.exists():
            encrypt_file(DEFAULT_DB_PATH, ENC_DB_PATH, passphrase)
        if args.remove_plaintext:
            for p in (DEFAULT_DB_PATH, DEFAULT_SNAPSHOT_PATH, WEB_SNAPSHOT_PATH):
                if p.exists():
                    p.unlink()
        print("Encrypted DB and dashboard snapshot successfully.")
        return 0

    if args.command == "decrypt-state":
        passphrase = os.environ.get("DASHBOARD_PASSPHRASE", "").strip()
        if not passphrase:
            print("Error: DASHBOARD_PASSPHRASE environment variable is not set.", file=sys.stderr)
            return 1
        if ENC_DB_PATH.exists():
            decrypt_file(ENC_DB_PATH, DEFAULT_DB_PATH, passphrase)
            print("Decrypted DB restored.")
        else:
            print("No encrypted DB found; starting fresh or using existing DB.")
        return 0

    if args.command == "reset":
        target = args.mode or args.pattern
        if target == "all":
            for p in cfg.patterns:
                storage.reset_portfolio(p.id, args.capital or p.initial_capital, unit_size=p.unit_size)
            print(f"Reset completed for {len(cfg.patterns)} patterns.")
        else:
            p = cfg.get_pattern(target)
            if p is None:
                print(f"Pattern not found: {target}", file=sys.stderr)
                return 1
            storage.reset_portfolio(p.id, args.capital or p.initial_capital, unit_size=p.unit_size)
            print(f"Reset completed for {p.id}.")
        storage.export_dashboard_snapshot(cfg)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
