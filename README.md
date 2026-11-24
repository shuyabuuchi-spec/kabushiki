# Kabushiki — 日本株監視・売却タイミング推定＆複数パターン並行シミュレータ

日本の株式市場（東証）の株価データ（Yahoo Finance・無料／APIキー不要）を監視し、**初期資金・監視間隔（5〜60分）・取引単位（1株/100株）・売買タイミングが異なる複数パターンを並行稼働**できるアプリケーション＆実データ検証ツールです。

- **複数パターン並行シミュレーション（初期資金・タイミング自由設定）**:
  - デフォルトで4パターンを並行稼働（UIまたは設定JSONから何パターンでも追加・複製・編集・ON/OFF・削除可能）：
    1. `標準バランス 10万円（1株・15分足）`
    2. `短期回転 10万円（1株・10分足・積極・引け決済）`
    3. `厳選スイング 10万円（1株・30分足・慎重）`
    4. `単元株 10万円（100株・15分足）`
  - 初期資金（5万円・10万円・30万円・50万円・100万円など自由入力）、売買チェック間隔（`5m`/`10m`/`15m`/`30m`/`60m`）、買いスコア閾値、売りスコア閾値、利確%、損切り%、トレーリングストップ%、大引け前決済をパターンごとに個別設定可能。
- **おすすめ投資先ランキング**: EMA・SMA60・VWAP・MACD・RSI・ボリンジャーバンド・出来高急増倍率から **0〜100点** でスコアリング。
- **保有銘柄の売却タイミング推定**: 損切り・利確目標・トレーリングストップ・テクニカル反転・大引け前決済を常時診断。
- **実データ並行比較バックテスト**: 全有効パターンを同じ実データ期間で一括比較し、ランキング表＋マルチライン利回りチャートで勝率・PF・最大DD・日経平均（`^N225`）比較を即時可視化。
- **PC電源OFF時の継続監視（Publicリポジトリ推奨）**:
  - **Public リポジトリ**なら GitHub Actions は**完全無料・無制限**（他の Private リポジトリの月2,000分枠を一切消費しません）。
  - 平日 9:00〜15:30 JST に10分間隔で全有効パターンを並行実行し、欠損バーも自動補完（PC-OFF Catch-Up）。

---

## クイックスタート（ローカルPC）

```powershell
# 1. 仮想環境＆パッケージ導入
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Webダッシュボード起動（http://127.0.0.1:8000）
.\.venv\Scripts\python.exe -m kabushiki.server
```

ブラウザで `http://127.0.0.1:8000` を開き、画面右上の **「＋ パターン追加」** から初期資金や売買タイミングを変えたパターンを自由に増やせます。

### CLIでの実行例
```powershell
# 全有効パターンを実データで並行比較測定（直近14日間）
.\.venv\Scripts\python.exe -m kabushiki.runner backtest --compare-all --days 14

# 全有効パターンの初期資金を30万円に上書きして並行比較
.\.venv\Scripts\python.exe -m kabushiki.runner backtest --compare-all --days 14 --capital 300000

# 全有効パターンをPC OFF補完しつつ最新状態まで更新
.\.venv\Scripts\python.exe -m kabushiki.runner catchup --pattern all

# 全パターンの現在成績を一覧表示
.\.venv\Scripts\python.exe -m kabushiki.runner status
```

---

## PC電源OFFでも監視を続ける設定（GitHub Actions + GitHub Pages）

1. **Public リポジトリ**として GitHub へプッシュします（APIキー・個人情報不要のため安全で、他の Private リポジトリの Actions 枠を消費しません）。
2. `Settings → Actions → General → Workflow permissions` で **Read and write permissions** を有効にします。
3. （任意）`Settings → Pages → Source` を **GitHub Actions** に設定すると、スマホ等から全パターンの最新比較ダッシュボードを常時閲覧できます。
4. パターンを追加・変更した `data/user_config.json` をリポジトリにコミットしておけば、PC OFF 中の GitHub Actions でもその設定のまま並行稼働します。

---

## 免責事項
本ツールは教育・技術検証を目的としたシミュレータであり、特定の有価証券の売買を勧誘・保証するものではありません。実際の投資判断はご自身の責任で行ってください。
