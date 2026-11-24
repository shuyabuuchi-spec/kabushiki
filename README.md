# Kabushiki — 日本株監視・売却タイミング推定＆複数パターン並行シミュレータ

日本の株式市場（東証）の株価データ（Yahoo Finance・無料／APIキー不要）を監視し、**初期資金・監視間隔（5〜60分）・取引単位（1株/100株）・売買タイミングが異なる複数パターンを並行稼働**できるアプリケーション＆実データ検証ツールです。

- **複数パターン並行シミュレーション（Walk-Forward 機械学習＋押し目買い＋順張り）**:
  - デフォルトで5パターンを並行稼働（UIから初期資金・間隔・戦略タイプを自由に追加・編集可能）：
    1. `🤖 機械学習 Walk-Forward 10万円（1株・30分足）` — 過去蓄積データ＋日経平均特徴量でオンライン再学習し、期待リターン `+0.8%` 以上のみ厳選購入
    2. `🤖 機械学習 Walk-Forward 10万円（1株・15分足）`
    3. `📉 押し目買い平均回帰 10万円（1株・30分足）`
    4. `📈 順張りバランス 10万円（1株・15分足・比較用）`
    5. `🏢 単元株 ML 10万円（100株・30分足・ゼロ革命想定）`
- **未来情報ゼロの事前予測＆実際の手取り収益**:
  - 確定した足 `t` で事前予測（購入10〜30分前提案）を確定し、次の足 `t+1` の始値＋スリッページ＋手数料で実約定。
  - 取引手数料（デフォルト `0.10%`・自由変更）と譲渡益課税（デフォルト `20.315%`・損益通算対応・自由変更）を控除した**実際の手取り純資産**を表示。
- **機械学習モデルと複数ソースデータ自動蓄積（精度の継続向上）**:
  - **学習モデル**: **Walk-Forward アンサンブル（L2正則化 Ridge回帰 55% ＋ scikit-learn 勾配ブースティング決定木 `HistGradientBoostingRegressor` 45%）**
  - **データソース（Yahoo以外も収集）**:
    - **FRED（セントルイス連邦準備銀行・公式オープンデータ）**: 過去5年分のドル円（`DEXJPUS`）・米国VIX恐怖指数（`VIXCLS`）・米10年債利回り（`DGS10`）・日経225日足（`NIKKEI225`） ※ Frankfurter ECB為替APIフォールバック対応
    - **Yahoo Finance**: `15m` / `30m` 日中足＋**`60m` 長期過去データ（約1〜2年分）**＋日経平均 `^N225`
  - 収集データは暗号化 SQLite（`ml_bars` / `ml_macro` / `ml_training_history`）へ自動追記して長期蓄積し、ダッシュボード上部に **Walk-Forward 学習経過グラフ・データソース別件数・学習履歴ログ** を常時表示します。
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
4. **ログイン画面（AES-256-GCM暗号化保護）を有効にする場合**:
   - `Settings → Secrets and variables → Actions → Repository secrets → New repository secret`
   - **Name**: `DASHBOARD_PASSPHRASE` / **Secret**: 任意のパスワード → **Add secret**
   - `Actions` タブから `Monitor JP Stocks & Multi-Pattern Simulator (PC OFF)` を1回実行すると、DBとJSONが自動暗号化され、GitHub Pages にログイン画面が表示されます（コード・リポジトリ内にパスワード平文は一切保存されません）。
5. **GitHub Pages 上から直接パターン追加・編集・削除する場合**:
   - GitHub の [Fine-grained personal access tokens](https://github.com/settings/personal-access-tokens/new) で対象リポジトリを `kabushiki` のみに限定し、**Contents: Read and write** と **Actions: Read and write** を付与してトークンを発行します。
   - ダッシュボード右上の **「☁ GitHub編集連携」** からトークンを登録すると（トークンはブラウザ内のみ保持・Gitには一切保存されません）、スマホや外出先からでも直接パターンの追加・編集・削除・即時シミュレーション実行が可能になります。
6. パターンを追加・変更した `data/user_config.json` をリポジトリにコミットしておけば、PC OFF 中の GitHub Actions でもその設定のまま並行稼働します。

---

## 免責事項
本ツールは教育・技術検証を目的としたシミュレータであり、特定の有価証券の売買を勧誘・保証するものではありません。実際の投資判断はご自身の責任で行ってください。
