# Wav2Midi GUI

## 概要
これは、オーディオファイル（WAV, MP3など）を自動的にMIDIファイルに変換するGUIアプリケーションです。
Meta社の「Demucs」を使用して音源を各パート（ボーカル、ドラム、ベース、その他）に分離し、Spotifyの「Basic Pitch」および「ADTOF」を使用してそれぞれのパートをMIDIデータに変換します。
また、カスタムモデル「BandIt」を使用した高度な分離処理もサポートしており、特定の音源（スピーチや効果音など）の分離と統合が可能です。

## 機能
*   **音源分離**: 
    *   標準で4パート（Vocals, Drums, Bass, Other）への分離。
    *   オプションで6パート（+ Guitar, Piano）への分離に対応。
*   **MIDI変換**:
    *   分離された各パートを自動的にMIDIファイル（.mid）に変換。
    *   ドラムには専用の「ADTOF」、その他の楽器には「Basic Pitch」を使用。
*   **BandIt連携**:
    *   `GuiApp/bandit`フォルダ内のカスタムモデルを使用した分離処理。
    *   BandItで分離したステムをDemucsの入力として使用したり、Demucsの出力結果にマージしたりすることが可能。

## 環境構築 (Setup)

本ツールを使用するには、以下の手順で環境を構築する必要があります。

### 1. Python環境の作成
Python 3.11を使用して仮想環境（venv）を作成することを推奨します。

```bash
# プロジェクトルートディレクトリで実行
python3.11 -m venv venv
source venv/bin/activate
```

### 2. 依存ライブラリのインストール
作成した `requirements.txt` を使用して必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```
※ `requirements.txt` が存在しない場合は、`pip freeze > requirements.txt` を実行して作成してください（本リポジトリには生成済みファイルが含まれる場合があります）。

### 3. 必要なリポジトリのクローン
以下の2つのGitHubリポジトリをプロジェクトのルートディレクトリにクローンしてください。

*   **Music-Source-Separation-Training** (BandItモデル用)
*   **ADTOF-pytorch** (ドラムMIDI変換用)

```bash
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git
git clone https://github.com/xavriley/ADTOF-pytorch.git
```

### 4. BandItモデルのセットアップ
BandItモデル（Cinematic Sound Separationなど）を使用するために、モデルファイルと設定ファイルを配置します。

1.  `GuiApp/bandit` フォルダ内に、モデルごとのサブフォルダ（例: `BanditPlus`）を作成します。
    ```bash
    mkdir -p GuiApp/bandit/BanditPlus
    ```
2.  以下のリンク（または [Pretrained Models](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md)）から **Configファイル** と **Weights（チェックポイント）** をダウンロードします。
    *   **Config例 (BandIt Plus)**: [config_dnr_bandit_bsrnn_multi_mus64.yaml](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml)
    *   **Weights例**: [model_bandit_plus_dnr_sdr_11.47.chpt](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt)
3.  ダウンロードしたファイルを `GuiApp/bandit/BanditPlus/` に移動します。
4.  **重要**: ダウンロードしたConfigファイルの名前を `hparams.yaml` に変更してください。

配置後の構造例:
```
GuiApp/
  bandit/
    BanditPlus/
      hparams.yaml  <-- リネームしたConfig
      model_bandit_plus_dnr_sdr_11.47.chpt
```

## 使用方法

### 1. アプリケーションの起動
仮想環境が有効な状態で、以下のコマンドを実行します。

```bash
python GuiApp/wav2midi_gui.py
```

### 2. ファイルの選択とオプション設定
*   **Selected Audio File**: 変換するオーディオファイルを選択。
*   **Options**:
    *   `Enable 6-stem separation`: Guitar/Pianoを含める場合。
    *   `Force Separate`: 再分離を行う場合。
    *   `Force MIDI Conversion`: 再変換を行う場合。
*   **BandIt Options**:
    *   `Use BandIt`: オンにするとモデル選択が可能になります。
    *   `Stem Mapping`: 各ステムの扱い（Demucs入力、マージ先）を指定します。

### 3. 変換の実行
「Start Conversion」をクリックすると処理が開始され、ログが表示されます。

## 出力結果
`outputs/[曲名]/` に保存されます。
*   `midi/`: 生成されたMIDIファイル
*   `[model_name]/`: 分離された音声ファイル
