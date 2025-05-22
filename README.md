# ANAC環境構築
## Pythonインストール
1. リモートデスクトップで研究室PCに接続(詳細はTeamsチャネル参照)
2. デフォルトでpythonがインストールされていないので, Edgeで最新版の[Pythonインストーラ](https://www.python.org/downloads/release/python-3133/)をダウンロード
3. ダウンロードしたインストーラを実行
4. 管理者権限がないため`Use admin privileges when installing py.exe`のチェックを外す
5. `Add python.exe to PATH`のチェックを入れる
6. `Install Now`を選択しインストール終了まで待つ
7. powershellを開き`python -V`を入力してバージョンが表示されればインストール完了
### pythonのバージョンが表示されない
- powershellの再起動
- pythonのパスが通っているか確認
## pyenvのインストール
pyenvはpythonのバージョンを管理するためのツールである.
ライブラリによってはpythonのバージョンが依存関係となるので, 簡単にpythonのバージョンを切り替えたい.
windowsではpyenv-winを使う.
1. powershellを開く
2. `python -m pip install pyenv-win --target .pyenv`を実行
3. エクスプローラーでuserディレクトリ配下に`.pyenv`というフォルダが作成されていることを確認(無ければ表示から隠しファイルの表示にチェックを入れる)
4. 以下のコマンドを実行
   ```powershell
   [System.Environment]::SetEnvironmentVariable('path', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('path', "User"),"User")
   ```
5. powershellを再起動し`pyenv`と実行
6. Authエラーが出た場合, 以下のコマンドをpowershellにて実行
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
7. 5の手順をもう一度試し, Usageが表示されることを確認
8. 初回設定として以下のコマンドを実行
   ```powershell
   pyenv update
   pyenv install 3.11.9
   pyenv global 3.11.9
   ```
9.  `python -V`によってバージョンが3.11.9になっていることを確認
## venvの設定
venvとはpythonにデフォルトで搭載されている仮想環境構築ツールである.
venvを用いることで, インストールしたライブラリのコンフリクトを軽減できる. 実行したいスクリプトによって求められるライブラリのバージョンが異なるため, globalな環境(システム)に直接ライブラリ等をインストールする事は好ましくない.
また, フォルダを消すだけで簡単に仮想環境を消せるため, 他の管理ツール(conda等)よりシンプルでわかりやすい. venvの構築は以下のように行う.
1. powershellを開く
2. 仮想環境を作成したいディレクトリに移動(基本はプロジェクト配下に作成する)
3. `python -m venv .venv`で, `.venv`という名前の仮想環境を格納したディレクトリを生成.
   この時生成される仮想環境のバージョンは, 現在のシェルで使用しているpythonのバージョンと一致する. 他のpythonのバージョンが必要な時は, 適宜pyenvでインストールしlocalバージョンを設定する
4. `. .venv/Scripts/activate`で仮想環境に入る
5. シェルの左側に`(.venv)`と表示されていれば仮想環境内である. この仮想環境内で, pipなどを使用する. シェルを再起動すると仮想環境が無効になるので再度有効にする事を忘れずに. 永続させる方法もあるが割愛
6. `pip install -r requirements.txt`で必要なライブラリを一括でインストールできる. インストールされたライブラリは仮想環境内でのみ有効となる.