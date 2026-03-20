
## 準備
1. ホストに入れる
    ```
    sudo apt install libvulkan1 vulkan-tools
    ```
1. イメージをビルドする
    ```
    ./build.sh
    ```
## 実行
1. dockerを起動する
    ```
    ./run.sh
    ```
1. コードを実行する
    ```
    python3 simpler_rt1.py
    ```
    headlessの場合は以下コマンドで実行できる．
    ```
    xvfb-run python3 simpler_rt1.py 
    ```