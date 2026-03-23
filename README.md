
## 準備
1. ホストに入れる(これは厳密には真偽不明)
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
    python3 simpler_test.py
    ```
    headlessの場合は以下コマンドで実行できる．
    ```
    xvfb-run python3 simpler_test.py
    ```
    - simpler_test.pyのオプション
        - `--task-name` シミュレーション環境名 default : `google_robot_pick_coke_can`
            - シミュレーション環境
                - `google_robot_pick_coke_can`
                - `google_robot_pick_object`
                - `google_robot_move_near`
                - `google_robot_open_drawer`
                - `google_robot_close_drawer`
                - `google_robot_place_in_closed_drawer`
                - `widowx_spoon_on_towel`
                - `widowx_carrot_on_plate`
                - `widowx_stack_cube`
                - `widowx_put_eggplant_in_basket`
        - `--video-name` 保存する動画ファイル名 default : result.mp4
        - `--instruction` ロボットに対する指示 default : 環境毎の指定
        - `--model-name` VLAモデル名  default : rt_1_x
            - 使用可能VLAモデル
                - `rt_1_x`
                - `octo-base`
                - `octo-small`