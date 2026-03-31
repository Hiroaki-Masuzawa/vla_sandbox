import site
site.main()

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import numpy as np
import argparse
import os
import torch

from absl import logging

# WARNING以上のログを出さない
logging.set_verbosity(logging.ERROR)

# -----------------------------
# 設定
# -----------------------------
TASK_NAMES = [
    "google_robot_pick_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_place_in_closed_drawer",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]

N_TRIALS = 20

# -----------------------------
# コマンドライン引数
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", type=str, default="rt_1_x",
                    choices=['rt_1_x', "octo-base", "octo-small", "openvla-7b"])
args = parser.parse_args()
MODEL_NAME = args.model_name


# -----------------------------
# モデル初期化
# -----------------------------
def init_model(model_name: str, policy_setup: str):
    if "rt_1" in model_name:
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        ckpt_path = f"/SimplerEnv/{model_name}_checkpoints"
        return RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
    elif "octo" in model_name:
        from simpler_env.policies.octo.octo_model import OctoInference
        return OctoInference(model_type=model_name, policy_setup=policy_setup, init_rng=0)
    elif "openvla" in model_name:
        from simpler_env.policies.openvla.openvla_model import OpenVLAInference
        return OpenVLAInference(saved_model_path="openvla/openvla-7b", policy_setup=policy_setup)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# -----------------------------
# 単一エピソード実行
# -----------------------------
def run_episode(env, model, instruction):
    obs, _ = env.reset()
    model.reset(instruction)
    image = get_image_from_maniskill2_obs_dict(env, obs)

    predicted_terminated, truncated, success = False, False, False

    while not (predicted_terminated or truncated):
        if "openvla" in MODEL_NAME:
            raw_action, action = model.step(image, instruction)
        else:
            raw_action, action = model.step(image)

        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        obs, reward, success, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        )
        image = get_image_from_maniskill2_obs_dict(env, obs)

    return success


# -----------------------------
# タスク実行
# -----------------------------
def run_task(task_name: str):
    policy_setup = "google_robot" if "google" in task_name else "widowx_bridge"
    model = init_model(MODEL_NAME, policy_setup)

    success_count = 0
    for _ in range(N_TRIALS):
        env = simpler_env.make(task_name)
        instruction = env.get_language_instruction()
        success = run_episode(env, model, instruction)
        env.close()
        del env
        success_count += int(success)

    print(f"{task_name}: {success_count}/{N_TRIALS} successful")
    del model
    torch.cuda.empty_cache()
    return success_count


# -----------------------------
# メイン処理
# -----------------------------
if __name__ == "__main__":
    for task in TASK_NAMES:
        run_task(task)