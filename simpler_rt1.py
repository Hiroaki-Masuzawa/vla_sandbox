import site

site.main()

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import sapien.core as sapien
import os
import numpy as np
import argparse


def write_video(frames, output_file="test.mp4", fps=10):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v'はMP4フォーマット用のFourCCコード
    height, width, _ = frames[0].shape  # フレームのサイズ
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame[:, :, ::-1])

    video_writer.release()


task_names = [
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name",
        type=str,
        default="google_robot_pick_coke_can",
        choices=task_names,
    )
    parser.add_argument("--video-name", type=str, default="result.mp4")
    parser.add_argument("--instruction", type=str, default="")
    args = parser.parse_args()

    task_name = args.task_name

    if "env" in locals():
        print("Closing existing env")
        env.close()
        del env
    env = simpler_env.make(task_name)

    # Note: we turned off the denoiser as the colab kernel will crash if it's turned on
    # To use the denoiser, please git clone our SIMPLER environments
    # and perform evaluations locally.
    sapien.render_config.rt_use_denoiser = False

    obs, reset_info = env.reset()
    instruction = (
        env.get_language_instruction() if args.instruction == "" else args.instruction
    )
    print("Reset info", reset_info)
    print("Instruction", instruction)

    if "google" in task_name:
        policy_setup = "google_robot"
    else:
        policy_setup = "widowx_bridge"

    model_name = "rt_1_x"  # @param ["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small"]
    if "rt_1" in model_name:
        from simpler_env.policies.rt1.rt1_model import RT1Inference

        ckpt_path = "/SimplerEnv/checkpoints"
        model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
    elif "octo" in model_name:
        from simpler_env.policies.octo.octo_model import OctoInference

        model = OctoInference(
            model_type=model_name, policy_setup=policy_setup, init_rng=0
        )
    else:
        raise ValueError(model_name)

    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    model.reset(instruction)
    print(instruction)

    image = get_image_from_maniskill2_obs_dict(
        env, obs
    )  # np.ndarray of shape (H, W, 3), uint8
    images = [image]
    predicted_terminated, success, truncated = False, False, False
    timestep = 0
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        raw_action, action = model.step(image)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        obs, reward, success, truncated, info = env.step(
            np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            )
        )
        print(timestep, info)
        # update image observation
        image = get_image_from_maniskill2_obs_dict(env, obs)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})
    print(f"Episode success: {success}")
    write_video(images, output_file=args.video_name)


if __name__ == "__main__":
    main()
