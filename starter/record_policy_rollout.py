#!/usr/bin/env python3
"""Record ACT policy rollout videos for project demos.

This is separate from starter/eval_local.py because the official local evaluator
intentionally disables video to keep scoring deterministic and lightweight.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = KIT_ROOT / "external" / "robotwin_local"
PUBLIC_SEEDS = KIT_ROOT / "starter" / "public_seeds.json"

sys.path.insert(0, str(KIT_ROOT))
from recipes.eval.run_act_eval import (  # noqa: E402
    TASK_BY_TRACK,
    _setup_runtime,
    build_configs,
    load_seed_table,
    shadow_player_act,
)


def read_seeds(path: Path) -> list[int]:
    table = load_seed_table(str(path))
    return [int(row["episode_seed"]) for row in table]


def open_video_writer(path: Path, video_size: str):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track", required=True, choices=sorted(TASK_BY_TRACK))
    ap.add_argument("--ckpt-dir", required=True,
                    help="ACT ckpt directory containing policy_last.ckpt + dataset_stats.pkl")
    ap.add_argument("--out-dir", default="policy_rollout_videos")
    ap.add_argument("--seed", type=int, default=None,
                    help="Record a specific seed. If omitted, search public seeds.")
    ap.add_argument("--seeds", default=str(PUBLIC_SEEDS),
                    help="Seed table used when --seed is omitted.")
    ap.add_argument("--max-seeds", type=int, default=20,
                    help="Maximum public seeds to try when searching for a success.")
    ap.add_argument("--stop-on-success", action="store_true",
                    help="Stop after recording the first successful rollout.")
    ap.add_argument("--task-config", default=None,
                    help="Default: <task>_clean for the selected track.")
    ap.add_argument("--deploy-config", default="policy/ACT/deploy_policy.yml")
    ap.add_argument("--state-dim", type=int, default=16)
    ap.add_argument("--temporal-agg", action="store_true")
    ap.add_argument("--player-code-root", default=None)
    ap.add_argument("--summary", default=None)
    args_cli = ap.parse_args()

    ckpt_dir = Path(args_cli.ckpt_dir).resolve()
    if not ckpt_dir.exists():
        raise SystemExit(f"--ckpt-dir does not exist: {ckpt_dir}")

    out_dir = Path(args_cli.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args_cli.summary).resolve() if args_cli.summary else out_dir / "summary.json"

    if args_cli.seed is not None:
        seeds = [args_cli.seed]
    else:
        seeds = read_seeds(Path(args_cli.seeds).resolve())[: args_cli.max_seeds]

    root, ep = _setup_runtime(str(ROBOTWIN_ROOT))
    task_name = TASK_BY_TRACK[args_cli.track]
    task_config = args_cli.task_config or f"{task_name}_clean"
    usr, task_args = build_configs(
        ep,
        task_name,
        task_config,
        "video_demo",
        str(ckpt_dir),
        args_cli.state_dim,
        args_cli.temporal_agg,
        args_cli.deploy_config,
    )
    task_args["eval_video_log"] = True
    task_args["eval_video_save_dir"] = str(out_dir)
    task_args["render_freq"] = 0

    shadow_player_act(args_cli.player_code_root)

    get_model = ep.eval_function_decorator(usr["policy_name"], "get_model")
    eval_func = ep.eval_function_decorator(usr["policy_name"], "eval")
    reset_func = ep.eval_function_decorator(usr["policy_name"], "reset_model")
    model = get_model(usr)
    task_env = ep.class_decorator(task_args["task_name"])

    video_size = f"{task_args['head_camera_w']}x{task_args['head_camera_h']}"
    records = []

    for idx, seed in enumerate(seeds):
        video_path = out_dir / f"policy_rollout_seed_{seed}.mp4"
        ffmpeg = None
        success = False
        steps = 0
        error = None
        try:
            task_env.setup_demo(now_ep_num=idx, seed=seed, is_test=True, **task_args)
            task_env.test_num = idx
            reset_func(model)
            ffmpeg = open_video_writer(video_path, video_size)
            task_env._set_eval_video_ffmpeg(ffmpeg)

            while task_env.take_action_cnt < task_env.step_lim:
                observation = task_env.get_obs()
                eval_func(task_env, model, observation)
                if task_env.eval_success:
                    success = True
                    break
            steps = int(task_env.take_action_cnt)
        except Exception as exc:  # keep the search moving and record the failure cause.
            error = repr(exc)
            print(f"[record-policy-rollout] seed={seed} error={error}", flush=True)
        finally:
            try:
                if ffmpeg is not None:
                    task_env._del_eval_video_ffmpeg()
            finally:
                task_env.close_env(clear_cache=True)

        record = {
            "seed": seed,
            "success": bool(success),
            "steps": steps,
            "video": str(video_path),
        }
        if error:
            record["error"] = error
        records.append(record)
        print(
            f"[record-policy-rollout] seed={seed} success={success} "
            f"steps={steps} video={video_path}",
            flush=True,
        )

        if success and args_cli.stop_on_success:
            break

    summary = {
        "track": args_cli.track,
        "task": task_name,
        "ckpt_dir": str(ckpt_dir),
        "records": records,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[record-policy-rollout] wrote {summary_path}", flush=True)
    return 0 if any(r["success"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
