"""Re-render capx trial trajectories with Isaac Sim (ray/path tracing).

This script reads ``state_trajectory.npz`` and ``model.xml`` files saved
by the capx trial pipeline and produces high-quality renders using Isaac
Sim's Replicator.

Prerequisites
-------------
- NVIDIA Isaac Sim / Isaac Lab installed
- ``usd-core`` pip package
- ``state_trajectory.npz`` and ``model.xml`` in the trial output directory
  (produced when ``record_video: true`` in the trial config)

Usage
-----
Run from the Isaac Sim Python environment::

    # Render a single trial directory
    python capx/envs/scripts/render_isaacsim.py \
        --trial_dir outputs/.../trial_00_sandboxrc_0_reward_1.000_taskcompleted_1 \
        --cameras agentview robot0_robotview \
        --width 1920 --height 1080 \
        --renderer PathTracing

    # Render all trial directories under an experiment
    python capx/envs/scripts/render_isaacsim.py \
        --experiment_dir outputs/aws_anthropic_claude-opus-4-5/my_experiment \
        --cameras agentview \
        --save_video
"""

from __future__ import annotations

import argparse
import os
import re
import sys

parser = argparse.ArgumentParser(
    description="Re-render capx trial trajectories with Isaac Sim"
)
parser.add_argument(
    "--trial_dir",
    type=str,
    default=None,
    help="Path to a single trial output directory containing state_trajectory.npz and model.xml",
)
parser.add_argument(
    "--experiment_dir",
    type=str,
    default=None,
    help="Path to an experiment directory; all trial_* subdirectories will be rendered",
)
parser.add_argument(
    "--cameras",
    type=str,
    nargs="+",
    default=["agentview"],
    help="Camera name(s) to render (default: agentview)",
)
parser.add_argument(
    "--width",
    type=int,
    default=1280,
    help="Output image width (default: 1280)",
)
parser.add_argument(
    "--height",
    type=int,
    default=720,
    help="Output image height (default: 720)",
)
parser.add_argument(
    "--renderer",
    type=str,
    default="RayTracedLighting",
    choices=["RayTracedLighting", "PathTracing"],
    help="Isaac Sim renderer (default: RayTracedLighting)",
)
parser.add_argument(
    "--spp",
    type=int,
    default=64,
    help="Samples per pixel for PathTracing (default: 64)",
)
parser.add_argument(
    "--framerate",
    type=int,
    default=30,
    help="USD framerate (default: 30)",
)
parser.add_argument(
    "--skip_frames",
    type=int,
    default=1,
    help="Render every nth frame (default: 1, render all)",
)
parser.add_argument(
    "--save_video",
    action="store_true",
    default=False,
    help="Assemble rendered frames into MP4 video files",
)
parser.add_argument(
    "--output_suffix",
    type=str,
    default="isaacsim",
    help="Suffix appended to output video filenames (default: isaacsim)",
)
parser.add_argument(
    "--hide_sites",
    action="store_true",
    default=False,
    help="Hide all sites in the scene",
)
parser.add_argument(
    "--dome_light_intensity",
    type=float,
    default=1500.0,
    help="Dome light intensity (default: 1500)",
)

# Isaac Sim / AppLauncher args will be added by AppLauncher.add_app_launcher_args
try:
    from omni.isaac.lab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
except ImportError:
    print(
        "ERROR: Isaac Sim / Isaac Lab not found.\n"
        "This script must be run from an Isaac Sim Python environment.\n"
        "See: https://docs.omniverse.nvidia.com/isaacsim/latest/index.html"
    )
    sys.exit(1)

import shutil

import carb.settings
import cv2
import lxml.etree as ET
import mujoco
import numpy as np
import omni
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
from tqdm import tqdm

import robosuite.utils.usd.exporter as usd_exporter


scene_option = mujoco.MjvOption()
scene_option.geomgroup = [0, 1, 0, 0, 0, 0]


def _make_sites_invisible(xml_string: str) -> str:
    root = ET.fromstring(xml_string)
    for site in root.findall(".//site"):
        site.set("rgba", "0 0 0 0")
    return ET.tostring(root, encoding="unicode")


def _natural_sort_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


class TrialUSDExporter:
    """Load a capx trial's state trajectory and export it to USD."""

    def __init__(self, trial_dir: str) -> None:
        self.trial_dir = trial_dir
        self.output_dir = os.path.join(trial_dir, f"isaacsim_render")

        traj_path = os.path.join(trial_dir, "state_trajectory.npz")
        xml_path = os.path.join(trial_dir, "model.xml")

        if not os.path.isfile(traj_path):
            raise FileNotFoundError(f"No state_trajectory.npz found in {trial_dir}")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"No model.xml found in {trial_dir}")

        data = np.load(traj_path)
        self.states: np.ndarray = data["states"][:: args.skip_frames]
        self.traj_len = len(self.states)

        with open(xml_path) as f:
            model_xml = f.read()
        if args.hide_sites:
            model_xml = _make_sites_invisible(model_xml)

        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)

        self._set_initial_state(self.states[0])

        stage = None
        if args.renderer == "PathTracing" or not hasattr(args, "online"):
            pass
        stage = stage_utils.get_current_stage() if getattr(args, "online", False) else None

        self.exporter = usd_exporter.USDExporter(
            model=self.model,
            output_directory_name=self.output_dir,
            camera_names=args.cameras,
            online=getattr(args, "online", False),
            shareable=not getattr(args, "online", False),
            framerate=args.framerate,
            stage=stage,
        )
        self.exporter.update_scene(data=self.data, scene_option=scene_option)
        self.exporter.add_light(
            pos=[0, 0, 0],
            intensity=args.dome_light_intensity,
            light_type="dome",
            light_name="dome_1",
        )

    def _set_initial_state(self, flat_state: np.ndarray) -> None:
        time_val = flat_state[0]
        nq = self.model.nq
        nv = self.model.nv
        qpos = flat_state[1 : 1 + nq]
        qvel = flat_state[1 + nq : 1 + nq + nv]

        self.data.time = time_val
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def update_simulation(self, index: int) -> None:
        self._set_initial_state(self.states[index])
        self.exporter.update_scene(data=self.data, scene_option=scene_option)

    def export_all(self) -> None:
        print(f"Exporting {self.traj_len} frames to USD...")
        for i in tqdm(range(self.traj_len)):
            self.update_simulation(i)
        self.exporter.save_scene(filetype="usd")
        print(f"USD saved to {self.output_dir}")

    def close(self) -> None:
        self.exporter.save_scene(filetype="usd")


class IsaacSimWriter(rep.Writer):
    """Replicator writer that saves rendered frames to disk."""

    def __init__(
        self,
        output_dir: str = None,
        image_output_format: str = "png",
        frame_padding: int = 4,
    ):
        self._output_dir = output_dir
        if output_dir:
            self._backend = rep.BackendDispatch(output_dir=output_dir)
        self._frame_id = 0
        self._frame_padding = frame_padding
        self._image_output_format = image_output_format
        self._output_data_format = {}
        self.annotators = [rep.AnnotatorRegistry.get_annotator("rgb")]
        self.version = "0.1.0"
        self.data_structure = "annotator"
        self.write_ready = False

    def write(self, data: dict):
        if not self.write_ready:
            return
        for annotator_name, annotator_data in data["annotators"].items():
            for idx, (_, anno_rp_data) in enumerate(annotator_data.items()):
                if annotator_name == "rgb":
                    cam_name = args.cameras[idx] if idx < len(args.cameras) else f"cam_{idx}"
                    filepath = os.path.join(
                        cam_name, "rgb", f"rgb_{self._frame_id}.{self._image_output_format}"
                    )
                    self._backend.write_image(filepath, anno_rp_data["data"])
        self._frame_id += 1


rep.WriterRegistry.register(IsaacSimWriter)


class IsaacSimRenderer:
    """Render a USD trajectory with Isaac Sim's Replicator."""

    def __init__(self, trial_exporter: TrialUSDExporter) -> None:
        self.exporter = trial_exporter
        self.output_dir = trial_exporter.output_dir
        self.num_frames = trial_exporter.traj_len
        self.current_frame = 0
        self.writer = None
        self.render_products = []
        self.initial_skip = 5

    def _load_usd_stage(self) -> None:
        usd_path = os.path.join(
            self.output_dir, f"frames/frame_{self.num_frames + 1}.usd"
        )
        print(f"Opening USD stage: {usd_path}")
        stage_utils.open_stage(usd_path)
        print("Stage loaded")

    def _init_renderer(self) -> bool:
        self._load_usd_stage()

        settings = carb.settings.get_settings()
        if settings.get("/omni/replicator/captureOnPlay"):
            settings.set_bool("/omni/replicator/captureOnPlay", False)
        settings.set_bool("/app/renderer/waitIdle", False)
        settings.set_bool("/app/hydraEngine/waitIdle", False)
        settings.set_bool("/app/asyncRendering", True)
        settings.set(f"/rtx/pathtracing/spp", args.spp)
        settings.set_bool("/exts/omni.replicator.core/Orchestrator/enabled", True)

        self.writer = rep.WriterRegistry.get("IsaacSimWriter")
        self.writer.initialize(output_dir=self.output_dir)

        for camera_name in args.cameras:
            camera_path = f"/World/Camera_Xform_{camera_name}/Camera_{camera_name}"
            context = omni.usd.get_context()
            stage = context.get_stage()
            prim = stage.GetPrimAtPath(camera_path)
            if prim.IsValid() and prim.GetTypeName() == "Camera":
                rp = rep.create.render_product(
                    camera_path, (args.width, args.height), force_new=True
                )
                self.render_products.append(rp)
            else:
                print(f"Warning: camera prim not found at {camera_path}")

        if not self.render_products:
            print("No valid render products found.")
            return False

        self.writer.attach(self.render_products)
        return True

    def render(self) -> None:
        if not self._init_renderer():
            return

        for _ in range(self.initial_skip):
            rep.orchestrator.step(rt_subframes=1, delta_time=None, pause_timeline=False)

        self.writer.write_ready = True
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_end_time(self.num_frames)

        print(f"Rendering {self.num_frames} frames...")
        with tqdm(total=self.num_frames) as pbar:
            while self.current_frame < self.num_frames:
                timeline.forward_one_frame()
                rep.orchestrator.step(
                    rt_subframes=1, delta_time=None, pause_timeline=True
                )
                self.current_frame += 1
                pbar.update(1)

        timeline.stop()
        rep.orchestrator.wait_until_complete()
        print(f"Rendered {self.current_frame} frames to {self.output_dir}")

        self._cleanup()

        if args.save_video:
            self._assemble_videos()

    def _cleanup(self) -> None:
        if self.writer:
            self.writer.detach()
            self.writer = None
        for rp in self.render_products:
            rp.destroy()
        self.render_products.clear()
        stage_utils.clear_stage()
        stage_utils.update_stage()

    def _assemble_videos(self) -> None:
        videos_dir = os.path.join(self.output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        for camera_name in args.cameras:
            rgb_dir = os.path.join(self.output_dir, camera_name, "rgb")
            if not os.path.isdir(rgb_dir):
                continue

            frames = sorted(
                [f for f in os.listdir(rgb_dir) if f.endswith(".png")],
                key=_natural_sort_key,
            )
            if not frames:
                continue

            first = cv2.imread(os.path.join(rgb_dir, frames[0]))
            h, w, _ = first.shape

            out_path = os.path.join(
                self.exporter.trial_dir,
                f"video_combined_{args.output_suffix}.mp4",
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, args.framerate, (w, h))
            for fname in frames:
                writer.write(cv2.imread(os.path.join(rgb_dir, fname)))
            writer.release()
            print(f"Saved video: {out_path}")

            shutil.rmtree(os.path.join(self.output_dir, camera_name), ignore_errors=True)


def _collect_trial_dirs(path: str) -> list[str]:
    if os.path.isfile(os.path.join(path, "state_trajectory.npz")):
        return [path]
    dirs = []
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        if os.path.isdir(full) and entry.startswith("trial_"):
            if os.path.isfile(os.path.join(full, "state_trajectory.npz")):
                dirs.append(full)
    return dirs


def main():
    if not args.trial_dir and not args.experiment_dir:
        print("ERROR: Provide either --trial_dir or --experiment_dir")
        sys.exit(1)

    target = args.trial_dir or args.experiment_dir
    trial_dirs = _collect_trial_dirs(target)

    if not trial_dirs:
        print(f"No trial directories with state_trajectory.npz found in {target}")
        sys.exit(1)

    print(f"Found {len(trial_dirs)} trial(s) to render")

    for trial_dir in trial_dirs:
        print(f"\n{'='*80}\nRendering: {trial_dir}\n{'='*80}")
        try:
            exporter = TrialUSDExporter(trial_dir)
            exporter.export_all()
            renderer = IsaacSimRenderer(exporter)
            renderer.render()
        except Exception as e:
            print(f"Failed to render {trial_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nAll done.")


if __name__ == "__main__":
    main()
