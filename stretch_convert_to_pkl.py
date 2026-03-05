#!/usr/bin/env python3
"""
convert_stretch_to_pkl.py — Convert MCAP episodes to ViSk pkl format.

Usage:
    python convert_stretch_to_pkl.py place_coffee_cup

Reads from:  ~/VTAM/data/processed/<task>/episode_*.mcap
Writes to:   ~/VTAM/dependencies/visuoskin/data/processed_data_pkl/<task>.pkl
"""

import argparse
import glob
import os
import sys
import numpy as np
import pickle as pkl
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
VTAM_ROOT = os.path.expanduser("~/VTAM")
sys.path.insert(0, os.path.join(VTAM_ROOT, "training", "utils"))
sys.path.insert(0, os.path.join(VTAM_ROOT, "dependencies", "lerobot"))

from mcap_ros2.reader import read_ros2_messages
from workspace_projection import WorkspaceProjector

# ── Config ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE = (128, 128)  # ViSk default
MIN_FRAMES = 30

SYNC_TOPIC    = "/sync_pulse"
IMAGE_TOPIC   = "/camera_arm/color/image_rect_raw/compressed"
GRIPPER_TOPIC = "/gripper_width_normalized"
TACTILE_LEFT  = "/tactile_left"
TACTILE_RIGHT = "/tactile_right"

TF_CHAIN = [
    ("base_link", "umi_disconnect"),
    ("umi_disconnect", "umi_gripper"),
]

ALL_TOPICS = [SYNC_TOPIC, IMAGE_TOPIC, GRIPPER_TOPIC,
              TACTILE_LEFT, TACTILE_RIGHT, "/tf"]


# ── TF utilities 

def tf_to_matrix(pose7):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose7[3:]).as_matrix()
    T[:3, 3] = pose7[:3]
    return T


def extract_ee_pose(tf_data, tf_ts):
    drive_pair = TF_CHAIN[0]
    if not tf_data[drive_pair]:
        return [], []

    ts_arrays = {p: np.array(tf_ts[p]) for p in TF_CHAIN}
    ee_poses, ee_timestamps = [], []

    for i in range(len(tf_data[drive_pair])):
        t = tf_ts[drive_pair][i]
        T_total = tf_to_matrix(tf_data[drive_pair][i])
        for pair in TF_CHAIN[1:]:
            idx = np.clip(np.searchsorted(ts_arrays[pair], t) - 1,
                          0, len(tf_data[pair]) - 1)
            T_total = T_total @ tf_to_matrix(tf_data[pair][idx])

        pos = T_total[:3, 3]
        quat = Rotation.from_matrix(T_total[:3, :3]).as_quat()
        ee_poses.append(np.concatenate([pos, quat]).astype(np.float32))
        ee_timestamps.append(t)

    return ee_poses, ee_timestamps


# ── Per-episode processing

def process_episode(mcap_path, projector):
    sync_ts = []
    tf_data = {p: [] for p in TF_CHAIN}
    tf_ts   = {p: [] for p in TF_CHAIN}
    img_buf, img_ts = [], []
    gripper_buf, gripper_ts = [], []
    tactile_left_buf, tactile_left_ts = [], []
    tactile_right_buf, tactile_right_ts = [], []

    for msg in read_ros2_messages(mcap_path, topics=ALL_TOPICS):
        topic = msg.channel.topic
        t = msg.publish_time_ns / 1e9

        if topic == SYNC_TOPIC:
            stamp = msg.ros_msg.header.stamp
            sync_ts.append(stamp.sec + stamp.nanosec / 1e9)

        elif topic == "/tf":
            for tf in msg.ros_msg.transforms:
                pair = (tf.header.frame_id, tf.child_frame_id)
                if pair in tf_data:
                    tr, ro = tf.transform.translation, tf.transform.rotation
                    tf_data[pair].append(
                        np.array([tr.x, tr.y, tr.z, ro.x, ro.y, ro.z, ro.w]))
                    tf_ts[pair].append(t)

        elif topic == IMAGE_TOPIC:
            buf = np.frombuffer(msg.ros_msg.data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Center crop to square then resize
            h, w = img.shape[:2]
            s = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            img = img[y0:y0+s, x0:x0+s]
            img = cv2.resize(img, IMAGE_SIZE)
            img_buf.append(img)
            img_ts.append(t)

        elif topic == GRIPPER_TOPIC:
            gripper_buf.append(float(msg.ros_msg.data))
            gripper_ts.append(t)

        elif topic == TACTILE_LEFT:
            tactile_left_buf.append(np.array(msg.ros_msg.data, dtype=np.float32))
            tactile_left_ts.append(t)

        elif topic == TACTILE_RIGHT:
            tactile_right_buf.append(np.array(msg.ros_msg.data, dtype=np.float32))
            tactile_right_ts.append(t)

    # ── Validate 
    if len(sync_ts) < MIN_FRAMES:
        print(f"  SKIP: {len(sync_ts)} sync frames")
        return None
    if not tf_data[TF_CHAIN[0]]:
        print("  SKIP: no TF data")
        return None
    if not img_buf:
        print("  SKIP: no images")
        return None
    if not tactile_left_buf or not tactile_right_buf:
        print("  SKIP: missing tactile data")
        return None

    # ── EE poses 
    ee_poses_raw, ee_ts = extract_ee_pose(tf_data, tf_ts)
    if not ee_poses_raw:
        print("  SKIP: EE extraction failed")
        return None

    # ── Workspace projection 
    sync_arr = np.array(sync_ts)
    positions = np.array([p[:3] for p in ee_poses_raw], dtype=np.float64)
    quaternions = np.array([p[3:] for p in ee_poses_raw], dtype=np.float64)
    result = projector.project(positions, quaternions, np.array([t for t in ee_ts]))
    ee_canonical = result['ee_poses'].astype(np.float32)  # (N_tf, 7) xyz+quat

    # ── Snap to sync timestamps 
    ee_ts_arr = np.array([t for t in ee_ts])
    img_ts_arr = np.array(img_ts)
    grip_ts_arr = np.array(gripper_ts)
    tl_ts_arr = np.array(tactile_left_ts)
    tr_ts_arr = np.array(tactile_right_ts)

    def snap(buf, ts_arr):
        idx = np.clip(np.searchsorted(ts_arr, sync_arr) - 1, 0, len(buf) - 1)
        return [buf[i] for i in idx]

    N = len(sync_ts)
    ee_synced = snap(ee_canonical, ee_ts_arr)          # list of (7,)
    images_synced = snap(img_buf, img_ts_arr)           # list of (H,W,3)
    gripper_synced = snap(gripper_buf, grip_ts_arr)     # list of float
    tl_synced = snap(tactile_left_buf, tl_ts_arr)       # list of (15,)
    tr_synced = snap(tactile_right_buf, tr_ts_arr)      # list of (15,)

    # ── Convert to ViSk format ─────────────────────────────────────────────
    # Images: (T, H, W, 3) uint8
    cam_gripper = np.array(images_synced, dtype=np.uint8)

    # EE: quat → axis-angle for cartesian_states (T, 6)
    ee_arr = np.array(ee_synced, dtype=np.float32)      # (T, 7) xyz+quat
    positions_canon = ee_arr[:, :3]
    quats_canon = ee_arr[:, 3:]
    rotvecs = Rotation.from_quat(quats_canon).as_rotvec().astype(np.float32)
    cartesian_states = np.concatenate([positions_canon, rotvecs], axis=1)  # (T, 6)

    # Gripper: (T,)
    gripper_states = np.array(gripper_synced, dtype=np.float32)

    # Tactile: concat left + right → (T, 30)
    sensor_states = np.concatenate(
        [np.array(tl_synced), np.array(tr_synced)], axis=1
    ).astype(np.float32)

    print(f"  {N} frames | pos [{positions_canon[:,0].min():.3f}, {positions_canon[:,0].max():.3f}] "
          f"| gripper [{gripper_states.min():.2f}, {gripper_states.max():.2f}] "
          f"| tactile shape {sensor_states.shape}")

    return {
        "cam_gripper": cam_gripper,
        "cartesian_states": cartesian_states,
        "gripper_states": gripper_states,
        "sensor_states": sensor_states,
    }


# ── Main 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Task name (e.g. place_coffee_cup)")
    args = parser.parse_args()

    processed_dir = Path(VTAM_ROOT) / "data" / "processed" / args.task
    mcap_paths = sorted(glob.glob(str(processed_dir / "*.mcap")))
    if not mcap_paths:
        print(f"No .mcap files in {processed_dir}")
        sys.exit(1)

    print(f"Found {len(mcap_paths)} episodes\n")
    projector = WorkspaceProjector()

    observations = []
    global_max_cart = None
    global_min_cart = None
    global_max_grip = -np.inf
    global_min_grip = np.inf

    for i, mcap_path in enumerate(tqdm(mcap_paths, desc="Converting")):
        print(f"\nEpisode {i}: {os.path.basename(mcap_path)}")
        ep = process_episode(mcap_path, projector)
        if ep is None:
            continue

        observations.append(ep)

        # Track stats
        cart = ep["cartesian_states"]
        grip = ep["gripper_states"]
        if global_max_cart is None:
            global_max_cart = cart.max(axis=0)
            global_min_cart = cart.min(axis=0)
        else:
            global_max_cart = np.maximum(global_max_cart, cart.max(axis=0))
            global_min_cart = np.minimum(global_min_cart, cart.min(axis=0))
        global_max_grip = max(global_max_grip, grip.max())
        global_min_grip = min(global_min_grip, grip.min())

    if not observations:
        print("No valid episodes. Exiting.")
        sys.exit(1)

    # ── Save PKL 
    out_dir = Path(VTAM_ROOT) / "dependencies" / "visuoskin" / "data" / "processed_data_pkl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task}.pkl"

    data = {
        "observations": observations,
        "max_cartesian": global_max_cart,
        "min_cartesian": global_min_cart,
        "max_gripper": global_max_grip,
        "min_gripper": global_min_grip,
    }

    pkl.dump(data, open(str(out_path), "wb"))
    print(f"\nSaved {len(observations)} episodes to {out_path}")
    print(f"  cartesian range: {global_min_cart} → {global_max_cart}")
    print(f"  gripper range: {global_min_grip:.3f} → {global_max_grip:.3f}")


if __name__ == "__main__":
    main()