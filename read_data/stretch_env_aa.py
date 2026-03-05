from collections import defaultdict
import random
import numpy as np
import pickle as pkl
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset
from scipy.spatial.transform import Rotation as R


def get_quaternion_orientation(cartesian):
    new_cartesian = []
    for i in range(len(cartesian)):
        pos = cartesian[i, :3]
        ori = cartesian[i, 3:]
        quat = R.from_rotvec(ori).as_quat()
        new_cartesian.append(np.concatenate([pos, quat], axis=-1))
    return np.array(new_cartesian, dtype=np.float32)


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        tasks,
        num_demos_per_task,
        temporal_agg,
        num_queries,
        img_size,
        action_after_steps,
        store_actions,
        pixel_keys,
        aux_keys,
        subsample,
        skip_first_n,
        relative_actions,
        random_mask_proprio,
        sensor_params,
    ):
        self._img_size = img_size
        self._action_after_steps = action_after_steps
        self._store_actions = store_actions
        self._pixel_keys = pixel_keys
        self._aux_keys = aux_keys
        self._random_mask_proprio = random_mask_proprio
        self._subtract_sensor_baseline = sensor_params.subtract_sensor_baseline
        self._num_anyskin_sensors = 2

        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # get data paths
        self._paths = {}
        for idx, task in enumerate(tasks):
            self._paths[idx] = Path(path) / f"{task}.pkl"

        if self._store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._max_episode_len = 0
        self._num_samples = 0
        min_stat, max_stat = None, None
        min_sensor_stat, max_sensor_stat = None, None
        min_act, max_act = None, None

        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = data["observations"]

            self._episodes[_path_idx] = []
            for i in range(min(num_demos_per_task, len(observations))):
                obs = observations[i]

                # Build actions: absolute EE pose + gripper (7D)
                # cartesian_states is (T, 6) = xyz + rotvec
                # gripper_states is (T,)
                actions = np.concatenate([
                    obs["cartesian_states"],
                    obs["gripper_states"][:, None],
                ], axis=1).astype(np.float32)

                if len(actions) == 0:
                    continue

                # skip first n
                if skip_first_n:
                    for key in obs:
                        obs[key] = obs[key][skip_first_n:]
                    actions = actions[skip_first_n:]

                # subsample
                if subsample:
                    for key in obs:
                        obs[key] = obs[key][::subsample]
                    actions = actions[::subsample]

                # Action target: shift by action_after_steps
                actions = actions[self._action_after_steps:]

                # Convert cartesian_states to quaternion for proprioceptive input
                obs["cartesian_states"] = get_quaternion_orientation(
                    obs["cartesian_states"]
                )

                # Process sensor data (baseline subtraction)
                sensor_baseline = np.median(
                    obs["sensor_states"][:5], axis=0, keepdims=True
                )
                if self._subtract_sensor_baseline:
                    obs["sensor_states"] = obs["sensor_states"] - sensor_baseline
                    if max_sensor_stat is None:
                        max_sensor_stat = np.max(obs["sensor_states"], axis=0)
                        min_sensor_stat = np.min(obs["sensor_states"], axis=0)
                    else:
                        max_sensor_stat = np.maximum(
                            max_sensor_stat, np.max(obs["sensor_states"], axis=0))
                        min_sensor_stat = np.minimum(
                            min_sensor_stat, np.min(obs["sensor_states"], axis=0))

                # Split sensor into per-finger
                for sensor_idx in range(self._num_anyskin_sensors):
                    obs[f"sensor{sensor_idx}_states"] = obs["sensor_states"][
                        ..., sensor_idx * 15 : (sensor_idx + 1) * 15
                    ]

                # Prepend duplicate first frame (ViSk convention)
                for key in obs:
                    obs[key] = np.concatenate([[obs[key][0]], obs[key]], axis=0)
                actions = np.concatenate([[actions[0]], actions], axis=0)

                episode = dict(observation=obs, action=actions)
                self._episodes[_path_idx].append(episode)
                self._max_episode_len = max(
                    self._max_episode_len, len(obs[self._pixel_keys[0]]))
                self._num_samples += len(obs[self._pixel_keys[0]])

                # Stats
                if min_act is None:
                    min_act = np.min(actions, axis=0)
                    max_act = np.max(actions, axis=0)
                else:
                    min_act = np.minimum(min_act, np.min(actions, axis=0))
                    max_act = np.maximum(max_act, np.max(actions, axis=0))

                if self._store_actions:
                    self.actions.append(actions)

            # Cartesian stats (with quaternion bounds)
            max_cartesian = np.concatenate([data["max_cartesian"][:3], [1]*4])
            min_cartesian = np.concatenate([data["min_cartesian"][:3], [-1]*4])
            max_val = np.concatenate([max_cartesian, data["max_gripper"][None]])
            min_val = np.concatenate([min_cartesian, data["min_gripper"][None]])
            if max_stat is None:
                max_stat = max_val
                min_stat = min_val
            else:
                max_stat = np.maximum(max_stat, max_val)
                min_stat = np.minimum(min_stat, min_val)

        # Fix rotation normalization range (ViSk convention)
        min_act[3:6], max_act[3:6] = 0, 1

        self.stats = {
            "actions": {"min": min_act, "max": max_act},
            "proprioceptive": {"min": min_stat, "max": max_stat},
        }

        # Sensor normalization (baseline-subtracted: use 3*std clipped)
        if self._subtract_sensor_baseline:
            max_sensor_stat = np.maximum(
                np.abs(max_sensor_stat), np.abs(min_sensor_stat))
            min_sensor_stat = np.zeros_like(max_sensor_stat)

        for sensor_idx in range(self._num_anyskin_sensors):
            mask = np.zeros_like(min_sensor_stat, dtype=bool)
            mask[sensor_idx * 15 : (sensor_idx + 1) * 15] = True
            self.stats[f"sensor{sensor_idx}"] = {
                "min": min_sensor_stat[mask],
                "max": max_sensor_stat[mask],
            }

        # Refine sensor stats with std-based normalization
        for key in self.stats:
            if key.startswith("sensor"):
                all_sensor = np.concatenate([
                    ep["observation"][f"{key}_states"]
                    for eps in self._episodes.values()
                    for ep in eps
                ], axis=0)
                sensor_std = np.std(all_sensor, axis=0).reshape((5, 3)).max(axis=0)
                sensor_std[:2] = sensor_std[:2].max()
                sensor_std = np.clip(sensor_std * 3, a_min=100, a_max=None)
                self.stats[key]["max"] = np.tile(
                    sensor_std, int(self.stats[key]["max"].shape[0] / 3))

        # Augmentation
        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(self._img_size, padding=4),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
        ])

        self.envs_till_idx = len(self._episodes)
        self.prob = np.ones(self.envs_till_idx) / self.envs_till_idx

    def preprocess(self, key, x):
        return (x - self.stats[key]["min"]) / (
            self.stats[key]["max"] - self.stats[key]["min"] + 1e-5
        )

    def _sample(self):
        idx = np.random.choice(list(self._episodes.keys()), p=self.prob)
        episode = random.choice(self._episodes[idx])
        observations = episode["observation"]
        actions = episode["action"]

        sample_idx = np.random.randint(1, len(observations[self._pixel_keys[0]]) - 1)

        # Sample pixel obs
        sampled_pixel = {}
        for key in self._pixel_keys:
            sampled_pixel[key] = observations[key][-(sample_idx + 1):-sample_idx]
            sampled_pixel[key] = torch.stack([
                self.aug(sampled_pixel[key][i])
                for i in range(len(sampled_pixel[key]))
            ])

        # Sample proprioceptive + sensor
        sampled_state = {}
        sampled_state["proprioceptive"] = np.concatenate([
            observations["cartesian_states"][-(sample_idx + 1):-sample_idx],
            observations["gripper_states"][-(sample_idx + 1):-sample_idx][:, None],
        ], axis=1)

        if self._random_mask_proprio and np.random.rand() < 0.5:
            sampled_state["proprioceptive"] = (
                np.ones_like(sampled_state["proprioceptive"])
                * self.stats["proprioceptive"]["min"]
            )

        for sensor_idx in range(self._num_anyskin_sensors):
            skey = f"sensor{sensor_idx}"
            sampled_state[skey] = observations[f"{skey}_states"][
                -(sample_idx + 1):-sample_idx
            ]

        # Sample action
        if self._temporal_agg:
            num_actions = 1 + self._num_queries - 1
            act = np.zeros((num_actions, actions.shape[-1]))
            if num_actions - sample_idx < 0:
                act[:num_actions] = actions[-sample_idx:-sample_idx + num_actions]
            else:
                act[:sample_idx] = actions[-sample_idx:]
                act[sample_idx:] = actions[-1]
            sampled_action = np.lib.stride_tricks.sliding_window_view(
                act, (self._num_queries, actions.shape[-1]))
            sampled_action = sampled_action[:, 0]
        else:
            sampled_action = actions[-(sample_idx + 1):-sample_idx]

        # Build return dict
        return_dict = {}
        for key in self._pixel_keys:
            return_dict[key] = sampled_pixel[key]
        for key in self._aux_keys:
            return_dict[key] = self.preprocess(key, sampled_state[key])
        return_dict["actions"] = self.preprocess("actions", sampled_action)
        return return_dict

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples