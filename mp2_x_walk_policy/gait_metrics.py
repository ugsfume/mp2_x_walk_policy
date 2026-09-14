"""Swing-clearance, stance-slip and footfall-regularity metrics.

Contact-count gates (support histogram, all-feet fraction, foot z-range) cannot
tell a real gait from a shuffle: a sliding gait passes all of them. This module
measures the two quantities that do discriminate -- vertical foot lift during
swing and fore/aft foot travel during stance.

Pure numpy plus an optional matplotlib import, so it is safe to import before
``SimulationApp`` starts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop) index pairs for each maximal run of True in mask."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _quat_xyzw_to_yaw_pitch(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yaw and pitch, radians, from an (N, 4) xyzw quaternion array.

    Isaac Lab stores orientation as ``(x, y, z, w)`` -- see
    ``isaaclab.utils.math.yaw_quat``, whose docstring says so and which computes
    the same yaw expression. Keep the two in step if the asset is ever upgraded:
    under a wxyz convention this returns roll, not yaw. Two checks confirm the
    layout on recorded traces: identity spawns store as (0, 0, 0, 1), and
    PlayRand spawn quats decode to the configured yaw range with near-zero
    roll only under this reading.
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return yaw, pitch


def _quat_wxyz_to_yaw_pitch(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Yaw and pitch, radians, from an (N, 4) wxyz quaternion array.

    Not used by the metrics: the traces this module records are (x, y, z, w),
    and applying this reader to them returns a roll-mixed angle with a pi
    offset at identity. Kept only for quaternions stored in the other
    convention.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return yaw, pitch


def summarize_traces(npz_path) -> dict:
    """Recompute the gait summary from a saved ``*_gait_traces.npz``.

    Lets metric definitions be revised and re-applied to completed runs without
    paying for another Isaac rollout.

    Handles both single-robot traces and multi-env traces, which carry a
    leading env axis. For a multi-env file this returns the aggregate across
    envs; use :func:`summarize_traces_per_env` for the individual rollouts.
    """
    summaries = summarize_traces_per_env(npz_path)
    return summaries[0] if len(summaries) == 1 else aggregate_summaries(summaries)


def summarize_traces_per_env(npz_path) -> list[dict]:
    """Recompute one gait summary per recorded environment."""
    data = np.load(npz_path, allow_pickle=False)
    foot_names = [str(name) for name in data["foot_names"]]
    step_dt = float(data["step_dt"])

    contacts = data["contacts"]
    multi = contacts.ndim == 3  # (E, T, F) rather than (T, F)
    num_envs = contacts.shape[0] if multi else 1

    def slice_env(key, env):
        if key not in data.files:
            return None
        return data[key][env] if multi else data[key]

    summaries = []
    for env in range(num_envs):
        recorder = GaitRecorder(foot_names, step_dt)
        env_contacts = contacts[env] if multi else contacts
        foot_pos = slice_env("foot_pos_w", env)
        root_pos = slice_env("root_pos_w", env)
        torque = slice_env("joint_torque", env)
        quat = slice_env("root_quat_w", env)
        for index in range(env_contacts.shape[0]):
            recorder.record(
                env_contacts[index],
                foot_pos[index],
                root_pos[index],
                None if torque is None else torque[index],
                None if quat is None else quat[index],
            )
        summaries.append(recorder.summary())
    return summaries


#: Scalar gate keys worth reducing across a randomised evaluation set.
AGGREGATE_KEYS = (
    "swing_clearance_median_m",
    "swing_clearance_min_foot_m",
    "foot_path_excess",
    "stance_path_fraction",
    "planted_foot_slip_ratio",
    "duty_factor_mean",
    "swing_count_spread",
    "swing_duration_cv",
    "per_foot_lift_spread_m",
    "per_foot_duty_spread",
    "base_height_std_m",
    "abs_yaw_drift_deg",
    "abs_lateral_displacement_m",
    "abs_pitch_median_deg",
    "joint_torque_abs_mean_nm",
    "cadence_hz",
    "stride_median_m",
    "forward_displacement_m",
    "lateral_displacement_m",
    "mean_body_speed_mps",
    # Spawn-frame displacement: the straightness measure that survives
    # spawn-heading randomisation.
    "forward_displacement_spawnframe_m",
    "lateral_displacement_spawnframe_m",
)


def aggregate_summaries(summaries: list[dict]) -> dict:
    """Reduce per-env summaries to median / IQR / min / max per gate.

    The Play tasks have all reset ranges at zero and corruption disabled, so
    their envs are bit-identical and a single recorder yields n = 1. A
    high-variance quantity such as lateral drift cannot be gated on one
    deterministic rollout, so gates are applied to the median here and the
    spread is reported alongside.
    """
    if not summaries:
        return {"gait_metrics_present": False}
    out = {
        "gait_metrics_present": True,
        "num_rollouts": len(summaries),
        # A rollout set is only as trustworthy as its worst member.
        "gait_metrics_reliable": all(s.get("gait_metrics_reliable", False) for s in summaries),
    }
    for key in AGGREGATE_KEYS:
        values = np.array(
            [s[key] for s in summaries if isinstance(s.get(key), (int, float)) and np.isfinite(s[key])],
            dtype=float,
        )
        if values.size == 0:
            continue
        out[key] = float(np.median(values))
        out[f"{key}__iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        out[f"{key}__min"] = float(values.min())
        out[f"{key}__max"] = float(values.max())
    # Sign matters for drift: medians of absolute values hide whether every
    # rollout turned the same way, which is the systematic-vs-seed question.
    for key in (
        "lateral_displacement_m",
        "yaw_drift_deg",
        "lateral_displacement_spawnframe_m",
        "forward_displacement_spawnframe_m",
    ):
        signed = [s[key] for s in summaries if isinstance(s.get(key), (int, float)) and np.isfinite(s[key])]
        if signed:
            out[f"{key}__signed_mean"] = float(np.mean(signed))
            out[f"{key}__same_sign_fraction"] = float(
                max(sum(v > 0 for v in signed), sum(v < 0 for v in signed)) / len(signed)
            )
    return out


def write_traces_multi(output_dir: Path, stem: str, recorders: list["GaitRecorder"]) -> str:
    """Write one npz holding every recorded env, stacked on a leading axis.

    Same file name and key names as the single-robot writer, so downstream
    readers only have to cope with one extra leading dimension.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}_gait_traces.npz"
    per_env = [recorder._payload() for recorder in recorders]
    payload = {
        key: np.stack([p[key] for p in per_env]) for key in per_env[0]
    }
    payload["foot_names"] = np.array(recorders[0].foot_names)
    payload["step_dt"] = np.array(recorders[0].step_dt)
    np.savez_compressed(path, **payload)
    return str(path)


class GaitRecorder:
    """Accumulate per-step gait data for one robot and reduce it to metrics."""

    def __init__(self, foot_names: list[str], step_dt: float) -> None:
        self.foot_names = list(foot_names)
        self.step_dt = float(step_dt)
        self._contacts: list[np.ndarray] = []
        self._foot_pos: list[np.ndarray] = []
        self._root_pos: list[np.ndarray] = []
        self._torque: list[np.ndarray] = []
        self._quat: list[np.ndarray] = []

    def record(
        self,
        foot_contacts: np.ndarray,
        foot_pos: np.ndarray,
        root_pos: np.ndarray,
        joint_torque: np.ndarray | None = None,
        root_quat: np.ndarray | None = None,
    ) -> None:
        self._contacts.append(np.asarray(foot_contacts, dtype=bool).copy())
        self._foot_pos.append(np.asarray(foot_pos, dtype=np.float64).copy())
        self._root_pos.append(np.asarray(root_pos, dtype=np.float64).copy())
        if joint_torque is not None:
            self._torque.append(np.asarray(joint_torque, dtype=np.float64).copy())
        if root_quat is not None:
            self._quat.append(np.asarray(root_quat, dtype=np.float64).copy())

    @property
    def num_steps(self) -> int:
        return len(self._contacts)

    def _arrays(self):
        contacts = np.asarray(self._contacts)  # (T, F) bool
        foot_pos = np.asarray(self._foot_pos)  # (T, F, 3)
        root_pos = np.asarray(self._root_pos)  # (T, 3)
        return contacts, foot_pos, root_pos

    def summary(self) -> dict:
        """Reduce the recording to the gait-quality gates.

        Definitions that matter:

        swing clearance
            Peak foot height during a swing, measured against that foot's own
            median stance height rather than world z. Foot-link origins sit
            above ground in stance on this asset, so an absolute-z clearance
            would be offset by that height.

        foot_path_excess
            Total foot world-frame horizontal path divided by body horizontal
            displacement. Exactly 1.0 for clean stepping at any duty factor,
            because stance feet are stationary and each foot advances once per
            cycle by the stride. Above 1.0 is slip.

        Swings touching either end of the recording are incomplete and are
        excluded from the per-swing statistics.
        """
        if self.num_steps < 3:
            return {"gait_metrics_present": False}

        contacts, foot_pos, root_pos = self._arrays()
        num_steps, num_feet = contacts.shape
        elapsed = num_steps * self.step_dt

        body_disp = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))
        step_xy = np.linalg.norm(np.diff(foot_pos[:, :, :2], axis=0), axis=-1)  # (T-1, F)
        total_path = step_xy.sum(axis=0)  # (F,)
        # A step counts as stance travel only when the foot was in contact at
        # both ends of the interval, so touchdown/liftoff transitions are not
        # charged as slip.
        stance_interval = contacts[:-1] & contacts[1:]
        stance_path = np.where(stance_interval, step_xy, 0.0).sum(axis=0)

        # Slip, measured geometrically rather than from the contact flag.
        # The 2 N force threshold over-reports contact -- on a clean stepping
        # gait ~17% of "in contact" samples are feet more than 5 mm above the
        # lowest foot -- which inflates any contact-gated slip metric above body speed.
        # The lowest foot in each frame is unambiguously the planted one, needs
        # no threshold, and needs no tuning.
        foot_speed = np.linalg.norm(np.diff(foot_pos[:, :, :2], axis=0), axis=-1) / self.step_dt
        body_speed = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=-1) / self.step_dt
        height_above_lowest = foot_pos[:-1, :, 2] - foot_pos[:-1, :, 2].min(axis=1, keepdims=True)
        planted = height_above_lowest < 1e-9
        mean_body_speed = float(body_speed.mean())
        planted_speed = float(foot_speed[planted].mean()) if planted.any() else float("nan")
        near_planted = height_above_lowest < 0.002
        near_planted_speed = (
            float(foot_speed[near_planted].mean()) if near_planted.any() else float("nan")
        )

        duty = contacts.mean(axis=0)
        # Mean height of each foot above the lowest foot in the same frame.
        # Needs no contact flag and no per-foot datum, so unlike the per-swing
        # clearance it cannot be distorted by contact-flag segmentation.
        mean_lift = (foot_pos[:, :, 2] - foot_pos[:, :, 2].min(axis=1, keepdims=True)).mean(axis=0)
        clearance_by_foot: dict[str, float] = {}
        stride_by_foot: dict[str, float] = {}
        swing_counts: dict[str, int] = {}
        all_clearances: list[float] = []
        all_swing_durations: list[float] = []
        all_strides: list[float] = []

        for index, name in enumerate(self.foot_names):
            in_contact = contacts[:, index]
            stance_z = (
                float(np.median(foot_pos[in_contact, index, 2])) if in_contact.any() else float("nan")
            )
            swings = [
                (start, stop)
                for start, stop in _runs(~in_contact)
                if start > 0 and stop < num_steps
            ]
            swing_counts[name] = len(swings)
            clearances = [float(foot_pos[start:stop, index, 2].max() - stance_z) for start, stop in swings]
            durations = [(stop - start) * self.step_dt for start, stop in swings]
            strides = [
                float(np.linalg.norm(foot_pos[stop - 1, index, :2] - foot_pos[start, index, :2]))
                for start, stop in swings
            ]
            clearance_by_foot[name] = float(np.median(clearances)) if clearances else 0.0
            stride_by_foot[name] = float(np.median(strides)) if strides else 0.0
            all_clearances.extend(clearances)
            all_swing_durations.extend(durations)
            all_strides.extend(strides)

        counts = np.array(list(swing_counts.values()), dtype=float)
        mean_count = float(counts.mean()) if counts.size else 0.0
        count_spread = float(np.abs(counts - mean_count).max() / mean_count) if mean_count > 0 else float("inf")
        durations = np.array(all_swing_durations, dtype=float)
        duration_cv = float(durations.std() / durations.mean()) if durations.size and durations.mean() > 0 else float("inf")

        summary = {
            "gait_metrics_present": True,
            "gait_elapsed_s": elapsed,
            # --- gate 1: swing clearance ---
            "swing_clearance_median_m": float(np.median(all_clearances)) if all_clearances else 0.0,
            "swing_clearance_min_foot_m": float(min(clearance_by_foot.values())) if clearance_by_foot else 0.0,
            "swing_clearance_by_name": clearance_by_foot,
            "swing_count_total": int(sum(swing_counts.values())),
            # --- gates 2 and 3: stance slip ---
            # planted_foot_slip_ratio is the headline slip gate: 0.0 is a foot
            # that stays put, 1.0 is a foot dragging along at body speed.
            "planted_foot_slip_ratio": (
                planted_speed / mean_body_speed if mean_body_speed > 1e-6 else float("inf")
            ),
            "planted_foot_speed_mps": planted_speed,
            "near_planted_foot_slip_ratio": (
                near_planted_speed / mean_body_speed if mean_body_speed > 1e-6 else float("inf")
            ),
            "mean_body_speed_mps": mean_body_speed,
            "foot_path_excess": float(total_path.mean() / body_disp) if body_disp > 1e-6 else float("inf"),
            "foot_path_excess_by_name": {
                name: (float(total_path[i] / body_disp) if body_disp > 1e-6 else float("inf"))
                for i, name in enumerate(self.foot_names)
            },
            "stance_path_fraction": float(stance_path.sum() / total_path.sum()) if total_path.sum() > 0 else 0.0,
            "stance_path_fraction_by_name": {
                name: (float(stance_path[i] / total_path[i]) if total_path[i] > 0 else 0.0)
                for i, name in enumerate(self.foot_names)
            },
            # --- gates 5 and 6: regularity ---
            "duty_factor_by_name": {name: float(duty[i]) for i, name in enumerate(self.foot_names)},
            "duty_factor_mean": float(duty.mean()),
            "swing_count_by_name": swing_counts,
            "swing_count_spread": count_spread,
            "swing_duration_cv": duration_cv,
            "swing_duration_median_s": float(np.median(durations)) if durations.size else 0.0,
            "cadence_hz": mean_count / elapsed if elapsed > 0 else 0.0,
            "stride_median_m": float(np.median(all_strides)) if all_strides else 0.0,
            "stride_by_name": stride_by_foot,
            "body_horizontal_displacement_m": body_disp,
            # --- gate 7: straightness ---
            "forward_displacement_m": float(root_pos[-1, 0] - root_pos[0, 0]),
            "lateral_displacement_m": float(root_pos[-1, 1] - root_pos[0, 1]),
            # Kept as its own key so the aggregate takes median(|lateral|) and
            # not |median(lateral)|. Those differ enormously once rollouts are
            # randomised: a policy whose rollouts span -0.40 to +0.40 m has a
            # signed median near zero while the typical excursion is 0.12 m.
            # Gating the former would pass a policy that wanders badly in both
            # directions.
            "abs_lateral_displacement_m": float(abs(root_pos[-1, 1] - root_pos[0, 1])),
            # --- gate 8: is the gait actually symmetric? ---
            # swing_count_spread counts swings without weighing them, so it
            # can read "even" for a gait whose per-foot duty is e.g.
            # [0.80, 0.49, 0.59, 0.87] and mean lift [7.0, 17.4, 10.2, 4.5] mm
            # -- one leg doing nearly all the swinging while another is a
            # near-permanent post. These two catch that: a symmetric gait
            # measures ~0.3 mm and ~0.06, the example above 12.9 mm and 0.38.
            "per_foot_mean_lift_m": {
                name: float(mean_lift[i]) for i, name in enumerate(self.foot_names)
            },
            "per_foot_lift_spread_m": float(mean_lift.max() - mean_lift.min()),
            "per_foot_duty_spread": float(duty.max() - duty.min()),
            # --- gate 9: posture ---
            # Bouncing and a nose-up body are visible in video but invisible to
            # the contact and clearance gates; these two make them measurable.
            "base_height_mean_m": float(root_pos[:, 2].mean()),
            "base_height_std_m": float(root_pos[:, 2].std()),
        }

        if self._quat:
            yaw, pitch = _quat_xyzw_to_yaw_pitch(np.asarray(self._quat))
            # Unwrap before differencing: a drift past +-pi would otherwise
            # read as a jump back the other way.
            yaw_unwrapped = np.unwrap(yaw)
            yaw_drift = float(np.degrees(yaw_unwrapped[-1] - yaw_unwrapped[0]))
            pitch_deg = np.degrees(pitch)
            summary.update(
                {
                    # A large world-frame lateral displacement can be almost
                    # entirely heading rotation (tens of degrees over an
                    # episode) with near-zero body-frame lateral velocity.
                    # Reporting the two separately distinguishes turning from
                    # sideslip.
                    "yaw_drift_deg": yaw_drift,
                    "abs_yaw_drift_deg": abs(yaw_drift),
                    "yaw_drift_rate_deg_s": yaw_drift / elapsed if elapsed > 0 else 0.0,
                    "pitch_median_deg": float(np.median(pitch_deg)),
                    "abs_pitch_median_deg": float(abs(np.median(pitch_deg))),
                    "pitch_abs_max_deg": float(np.abs(pitch_deg).max()),
                }
            )
            # With randomised initial conditions the spawn heading is no longer
            # world +x, so world-frame lateral displacement stops meaning
            # "sideways relative to where the robot was pointing". Rotate the
            # displacement into the spawn frame so the straightness number
            # survives randomisation.
            yaw0 = float(yaw_unwrapped[0])
            delta = root_pos[-1, :2] - root_pos[0, :2]
            cos0, sin0 = np.cos(-yaw0), np.sin(-yaw0)
            summary["forward_displacement_spawnframe_m"] = float(
                cos0 * delta[0] - sin0 * delta[1]
            )
            summary["lateral_displacement_spawnframe_m"] = float(
                sin0 * delta[0] + cos0 * delta[1]
            )

        # With almost no swings the per-swing statistics are computed from one
        # or two noisy segments and can even come out negative. That is a
        # degenerate input, not a measurement -- flag it rather than let a
        # non-stepping policy report a plausible-looking clearance.
        summary["gait_metrics_reliable"] = bool(summary["swing_count_total"] >= 8)

        if self._torque:
            torque = np.abs(np.asarray(self._torque))
            summary["joint_torque_abs_max_nm"] = float(torque.max())
            summary["joint_torque_abs_mean_nm"] = float(torque.mean())
            summary["joint_torque_abs_max_by_index"] = [float(v) for v in torque.max(axis=0)]
        return summary

    def write_traces(self, output_dir: Path, stem: str) -> str:
        """Dump the raw per-step arrays so analysis does not need a re-run."""
        contacts, foot_pos, root_pos = self._arrays()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{stem}_gait_traces.npz"
        payload = {
            "contacts": contacts,
            "foot_pos_w": foot_pos,
            "root_pos_w": root_pos,
            "foot_names": np.array(self.foot_names),
            "step_dt": np.array(self.step_dt),
        }
        if self._torque:
            payload["joint_torque"] = np.asarray(self._torque)
        if self._quat:
            payload["root_quat_w"] = np.asarray(self._quat)
        np.savez_compressed(path, **payload)
        return str(path)

    def _payload(self) -> dict:
        contacts, foot_pos, root_pos = self._arrays()
        payload = {"contacts": contacts, "foot_pos_w": foot_pos, "root_pos_w": root_pos}
        if self._torque:
            payload["joint_torque"] = np.asarray(self._torque)
        if self._quat:
            payload["root_quat_w"] = np.asarray(self._quat)
        return payload

    def write_plots(self, output_dir: Path, stem: str, title: str = "") -> list[str]:
        """Write the gait diagram, foot-height traces and limb-cycle plot.

        These are the visual review artifacts. The gait diagram in particular
        makes a shuffle obvious at a glance, which a wide-framed rollout video
        does not.
        """
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover - plotting is best effort
            return []

        if self.num_steps < 3:
            return []

        contacts, foot_pos, root_pos = self._arrays()
        num_steps = contacts.shape[0]
        time = np.arange(num_steps) * self.step_dt
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)

        ax = axes[0]
        for index, name in enumerate(self.foot_names):
            for start, stop in _runs(contacts[:, index]):
                ax.broken_barh([(time[start], (stop - start) * self.step_dt)], (index - 0.35, 0.7), color="#2b2b2b")
        ax.set_yticks(range(len(self.foot_names)))
        ax.set_yticklabels(self.foot_names)
        ax.set_ylim(-0.6, len(self.foot_names) - 0.4)
        ax.set_xlabel("time [s]")
        ax.set_title(f"gait diagram (bar = stance){' - ' + title if title else ''}")
        ax.grid(axis="x", alpha=0.3)

        ax = axes[1]
        for index, name in enumerate(self.foot_names):
            in_contact = contacts[:, index]
            stance_z = float(np.median(foot_pos[in_contact, index, 2])) if in_contact.any() else 0.0
            ax.plot(time, (foot_pos[:, index, 2] - stance_z) * 1000.0, label=name, linewidth=1.0)
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.axhline(30.0, color="tab:green", linestyle="--", linewidth=0.9, label="reference z_clearance 30 mm")
        # Must track the swing_clearance_median_m gate used in reporting, or the
        # plot and the pass/fail verdict disagree with each other.
        ax.axhline(15.0, color="tab:orange", linestyle=":", linewidth=0.9, label="clearance gate 15 mm")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("foot height above own stance [mm]")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)

        ax = axes[2]
        for index, name in enumerate(self.foot_names):
            in_contact = contacts[:, index]
            stance_z = float(np.median(foot_pos[in_contact, index, 2])) if in_contact.any() else 0.0
            rel_x = (foot_pos[:, index, 0] - root_pos[:, 0]) * 1000.0
            rel_z = (foot_pos[:, index, 2] - stance_z) * 1000.0
            ax.plot(rel_x, rel_z, linewidth=0.8, label=name)
        ax.set_xlabel("foot x relative to body [mm]")
        ax.set_ylabel("foot height above own stance [mm]")
        ax.set_title("limb cycle -- a clean gait draws a loop, a shuffle draws a line")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        path = output_dir / f"{stem}_gait.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(str(path))
        return written
