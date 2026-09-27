"""experiment.py 的收敛性检查、分片与配置测试。"""

from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from src.experiment import find_nonconverged, shard, target_cfg


def _write(root, name: str, val: float) -> None:
    d = root / name
    d.mkdir()
    (d / "history.json").write_text(
        json.dumps({"history": [{"epoch": 1, "train_loss": val, "val_loss": val}]}),
        encoding="utf-8",
    )


def test_groupwise_criterion_catches_collapse_only(tmp_path):
    """高噪声组整体偏高是条件本身难，不能误判；低噪声组里的塌陷必须抓到。"""
    vals = {"a0": 1.6, "a1": 1.7, "a2": 4.4, "b0": 2.7, "b1": 2.8, "b2": 2.6}
    for n, v in vals.items():
        _write(tmp_path, n, v)
    bad = find_nonconverged(tmp_path, {"low": ["a0", "a1", "a2"],
                                       "high": ["b0", "b1", "b2"]}, 2.0)
    assert set(bad) == {"a2"}
    # 同一批数据若用全体中位数（2.65），a2 = 4.4 < 2 × 2.65，会被漏掉 —— 这正是改成逐组判定的原因
    assert 4.4 < 2.0 * 2.65


def test_groups_with_fewer_than_three_runs_are_not_judged(tmp_path):
    _write(tmp_path, "x0", 1.0)
    _write(tmp_path, "x1", 5.0)
    assert find_nonconverged(tmp_path, {"x": ["x0", "x1"]}, 2.0) == {}


def test_missing_history_is_skipped(tmp_path):
    for n, v in {"a0": 1.0, "a1": 1.1, "a2": 1.2}.items():
        _write(tmp_path, n, v)
    assert find_nonconverged(tmp_path, {"g": ["a0", "a1", "a2", "ghost"]}, 2.0) == {}


def test_shard_partitions_jobs_exactly():
    jobs = list(range(11))
    parts = [shard(jobs, f"{k}/3") for k in range(3)]
    assert sorted(sum(parts, [])) == jobs
    assert all(abs(len(p) - len(jobs) / 3) < 1 for p in parts)
    assert shard(jobs, None) == jobs


def test_shard_rejects_bad_spec():
    with pytest.raises(ValueError):
        shard([1, 2, 3], "3/3")


def test_target_cfg_overrides_depth():
    base = OmegaConf.load("configs/fast.yaml")
    cfg = target_cfg(base, "x", 1, n_blocks=4)
    assert cfg.model.n_blocks == 4
    assert cfg.model.hidden == 30 and cfg.feature.n_mels == 6 and cfg.feature.frames == 5
    with pytest.raises(KeyError):
        target_cfg(base, "x", 1, depth=4)
