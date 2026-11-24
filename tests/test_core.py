"""Fast checks that need no model download."""
import hashlib
import json

import numpy as np
import pytest

from wspd import config, metrics, splits


def test_splits_are_disjoint_and_complete():
    built = splits.build_splits()
    assert {len(built["dev"]), len(built["test_seen"]), len(built["test_new"])} == {24, 40, 32}
    prompts = [it.prompt for items in built.values() for it in items]
    assert len(prompts) == len(set(prompts)) == 96
    seen = {it.category for it in built["dev"]}
    assert seen == {it.category for it in built["test_seen"]}
    assert not seen & {it.category for it in built["test_new"]}
    assert len({it.category for it in built["dev"][:8]}) == 8


def test_splits_are_deterministic():
    assert splits.build_splits() == splits.build_splits()


def test_development_prompts_are_unchanged():
    """The 16 prompts and seeds behind sections 1 to 4 of the results."""
    payload = json.dumps(splits.as_json(splits.dev_items(16)), sort_keys=True).encode()
    assert hashlib.sha256(payload).hexdigest().startswith("58176ddbf5cd03e7")


def _noise_image(seed=0, blur=0):
    rng = np.random.default_rng(seed)
    x = rng.random((1, 256, 256, 3))
    if blur:
        k = np.ones(blur) / blur
        for axis in (1, 2):
            x = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis, x)
    return (x * 255).astype(np.uint8)


def test_blur_produces_positive_deficit_at_high_frequency():
    ref, soft = _noise_image(), _noise_image(blur=5)
    deficits = metrics.band_deficit_db(soft, ref)
    assert deficits["grain"][0] > deficits["structure"][0] and deficits["grain"][0] > 3


def test_deficit_of_identical_images_is_zero():
    ref = _noise_image()
    assert all(abs(v[0]) < 1e-9 for v in metrics.band_deficit_db(ref, ref).values())


def test_swd_separates_blur_from_identity():
    ref = np.concatenate([_noise_image(i) for i in range(4)])
    soft = np.concatenate([_noise_image(i, blur=5) for i in range(4)])
    other = np.concatenate([_noise_image(i + 10) for i in range(4)])
    assert metrics.swd(soft, ref)["mean"] > 5 * metrics.swd(other, ref)["mean"]


def test_paired_bootstrap_brackets_the_mean():
    delta = np.random.default_rng(0).normal(1.0, 0.5, 64)
    ci = metrics.paired_bootstrap(delta)
    assert ci["lo"] < ci["mean"] < ci["hi"] and ci["lo"] > 0.5


def test_step_equivalent_interpolates_and_clamps():
    steps, curve = [5, 10, 20], [4.0, 2.0, 1.0]
    assert metrics.step_equivalent(2.0, steps, curve) == pytest.approx(10)
    assert 10 < metrics.step_equivalent(1.5, steps, curve) < 20
    assert metrics.step_equivalent(9.0, steps, curve) == 5
    assert metrics.step_equivalent(0.1, steps, curve) == 20


def test_texture_metric_is_mean_absolute_coarse_and_fine():
    class Stub(metrics.Evaluator):
        def lpips(self, images, refs, batch=8):
            return np.zeros(len(images))

        def clip_similarity(self, images, prompts, batch=16):
            return np.zeros(len(images))

    ref, soft = _noise_image(), _noise_image(blur=5)
    out = Stub("cpu").evaluate(soft, ref, ["x"])
    expected = (abs(out["deficit_coarse"][0]) + abs(out["deficit_fine"][0])) / 2
    assert out["texture"][0] == pytest.approx(expected)
    assert metrics.summarize(out)["texture"] == pytest.approx(expected)


def test_device_lock_can_be_taken_and_released(tmp_path, monkeypatch):
    from wspd.store import device_lock

    monkeypatch.setattr(config, "RUNS", tmp_path / "runs")
    for _ in range(2):
        with device_lock():
            assert (tmp_path / "runs" / ".device.lock").exists()
