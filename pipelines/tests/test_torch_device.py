from __future__ import annotations

import unittest

from pipelines.common.torch_device import move_wrapped_model_to_device, resolve_torch_device


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeMPS:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _FakeMPS(mps_available)


class _FakeTorch:
    def __init__(self, cuda_available: bool = False, mps_available: bool = False) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.backends = _FakeBackends(mps_available)


class _FakeInnerModel:
    def __init__(self) -> None:
        self.moved_to: str | None = None

    def to(self, device: str) -> None:
        self.moved_to = device


class _FakeWrappedModel:
    def __init__(self) -> None:
        self.model = _FakeInnerModel()


class TorchDeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_then_mps_then_cpu(self) -> None:
        self.assertEqual(resolve_torch_device("auto", "TEST_DEVICE", _FakeTorch(cuda_available=True, mps_available=True)), "cuda")
        self.assertEqual(resolve_torch_device("auto", "TEST_DEVICE", _FakeTorch(mps_available=True)), "mps")
        self.assertEqual(resolve_torch_device("auto", "TEST_DEVICE", _FakeTorch()), "cpu")

    def test_explicit_unavailable_gpu_device_errors(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "TEST_DEVICE=mps"):
            resolve_torch_device("mps", "TEST_DEVICE", _FakeTorch())

    def test_rejects_unknown_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "TEST_DEVICE"):
            resolve_torch_device("metal", "TEST_DEVICE", _FakeTorch())

    def test_moves_wrapped_cross_encoder_model_to_device(self) -> None:
        wrapped = _FakeWrappedModel()

        move_wrapped_model_to_device(wrapped, "mps")

        self.assertEqual(wrapped.model.moved_to, "mps")


if __name__ == "__main__":
    unittest.main()
