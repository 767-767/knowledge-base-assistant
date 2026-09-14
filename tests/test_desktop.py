from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import desktop


class _Process:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class DesktopTests(unittest.TestCase):
    def test_startup_timeout_keeps_waiting(self):
        process = _Process()
        with (
            patch("desktop.urlopen", side_effect=[TimeoutError, _Response()]),
            patch("desktop.time.sleep"),
        ):
            desktop._wait_until_ready(process, "http://127.0.0.1:12345", "secret")

    def test_local_server_uses_private_single_slot_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            server = Path(directory, "llama-server")
            model = Path(directory, "model.gguf")
            server.touch()
            model.touch()
            process = _Process()
            with (
                patch("desktop._free_port", return_value=12345),
                patch("desktop.secrets.token_urlsafe", return_value="secret"),
                patch("desktop._wait_until_ready") as wait_ready,
                patch("desktop.subprocess.Popen", return_value=process) as popen,
            ):
                with desktop.local_model_server(server, model) as connection:
                    self.assertEqual(connection, ("http://127.0.0.1:12345/v1", "secret"))

            command = popen.call_args.args[0]
            self.assertIn("8192", command)
            self.assertIn("--parallel", command)
            self.assertIn("--offline", command)
            self.assertNotIn("secret", command)
            self.assertEqual(popen.call_args.kwargs["env"]["LLAMA_API_KEY"], "secret")
            wait_ready.assert_called_once()
            self.assertTrue(process.terminated)


if __name__ == "__main__":
    unittest.main()
