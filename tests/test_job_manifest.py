import json
import os
import tempfile
import unittest

from api.services.job_manifest import read_job_manifest, write_job_manifest


class JobManifestTest(unittest.TestCase):
    def test_write_manifest_redacts_secrets_and_lists_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = os.path.join(tmpdir, "clip_1.mp4")
            with open(artifact_path, "wb") as handle:
                handle.write(b"video")

            job = {
                "status": "completed",
                "created_at": 123,
                "output_dir": tmpdir,
                "source_kind": "file",
                "source_label": "input.mp4",
                "aspect_ratio": "9:16",
                "clip_count_target": 3,
                "video_type": "Topic-clips",
                "ownership_attested": True,
                "cmd": ["python", "main.py", "-i", "input.mp4"],
                "env": {
                    "GEMINI_API_KEY": "secret",
                    "OPENSHORTS_FFMPEG_CRF": "18",
                    "PATH": "/usr/bin",
                },
            }

            manifest_path = write_job_manifest("job-1", job, "completed", returncode=0)
            self.assertTrue(os.path.exists(manifest_path))

            manifest = read_job_manifest(tmpdir)
            self.assertEqual(manifest["job_id"], "job-1")
            self.assertEqual(manifest["returncode"], 0)
            self.assertEqual(manifest["runtime_env"]["GEMINI_API_KEY"], "***")
            self.assertEqual(manifest["runtime_env"]["OPENSHORTS_FFMPEG_CRF"], "18")
            self.assertNotIn("PATH", manifest["runtime_env"])
            self.assertEqual(manifest["artifacts"][0]["path"], "clip_1.mp4")

            with open(manifest_path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            self.assertEqual(raw["events"][-1]["event"], "completed")


if __name__ == "__main__":
    unittest.main()
