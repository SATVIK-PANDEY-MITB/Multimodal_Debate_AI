import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv_tools
import main


class CvTests(unittest.TestCase):
    def _make_temp_image_file(self):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.write(b"not a real image, but good enough for path existence in tests")
        temp_file.close()
        return Path(temp_file.name)

    @patch("cv_tools.call_vision_llm", return_value="classified as screenshot")
    def test_run_cv_mode_classify_dispatches(self, _mock_call):
        image_path = self._make_temp_image_file()
        try:
            result = cv_tools.run_cv_mode(str(image_path), mode="classify")
        finally:
            image_path.unlink(missing_ok=True)

        self.assertEqual(result, "classified as screenshot")

    @patch("cv_tools.call_vision_llm", return_value="extracted text")
    def test_run_cv_mode_ocr_dispatches(self, _mock_call):
        image_path = self._make_temp_image_file()
        try:
            result = cv_tools.run_cv_mode(str(image_path), mode="ocr")
        finally:
            image_path.unlink(missing_ok=True)

        self.assertEqual(result, "extracted text")

    def test_run_cv_mode_rejects_invalid_mode(self):
        image_path = self._make_temp_image_file()
        try:
            with self.assertRaises(ValueError):
                cv_tools.run_cv_mode(str(image_path), mode="invalid")
        finally:
            image_path.unlink(missing_ok=True)

    @patch("main.run_cv_mode", return_value="cv output")
    def test_cv_cli_prints_result(self, _mock_run_cv_mode):
        image_path = self._make_temp_image_file()
        try:
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                main._run_cv_cli(["--image", str(image_path), "--mode", "describe"])
        finally:
            image_path.unlink(missing_ok=True)

        self.assertEqual(output_buffer.getvalue().strip(), "cv output")


if __name__ == "__main__":
    unittest.main()
