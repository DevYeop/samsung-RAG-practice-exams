import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from download_utils import download_if_missing


class DownloadHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_GET(self):
        type(self).request_count += 1
        if self.path == "/document.pdf":
            body = b"%PDF-test-content"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


class DownloadIfMissingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DownloadHandler.request_count = 0
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), DownloadHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_existing_file_is_reused_without_network_request(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "document.pdf"
            destination.write_bytes(b"existing")
            before = DownloadHandler.request_count

            result = download_if_missing(f"{self.base_url}/document.pdf", destination)

            self.assertEqual(destination, result)
            self.assertEqual(b"existing", destination.read_bytes())
            self.assertEqual(before, DownloadHandler.request_count)

    def test_successful_download_is_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "document.pdf"

            download_if_missing(f"{self.base_url}/document.pdf", destination)

            self.assertEqual(b"%PDF-test-content", destination.read_bytes())

    def test_failed_download_does_not_leave_broken_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "document.pdf"

            with self.assertRaises(requests.HTTPError):
                download_if_missing(f"{self.base_url}/missing.pdf", destination)

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".pdf.part").exists())


if __name__ == "__main__":
    unittest.main()
