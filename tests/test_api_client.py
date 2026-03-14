from __future__ import annotations

import io
import logging
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from autoquant_cli.api_client import get_openapi_json, normalize_api_path, post_json

logger = logging.getLogger(__name__)


class ApiClientTest(unittest.TestCase):
    def test_normalize_api_path(self) -> None:
        self.assertEqual(normalize_api_path("/experiment/get"), "/api/v1/experiment/get")
        self.assertEqual(normalize_api_path("experiment/get"), "/api/v1/experiment/get")
        self.assertEqual(normalize_api_path("/api/v1/experiment/get"), "/api/v1/experiment/get")

    def test_post_json_parses_successful_response(self) -> None:
        response = MagicMock()
        response.read.return_value = b'{"ok": true}'
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        with (
            patch("autoquant_cli.api_client.get_backend_base_url", return_value="https://example.com"),
            patch("autoquant_cli.api_client.get_api_key", return_value="api-key"),
            patch("autoquant_cli.api_client.urlopen", return_value=response),
        ):
            logger.info("Testing successful backend response parsing")
            payload = post_json("/experiment/get", {"run_id": "run-1"})
        self.assertEqual(payload, {"ok": True})

    def test_post_json_surfaces_backend_detail(self) -> None:
        error = HTTPError(
            url="https://example.com/api/v1/experiment/get",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"Run not found"}'),
        )
        with (
            patch("autoquant_cli.api_client.get_backend_base_url", return_value="https://example.com"),
            patch("autoquant_cli.api_client.get_api_key", return_value="api-key"),
            patch("autoquant_cli.api_client.urlopen", side_effect=error),
        ):
            with self.assertRaises(RuntimeError) as context:
                post_json("/experiment/get", {"run_id": "run-1"})
        self.assertIn("Run not found", str(context.exception))

    def test_get_openapi_json_parses_successful_response(self) -> None:
        response = MagicMock()
        response.read.return_value = b'{"openapi": "3.1.0", "info": {"title": "AutoQuant API"}}'
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        with (
            patch("autoquant_cli.api_client.get_backend_base_url", return_value="https://example.com"),
            patch("autoquant_cli.api_client.urlopen", return_value=response),
        ):
            payload = get_openapi_json()
        self.assertEqual(payload, {"openapi": "3.1.0", "info": {"title": "AutoQuant API"}})


if __name__ == "__main__":
    unittest.main()
