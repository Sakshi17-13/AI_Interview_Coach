"""Offline startup tests for the watsonx environment configuration gate."""

from __future__ import annotations

import unittest

from watsonx_configuration import (
    CONFIGURATION_REQUIRED_MESSAGE,
    configuration_required_ui_message,
    initialize_watsonx_from_environment,
)


class WatsonxConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_calls: list[dict[str, object]] = []
        self.configured: list[object] = []

    def configure(self, model: object) -> None:
        self.configured.append(model)

    def credentials(self, **kwargs: object) -> dict[str, object]:
        return kwargs

    def parameters(self) -> dict[str, object]:
        return {"max_tokens": 100}

    def model(self, **kwargs: object) -> object:
        self.model_calls.append(kwargs)
        return object()

    def initialize(self, environ: dict[str, str], **overrides):
        dependencies = {
            "environ": environ,
            "credentials_factory": self.credentials,
            "parameters_factory": self.parameters,
            "model_factory": self.model,
            "configure_client": self.configure,
        }
        dependencies.update(overrides)
        return initialize_watsonx_from_environment(**dependencies)

    def test_startup_without_credentials_requires_configuration_and_creates_no_model(self) -> None:
        configuration = self.initialize({})

        self.assertEqual(configuration.status, "configuration_required")
        self.assertFalse(configuration.is_ready)
        self.assertIsNone(configuration.model)
        self.assertEqual(self.model_calls, [])
        self.assertEqual(self.configured, [])
        self.assertIn("WATSONX_API_KEY", configuration.message)
        self.assertEqual(
            configuration_required_ui_message(configuration),
            f"❌ {CONFIGURATION_REQUIRED_MESSAGE}",
        )

    def test_every_required_environment_value_is_required(self) -> None:
        for environ in (
            {"WATSONX_PROJECT_ID": "project", "WATSONX_URL": "https://example.test"},
            {"WATSONX_API_KEY": "secret", "WATSONX_URL": "https://example.test"},
            {"WATSONX_API_KEY": "secret", "WATSONX_PROJECT_ID": "project"},
        ):
            configuration = self.initialize(environ)
            self.assertEqual(configuration.status, "configuration_required")
        self.assertEqual(self.model_calls, [])

    def test_valid_environment_creates_and_configures_exactly_one_existing_model(self) -> None:
        configuration = self.initialize(
            {
                "WATSONX_API_KEY": "test-key",
                "WATSONX_PROJECT_ID": "project-id",
                "WATSONX_URL": "https://example.test",
            }
        )

        self.assertTrue(configuration.is_ready)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(len(self.configured), 1)
        self.assertEqual(self.model_calls[0]["model_id"], "meta-llama/llama-3-3-70b-instruct")
        self.assertEqual(self.model_calls[0]["project_id"], "project-id")
        self.assertEqual(self.model_calls[0]["credentials"], {"url": "https://example.test", "api_key": "test-key"})

    def test_initialization_failure_is_sanitized_and_never_exposes_api_key(self) -> None:
        api_key = "super-secret-key"
        configuration = self.initialize(
            {
                "WATSONX_API_KEY": api_key,
                "WATSONX_PROJECT_ID": "project",
                "WATSONX_URL": "https://example.test",
            },
            model_factory=lambda **_: (_ for _ in ()).throw(RuntimeError(f"Provider rejected {api_key}")),
        )

        self.assertEqual(configuration.status, "configuration_required")
        self.assertEqual(configuration.message, CONFIGURATION_REQUIRED_MESSAGE)
        self.assertNotIn(api_key, configuration.message)
        self.assertNotIn(api_key, repr(configuration))
        self.assertEqual(self.configured, [])


if __name__ == "__main__":
    unittest.main()
