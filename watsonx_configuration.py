"""Safe, environment-backed initialization of the application's one watsonx model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Literal


CONFIGURATION_REQUIRED_MESSAGE = (
    "Watsonx configuration is required. Set WATSONX_API_KEY, WATSONX_PROJECT_ID, "
    "and WATSONX_URL, then restart the application."
)


@dataclass(frozen=True)
class WatsonxConfiguration:
    """Application configuration state without retaining or exposing credentials."""

    status: Literal["ready", "configuration_required"]
    message: str
    model: Any | None = None

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and self.model is not None


def configuration_required_ui_message(configuration: WatsonxConfiguration) -> str | None:
    """Return the safe UI error used to block AI actions while unconfigured."""

    return None if configuration.is_ready else f"❌ {CONFIGURATION_REQUIRED_MESSAGE}"


def initialize_watsonx_from_environment(
    *,
    environ: Mapping[str, str] | None = None,
    credentials_factory: Callable[..., Any],
    parameters_factory: Callable[[], Any],
    model_factory: Callable[..., Any],
    configure_client: Callable[[Any], Any],
) -> WatsonxConfiguration:
    """Create/configure exactly one model only when all required values are present.

    The returned error is deliberately constant so an API key or provider exception
    can never be echoed into UI text or logs by this configuration path.
    """

    values = environ if environ is not None else os.environ
    api_key = values.get("WATSONX_API_KEY", "").strip()
    project_id = values.get("WATSONX_PROJECT_ID", "").strip()
    url = values.get("WATSONX_URL", "").strip()
    if not api_key or not project_id or not url:
        return WatsonxConfiguration("configuration_required", CONFIGURATION_REQUIRED_MESSAGE)
    try:
        credentials = credentials_factory(url=url, api_key=api_key)
        model = model_factory(
            model_id="meta-llama/llama-3-3-70b-instruct",
            credentials=credentials,
            project_id=project_id,
            params=parameters_factory(),
        )
        configure_client(model)
    except Exception:
        return WatsonxConfiguration("configuration_required", CONFIGURATION_REQUIRED_MESSAGE)
    return WatsonxConfiguration("ready", "Watsonx AI is configured.", model)
