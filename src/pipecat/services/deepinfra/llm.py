#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DeepInfra LLM service implementation using OpenAI-compatible interface."""

from dataclasses import dataclass

from loguru import logger

from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService


@dataclass
class DeepInfraLLMSettings(BaseOpenAILLMService.Settings):
    """Settings for DeepInfraLLMService."""

    pass


class DeepInfraLLMService(OpenAILLMService):
    """A service for interacting with DeepInfra using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to DeepInfra's API endpoint while
    maintaining compatibility with OpenAI's interface and functionality.
    """

    supports_developer_role = False

    Settings = DeepInfraLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        model: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the DeepInfra LLM service.

        Args:
            api_key: The API key for accessing DeepInfra.
            model: The model identifier to use. Defaults to "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".
            base_url: The base URL for DeepInfra API. Defaults to "https://api.deepinfra.com/v1/openai".
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to OpenAILLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for DeepInfra API endpoint.

        Args:
            api_key: API key for authentication. If None, uses instance default.
            base_url: Base URL for the API. If None, uses instance default.
            **kwargs: Additional arguments passed to the client constructor.

        Returns:
            Configured OpenAI client instance for DeepInfra API.
        """
        logger.debug(f"Creating DeepInfra client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)
