"""
Agent Configuration Module

This module provides centralized configuration management for AI agents.
It loads configuration from YAML files and environment variables, with
environment variables taking precedence over file configuration.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import yaml

from util.configuration import PROJECT_ROOT
from util.logger_config import logger


@dataclass
class AgentConfig:
    """Configuration for AI agents."""
    
    provider: str = "gemini"  # "gemini" or "openai"
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.0
    api_key: str = ""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.provider not in ["gemini", "openai"]:
            raise ValueError(f"Invalid provider: {self.provider}. Must be 'gemini' or 'openai'")
        
        if not self.api_key:
            raise ValueError("API key is required. Set AI_AGENTIC_API_KEY environment variable or configure in agent_config.yaml")
        
        if self.temperature < 0.0 or self.temperature > 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {self.temperature}")


@lru_cache()
def _load_config_file() -> dict:
    """
    Load configuration from YAML file.
    
    Returns:
        dict: Configuration dictionary from file, or empty dict if file not found
    """
    config_file = PROJECT_ROOT / "config" / "agent_config.yaml"
    
    if not config_file.exists():
        logger.warning(f"Configuration file not found: {config_file}. Using defaults and environment variables.")
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_file}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration file {config_file}: {e}")
        return {}


def _get_env_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get value from environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def get_agent_config() -> AgentConfig:
    """
    Get agent configuration, loading from file and environment variables.
    Environment variables take precedence over file configuration.
    
    Returns:
        AgentConfig: Configuration object
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    # Load configuration from file
    file_config = _load_config_file()
    agent_config = file_config.get("agent", {})
    
    # Get values from file or defaults
    provider = agent_config.get("provider", "gemini")
    model = agent_config.get("model", "gemini-2.5-flash-lite")
    temperature = agent_config.get("temperature", 0.0)
    
    # Override with environment variables (if set)
    provider = _get_env_value("AI_AGENTIC_PROVIDER", provider)
    model = _get_env_value("AI_AGENTIC_MODEL", model)
    
    temp_str = _get_env_value("AI_AGENTIC_TEMPERATURE")
    if temp_str is not None:
        try:
            temperature = float(temp_str)
        except ValueError:
            logger.warning(f"Invalid temperature value in environment: {temp_str}. Using default: {temperature}")
    
    # API key ONLY from environment variables (for security)
    # Should never be in configuration files
    api_key = _get_env_value("AI_AGENTIC_API_KEY")
    
    # Create and return configuration object
    config = AgentConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        api_key=api_key or ""  # Empty string if not set (will be validated in __post_init__)
    )
    
    logger.info(f"Agent configuration loaded: provider={provider}, model={model}, temperature={temperature}")
    
    return config

