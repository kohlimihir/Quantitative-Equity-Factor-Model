"""
Configuration Management for Equity Factor Model

Handles loading, validation, and management of configuration files
with support for profiles and inheritance.
"""

import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import warnings


class ConfigManager:
    """Configuration management system for equity factor model."""
    
    def __init__(self, config_dir: str = "configs", verbose: bool = True):
        self.config_dir = config_dir
        self.verbose = verbose
        self.config = None
        self.config_path = None
        self.load_timestamp = None
        
    def load_config(self, profile: str = "baseline") -> Dict[str, Any]:
        """Load configuration from profile."""
        config_path = os.path.join(self.config_dir, f"{profile}.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if self.verbose:
            print(f"\nLoading configuration: {profile}")
            print(f"  Path: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.config = config
        self.config_path = config_path
        self.load_timestamp = datetime.now()
        
        # Validate configuration
        self._validate_config(config)
        
        if self.verbose:
            print(f"  ✓ Configuration loaded successfully")
            print(f"  Version: {config.get('version', 'unknown')}")
            print(f"  Description: {config.get('description', 'N/A')}")
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and values."""
        required_sections = ["data", "features", "models", "validation", 
                           "turnover", "portfolio", "diagnostics"]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        data_config = config["data"]
        if data_config.get("min_train_months", 0) < 12:
            warnings.warn("min_train_months < 12 may lead to unstable models")
        
        if data_config.get("fundamental_lag_days", 0) < 45:
            warnings.warn("fundamental_lag_days < 45 may cause data leakage")
        
        models_config = config["models"]
        if not any(models_config.get(m, {}).get("enabled", False) 
                  for m in ["ridge", "lightgbm", "ensemble", "sector_specific"]):
            raise ValueError("At least one model must be enabled")
        
        turnover_config = config["turnover"]
        if turnover_config.get("target_turnover", 1.0) > 0.5:
            warnings.warn("target_turnover > 50% may generate excessive costs")
        
        portfolio_config = config["portfolio"]
        if portfolio_config.get("top_n_per_sector", 0) < 1:
            raise ValueError("top_n_per_sector must be >= 1")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'models.ridge.alpha')."""
        if self.config is None:
            raise ValueError("No config loaded. Call load_config() first.")
        
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set config value using dot notation."""
        if self.config is None:
            raise ValueError("No config loaded. Call load_config() first.")
        
        keys = key_path.split(".")
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to file.
        
        Parameters
        ----------
        output_path : str
            Path to save configuration file
        """
        if self.config is None:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        if self.verbose:
            print(f"  ✓ Configuration saved to: {output_path}")
    
    def list_profiles(self) -> List[str]:
        """
        List available configuration profiles.
        
        Returns
        -------
        list of str
            Available profile names
        """
        if not os.path.exists(self.config_dir):
            return []
        
        profiles = []
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json"):
                profiles.append(filename.replace(".json", ""))
        
        return sorted(profiles)
    
    def compare_profiles(self, profile1: str, profile2: str) -> Dict[str, Any]:
        """
        Compare two configuration profiles.
        
        Parameters
        ----------
        profile1 : str
            First profile name
        profile2 : str
            Second profile name
        
        Returns
        -------
        dict
            Dictionary containing differences between profiles
        """
        # Load both profiles
        config1 = self._load_config_file(profile1)
        config2 = self._load_config_file(profile2)
        
        # Find differences
        differences = self._find_differences(config1, config2)
        
        if self.verbose:
            print(f"\nComparing profiles: {profile1} vs {profile2}")
            print(f"  Found {len(differences)} differences")
            
            for diff in differences[:10]:  # Show first 10
                print(f"    {diff['path']}: {diff['value1']} → {diff['value2']}")
            
            if len(differences) > 10:
                print(f"    ... and {len(differences) - 10} more")
        
        return {
            "profile1": profile1,
            "profile2": profile2,
            "differences": differences,
            "n_differences": len(differences)
        }
    
    def _load_config_file(self, profile: str) -> Dict[str, Any]:
        """Load configuration file without setting as current config."""
        config_path = os.path.join(self.config_dir, f"{profile}.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _find_differences(self, config1: Dict[str, Any], config2: Dict[str, Any],
                         path: str = "") -> List[Dict[str, Any]]:
        """Recursively find differences between two configurations."""
        differences = []
        
        # Check keys in config1
        for key in config1:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config2:
                differences.append({
                    "path": current_path,
                    "type": "missing_in_config2",
                    "value1": config1[key],
                    "value2": None
                })
            elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
                # Recursively compare nested dicts
                differences.extend(
                    self._find_differences(config1[key], config2[key], current_path)
                )
            elif config1[key] != config2[key]:
                differences.append({
                    "path": current_path,
                    "type": "value_difference",
                    "value1": config1[key],
                    "value2": config2[key]
                })
        
        # Check keys only in config2
        for key in config2:
            if key not in config1:
                current_path = f"{path}.{key}" if path else key
                differences.append({
                    "path": current_path,
                    "type": "missing_in_config1",
                    "value1": None,
                    "value2": config2[key]
                })
        
        return differences
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration.
        
        Returns
        -------
        dict
            Configuration summary
        """
        if self.config is None:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        summary = {
            "profile": self.config.get("profile", "unknown"),
            "version": self.config.get("version", "unknown"),
            "description": self.config.get("description", "N/A"),
            "loaded_at": self.load_timestamp.isoformat() if self.load_timestamp else None,
            "config_path": self.config_path,
            "enabled_models": [],
            "enabled_diagnostics": [],
            "turnover_target": self.get("turnover.target_turnover"),
            "min_train_months": self.get("data.min_train_months"),
            "fundamental_lag_days": self.get("data.fundamental_lag_days"),
        }
        
        # Collect enabled models
        models_config = self.config.get("models", {})
        for model_name, model_config in models_config.items():
            if isinstance(model_config, dict) and model_config.get("enabled", False):
                summary["enabled_models"].append(model_name)
        
        # Collect enabled diagnostics
        diagnostics_config = self.config.get("diagnostics", {})
        for diag_name, diag_enabled in diagnostics_config.items():
            if diag_enabled is True:
                summary["enabled_diagnostics"].append(diag_name)
        
        return summary
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        summary = self.get_config_summary()
        
        print("\n" + "="*70)
        print("  CONFIGURATION SUMMARY")
        print("="*70)
        print(f"  Profile: {summary['profile']}")
        print(f"  Version: {summary['version']}")
        print(f"  Description: {summary['description']}")
        print(f"  Loaded at: {summary['loaded_at']}")
        print(f"\n  Data Configuration:")
        print(f"    Min train months: {summary['min_train_months']}")
        print(f"    Fundamental lag: {summary['fundamental_lag_days']} days")
        print(f"\n  Enabled Models: {', '.join(summary['enabled_models'])}")
        print(f"  Enabled Diagnostics: {', '.join(summary['enabled_diagnostics'])}")
        if summary['turnover_target'] is not None:
            print(f"  Turnover Target: {summary['turnover_target']:.1%}")
        print("="*70)


def load_config(profile: str = "baseline", verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Parameters
    ----------
    profile : str
        Configuration profile name (default: "baseline")
    verbose : bool
        Print progress messages (default: True)
    
    Returns
    -------
    dict
        Configuration dictionary
    
    Examples
    --------
    >>> config = load_config("baseline")
    >>> config = load_config("low-turnover")
    """
    manager = ConfigManager(verbose=verbose)
    return manager.load_config(profile)


if __name__ == "__main__":
    # Test configuration manager
    print("="*70)
    print("  CONFIGURATION MANAGER TEST")
    print("="*70)
    
    manager = ConfigManager(verbose=True)
    
    # List available profiles
    print("\nAvailable profiles:")
    profiles = manager.list_profiles()
    for profile in profiles:
        print(f"  - {profile}")
    
    # Load baseline configuration
    print("\n" + "="*70)
    print("  LOADING BASELINE CONFIGURATION")
    print("="*70)
    config = manager.load_config("baseline")
    manager.print_summary()
    
    # Test get/set
    print("\n" + "="*70)
    print("  TESTING GET/SET")
    print("="*70)
    print(f"  Ridge alpha: {manager.get('models.ridge.alpha')}")
    print(f"  EWM alpha: {manager.get('turnover.ewm_smoothing.alpha')}")
    print(f"  Top N per sector: {manager.get('portfolio.top_n_per_sector')}")
    
    # Compare profiles
    if len(profiles) >= 2:
        print("\n" + "="*70)
        print("  COMPARING PROFILES")
        print("="*70)
        comparison = manager.compare_profiles(profiles[0], profiles[1])
    
    print("\n" + "="*70)
    print("  CONFIGURATION MANAGER TEST COMPLETE")
    print("="*70)
