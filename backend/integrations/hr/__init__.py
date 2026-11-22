"""Heart rate integration for DriveMate: simulation and Suunto .FIT replay."""

from backend.integrations.hr.fit import parse_fit_heart_rate

__all__ = ["parse_fit_heart_rate"]
