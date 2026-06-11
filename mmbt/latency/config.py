from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LatencyConfig(BaseModel):
    """
    Lognormal latency model. feed_us/order_us/cancel_us are medians.
    jitter is the lognormal sigma 0.20 fits most co-lo setups.
    Sampling in LatencySimulator, not here. Lognormal fits real latencies well enough.
    """

    model_config = ConfigDict(frozen=True)

    feed_us:   float = Field(default=100.0, gt=0.0)
    order_us:  float = Field(default=500.0, gt=0.0)
    cancel_us: float = Field(default=300.0, gt=0.0)
    jitter:    float = Field(default=0.20, ge=0.0, le=2.0)

    @classmethod
    def from_yaml(cls, path: str) -> LatencyConfig:
        import yaml
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    def to_yaml(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
