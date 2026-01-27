import yaml
from datetime import date
from typing import List
from pydantic import BaseModel, field_validator, model_validator

class Period(BaseModel):
    start_date: date
    end_date: date
    
    @model_validator(mode='after')
    def check_dates(self) -> 'Period':
        if self.end_date < self.start_date:
            raise ValueError(f"end_date {self.end_date} cannot be before start_date {self.start_date}")
        return self

class DataConfig(BaseModel):
    path: str
    date_col: str
    geo_col: str
    outcome_col: str

class ExperimentConfig(BaseModel):
    name: str
    test_geo: str
    control_geos: List[str]
    pre_period: Period
    post_period: Period
    
    @field_validator('control_geos')
    @classmethod
    def check_control_geos_count(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("Must specify at least 2 control_geos")
        return v
    
    @model_validator(mode='after')
    def check_pre_post_continuity(self) -> 'ExperimentConfig':
        pre_len = (self.pre_period.end_date - self.pre_period.start_date).days + 1
        if pre_len < 14:
            raise ValueError(f"Pre-period must be at least 14 days. Current: {pre_len} days.")
            
        if self.post_period.start_date <= self.pre_period.end_date:
             raise ValueError("Post-period start_date must be after Pre-period end_date.")
             
        return self

class FullConfig(BaseModel):
    experiment: ExperimentConfig
    data: DataConfig

def load_experiment_config(path: str) -> FullConfig:
    with open(path, 'r') as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {e}")
            
    return FullConfig(**raw)
