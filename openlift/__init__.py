__all__ = ["app", "run_geo_lift", "run_geo_lift_df"]
__version__ = "3.0.0"


def __getattr__(name):
    if name == "app":
        from .cli.main import app

        return app
    if name in {"run_geo_lift", "run_geo_lift_df"}:
        from .core.pipeline import run_geo_lift, run_geo_lift_df

        return {"run_geo_lift": run_geo_lift, "run_geo_lift_df": run_geo_lift_df}[name]
    raise AttributeError(f"module 'openlift' has no attribute {name!r}")
