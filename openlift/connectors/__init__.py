from .base import DataConnector
from .google_sheets import GoogleSheetsConnector

# Optional imports — these require extra dependencies
try:
    from .google_ads import GoogleAdsConnector
except ImportError:
    GoogleAdsConnector = None

try:
    from .meta_ads import MetaAdsConnector
except ImportError:
    MetaAdsConnector = None

__all__ = [
    "DataConnector",
    "GoogleSheetsConnector",
    "GoogleAdsConnector",
    "MetaAdsConnector",
]
