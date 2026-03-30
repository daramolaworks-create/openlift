"""
Abstract base class for all OpenLift data connectors.

Every connector must return data in OpenLift's canonical long format:
    date | geo | outcome [| optional metric columns]
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataConnector(ABC):
    """
    Base interface for pulling marketing data into OpenLift.

    Subclasses implement platform-specific auth and fetch logic,
    but always return a standardised pd.DataFrame.
    """

    name: str = "base"
    description: str = "Abstract connector"

    def __init__(self):
        self._authenticated = False
        self._credentials: Dict = {}

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    @abstractmethod
    def authenticate(self, credentials: Dict) -> bool:
        """
        Validate and store credentials.  Returns True on success.
        """
        ...

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    @abstractmethod
    def list_available_geos(self) -> List[str]:
        """Return human-readable geo names available in the data source."""
        ...

    @abstractmethod
    def list_available_metrics(self) -> List[str]:
        """Return metric column names the connector can provide."""
        ...

    # ------------------------------------------------------------------
    # Data fetch
    # ------------------------------------------------------------------
    @abstractmethod
    def fetch_data(
        self,
        start_date: str,
        end_date: str,
        geo_col: str = "geo",
        outcome_col: str = "outcome",
        metric_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Pull data from the source and return it in long format.

        The returned DataFrame **must** have at least these columns:
            - ``date``   (datetime64)
            - ``<geo_col>``   (str)
            - ``<outcome_col>``  (float)

        Parameters
        ----------
        start_date : str  — ISO date string  e.g. "2024-01-01"
        end_date   : str  — ISO date string  e.g. "2024-12-31"
        geo_col    : str  — name to give the geo column (default "geo")
        outcome_col: str  — name to give the primary outcome column
        metric_cols: list  — optional extra metrics to include

        Returns
        -------
        pd.DataFrame in OpenLift long format.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_output(self, df: pd.DataFrame, geo_col: str, outcome_col: str) -> pd.DataFrame:
        """Quick sanity check on the returned DataFrame."""
        required = {"date", geo_col, outcome_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Connector output missing columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])
        df[outcome_col] = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0)
        return df
