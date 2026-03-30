"""
Google Sheets connector for OpenLift.

Pulls data from a Google Spreadsheet and returns it in OpenLift's
canonical long format.  Uses ``gspread`` with a service-account JSON
key or an API key.
"""

from typing import Dict, List, Optional
import pandas as pd
import logging

from .base import DataConnector

logger = logging.getLogger(__name__)

try:
    import gspread
    from gspread.exceptions import SpreadsheetNotFound, APIError
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False


class GoogleSheetsConnector(DataConnector):
    name = "google_sheets"
    description = "Import data from a Google Spreadsheet"

    def __init__(self):
        super().__init__()
        self._client = None
        self._sheet = None
        self._worksheet = None
        self._column_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def authenticate(self, credentials: Dict) -> bool:
        """
        Authenticate with Google Sheets.

        Accepted credential formats:
        1. ``{"service_account_json": "/path/to/creds.json"}``
        2. ``{"api_key": "AIza..."}``  (limited to public sheets)
        3. ``{"service_account_info": {dict of SA JSON contents}}``

        Also requires:
        - ``sheet_url`` or ``sheet_key``:  identifier of the spreadsheet
        - ``worksheet`` (optional): sheet tab name, defaults to first sheet
        """
        if not HAS_GSPREAD:
            logger.error("gspread is not installed.  Run: pip install gspread")
            return False

        try:
            # --- Build gspread client ---
            if "service_account_json" in credentials:
                self._client = gspread.service_account(filename=credentials["service_account_json"])
            elif "service_account_info" in credentials:
                self._client = gspread.service_account_from_dict(credentials["service_account_info"])
            elif "api_key" in credentials:
                self._client = gspread.api_key(credentials["api_key"])
            else:
                logger.error("No valid auth method provided.")
                return False

            # --- Open the spreadsheet ---
            sheet_url = credentials.get("sheet_url")
            sheet_key = credentials.get("sheet_key")

            if sheet_url:
                self._sheet = self._client.open_by_url(sheet_url)
            elif sheet_key:
                self._sheet = self._client.open_by_key(sheet_key)
            else:
                logger.error("Provide 'sheet_url' or 'sheet_key'.")
                return False

            worksheet_name = credentials.get("worksheet")
            if worksheet_name:
                self._worksheet = self._sheet.worksheet(worksheet_name)
            else:
                self._worksheet = self._sheet.sheet1

            # Store column mapping if user specified
            self._column_map = {
                "date_col": credentials.get("date_col", "date"),
                "geo_col": credentials.get("geo_col", "geo"),
                "outcome_col": credentials.get("outcome_col", "outcome"),
            }

            self._authenticated = True
            logger.info(f"Connected to Google Sheet: {self._sheet.title}")
            return True

        except SpreadsheetNotFound:
            logger.error("Spreadsheet not found. Check URL/key and sharing permissions.")
            return False
        except APIError as e:
            logger.error(f"Google API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Google Sheets auth failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def list_available_geos(self) -> List[str]:
        if not self._authenticated or self._worksheet is None:
            return []
        df = self._get_raw_df()
        geo_col = self._column_map.get("geo_col", "geo")
        if geo_col in df.columns:
            return sorted(df[geo_col].dropna().unique().tolist())
        return []

    def list_available_metrics(self) -> List[str]:
        if not self._authenticated or self._worksheet is None:
            return []
        df = self._get_raw_df()
        # Return all numeric-looking columns that aren't date/geo
        geo_col = self._column_map.get("geo_col", "geo")
        date_col = self._column_map.get("date_col", "date")
        return [c for c in df.columns if c not in [date_col, geo_col]]

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    def fetch_data(
        self,
        start_date: str,
        end_date: str,
        geo_col: str = "geo",
        outcome_col: str = "outcome",
        metric_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if not self._authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        df = self._get_raw_df()

        # Rename columns from sheet names to OpenLift standard names
        src_date = self._column_map.get("date_col", "date")
        src_geo = self._column_map.get("geo_col", "geo")
        src_outcome = self._column_map.get("outcome_col", "outcome")

        rename_map = {}
        if src_date != "date":
            rename_map[src_date] = "date"
        if src_geo != geo_col:
            rename_map[src_geo] = geo_col
        if src_outcome != outcome_col:
            rename_map[src_outcome] = outcome_col

        if rename_map:
            df = df.rename(columns=rename_map)

        # Parse dates and filter range
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
        df = df.loc[mask].copy()

        # Keep only relevant columns
        keep_cols = ["date", geo_col, outcome_col]
        if metric_cols:
            keep_cols += [c for c in metric_cols if c in df.columns]
        df = df[[c for c in keep_cols if c in df.columns]]

        return self._validate_output(df, geo_col, outcome_col)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _get_raw_df(self) -> pd.DataFrame:
        """Download the worksheet contents as a DataFrame."""
        records = self._worksheet.get_all_records()
        return pd.DataFrame(records)
