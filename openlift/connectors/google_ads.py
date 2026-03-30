"""
Google Ads connector for OpenLift.

Pulls geo-level campaign performance data (spend, conversions, clicks)
from Google Ads and returns it in OpenLift's canonical long format.

Requires the ``google-ads`` package:
    pip install google-ads
"""

from typing import Dict, List, Optional
import pandas as pd
import logging
from datetime import datetime

from .base import DataConnector

logger = logging.getLogger(__name__)

try:
    from google.ads.googleads.client import GoogleAdsClient
    HAS_GOOGLE_ADS = True
except ImportError:
    HAS_GOOGLE_ADS = False


# Google Ads geo target constants for common types
GEO_TARGET_TYPES = {
    "city": "segments.geo_target_city",
    "region": "segments.geo_target_region",
    "country": "segments.geo_target_country",
}


class GoogleAdsConnector(DataConnector):
    name = "google_ads"
    description = "Import geo-level campaign data from Google Ads"

    AVAILABLE_METRICS = [
        "spend",
        "conversions",
        "clicks",
        "impressions",
        "conversion_value",
    ]

    # Map friendly names → Google Ads GAQL field names
    _METRIC_MAP = {
        "spend": "metrics.cost_micros",
        "conversions": "metrics.conversions",
        "clicks": "metrics.clicks",
        "impressions": "metrics.impressions",
        "conversion_value": "metrics.conversions_value",
    }

    def __init__(self):
        super().__init__()
        self._client = None
        self._customer_id: str = ""
        self._geo_level: str = "region"

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def authenticate(self, credentials: Dict) -> bool:
        """
        Authenticate with Google Ads API.

        Required credentials dict keys:
        - developer_token: str
        - client_id: str
        - client_secret: str
        - refresh_token: str
        - customer_id: str  (10-digit, no dashes)
        - login_customer_id: str  (optional, for MCC accounts)
        - geo_level: str  (optional, "city" | "region" | "country", default "region")
        """
        if not HAS_GOOGLE_ADS:
            logger.error("google-ads package not installed. Run: pip install google-ads")
            return False

        try:
            config = {
                "developer_token": credentials["developer_token"],
                "client_id": credentials["client_id"],
                "client_secret": credentials["client_secret"],
                "refresh_token": credentials["refresh_token"],
                "use_proto_plus": True,
            }
            if "login_customer_id" in credentials:
                config["login_customer_id"] = credentials["login_customer_id"]

            self._client = GoogleAdsClient.load_from_dict(config)
            self._customer_id = credentials["customer_id"].replace("-", "")
            self._geo_level = credentials.get("geo_level", "region")
            self._authenticated = True
            logger.info(f"Connected to Google Ads (customer {self._customer_id})")
            return True

        except KeyError as e:
            logger.error(f"Missing credential: {e}")
            return False
        except Exception as e:
            logger.error(f"Google Ads auth failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def list_available_geos(self) -> List[str]:
        """Fetch distinct geo names from recent campaign data."""
        if not self._authenticated:
            return []

        geo_field = GEO_TARGET_TYPES.get(self._geo_level, GEO_TARGET_TYPES["region"])
        query = f"""
            SELECT {geo_field}
            FROM geographic_view
            WHERE segments.date DURING LAST_30_DAYS
        """
        try:
            service = self._client.get_service("GoogleAdsService")
            response = service.search(customer_id=self._customer_id, query=query)
            geos = set()
            for row in response:
                geo_resource = getattr(row.segments, self._geo_level_attr())
                if geo_resource:
                    # Resolve resource name to readable name
                    geos.add(self._resolve_geo_name(geo_resource))
            return sorted(geos)
        except Exception as e:
            logger.error(f"Error listing geos: {e}")
            return []

    def list_available_metrics(self) -> List[str]:
        return self.AVAILABLE_METRICS.copy()

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

        # Build the set of metrics to pull
        if metric_cols is None:
            metric_cols = ["spend", "conversions"]

        # Always include the outcome metric
        gaql_metrics = []
        for m in metric_cols:
            if m in self._METRIC_MAP:
                gaql_metrics.append(self._METRIC_MAP[m])

        if not gaql_metrics:
            gaql_metrics = ["metrics.cost_micros", "metrics.conversions"]

        geo_field = GEO_TARGET_TYPES.get(self._geo_level, GEO_TARGET_TYPES["region"])
        metrics_str = ", ".join(gaql_metrics)

        query = f"""
            SELECT
                segments.date,
                {geo_field},
                {metrics_str}
            FROM geographic_view
            WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
        """

        try:
            service = self._client.get_service("GoogleAdsService")
            response = service.search(customer_id=self._customer_id, query=query)

            rows = []
            for row in response:
                geo_resource = getattr(row.segments, self._geo_level_attr())
                geo_name = self._resolve_geo_name(geo_resource) if geo_resource else "Unknown"

                record = {
                    "date": row.segments.date,
                    geo_col: geo_name,
                }

                # Map metrics back to friendly names
                for friendly, gaql in self._METRIC_MAP.items():
                    if gaql in gaql_metrics:
                        val = self._extract_metric(row, gaql)
                        record[friendly] = val

                rows.append(record)

            df = pd.DataFrame(rows)

            if df.empty:
                logger.warning("No data returned from Google Ads for the given range.")
                return pd.DataFrame(columns=["date", geo_col, outcome_col])

            # Convert spend from micros to actual currency
            if "spend" in df.columns:
                df["spend"] = df["spend"] / 1_000_000

            # Set the primary outcome column
            if outcome_col not in df.columns and "conversions" in df.columns:
                df[outcome_col] = df["conversions"]
            elif outcome_col not in df.columns and "spend" in df.columns:
                df[outcome_col] = df["spend"]

            return self._validate_output(df, geo_col, outcome_col)

        except Exception as e:
            logger.error(f"Google Ads fetch error: {e}")
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _geo_level_attr(self) -> str:
        """Return the segments attribute name for the chosen geo level."""
        return f"geo_target_{self._geo_level}"

    def _resolve_geo_name(self, resource_name: str) -> str:
        """
        Convert a Google Ads geo target resource name to a human-readable label.
        Resource name format: 'geoTargetConstants/12345'
        """
        try:
            service = self._client.get_service("GeoTargetConstantService")
            geo_constant = service.get_geo_target_constant(resource_name=resource_name)
            return geo_constant.name
        except Exception:
            # Fallback: return the raw resource name
            return resource_name.split("/")[-1] if "/" in resource_name else resource_name

    def _extract_metric(self, row, gaql_field: str):
        """Safely extract a metric value from a Google Ads row."""
        try:
            parts = gaql_field.split(".")
            obj = row
            for part in parts:
                obj = getattr(obj, part)
            return float(obj)
        except Exception:
            return 0.0
