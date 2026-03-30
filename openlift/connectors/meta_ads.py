"""
Meta (Facebook) Ads connector for OpenLift.

Pulls geo-level campaign performance data (spend, conversions, impressions)
from Meta Ads and returns it in OpenLift's canonical long format.

Requires the ``facebook_business`` package:
    pip install facebook_business
"""

from typing import Dict, List, Optional
import pandas as pd
import logging
from datetime import datetime

from .base import DataConnector

logger = logging.getLogger(__name__)

try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.adobjects.adsinsights import AdsInsights
    HAS_META = True
except ImportError:
    HAS_META = False


class MetaAdsConnector(DataConnector):
    name = "meta_ads"
    description = "Import geo-level campaign data from Meta (Facebook) Ads"

    AVAILABLE_METRICS = [
        "spend",
        "impressions",
        "clicks",
        "conversions",
        "cpc",
        "cpm",
    ]

    # Conversion action types to look for in Meta's 'actions' list
    CONVERSION_ACTION_TYPES = [
        "offsite_conversion.fb_pixel_purchase",
        "offsite_conversion.fb_pixel_lead",
        "offsite_conversion.fb_pixel_complete_registration",
        "offsite_conversion",
        "lead",
        "purchase",
    ]

    def __init__(self):
        super().__init__()
        self._api = None
        self._account: Optional[AdAccount] = None
        self._ad_account_id: str = ""

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def authenticate(self, credentials: Dict) -> bool:
        """
        Authenticate with Meta Ads API.

        Required credentials dict keys:
        - access_token: str  (long-lived user or system user token)
        - ad_account_id: str  (e.g. "act_123456789")
        - app_id: str  (optional, default "0")
        - app_secret: str  (optional, default "")
        """
        if not HAS_META:
            logger.error("facebook_business package not installed. Run: pip install facebook_business")
            return False

        try:
            access_token = credentials["access_token"]
            self._ad_account_id = credentials["ad_account_id"]

            if not self._ad_account_id.startswith("act_"):
                self._ad_account_id = f"act_{self._ad_account_id}"

            app_id = credentials.get("app_id", "0")
            app_secret = credentials.get("app_secret", "")

            self._api = FacebookAdsApi.init(app_id, app_secret, access_token)
            self._account = AdAccount(self._ad_account_id)

            # Verify access by fetching account name
            account_info = self._account.api_get(fields=["name"])
            logger.info(f"Connected to Meta Ads: {account_info.get('name', self._ad_account_id)}")
            self._authenticated = True
            return True

        except KeyError as e:
            logger.error(f"Missing credential: {e}")
            return False
        except Exception as e:
            logger.error(f"Meta Ads auth failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def list_available_geos(self) -> List[str]:
        """Fetch distinct region names from recent campaign data."""
        if not self._authenticated:
            return []

        try:
            params = {
                "time_range": {"since": "2024-01-01", "until": datetime.now().strftime("%Y-%m-%d")},
                "breakdowns": ["region"],
                "level": "account",
                "limit": 500,
            }
            fields = ["region"]
            insights = self._account.get_insights(fields=fields, params=params)

            geos = set()
            for row in insights:
                region = row.get("region")
                if region and region != "Unknown":
                    geos.add(region)

            return sorted(geos)
        except Exception as e:
            logger.error(f"Error listing Meta geos: {e}")
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

        fields = [
            "date_start",
            "spend",
            "impressions",
            "clicks",
            "cpc",
            "cpm",
            "actions",
        ]
        params = {
            "time_range": {"since": start_date, "until": end_date},
            "time_increment": 1,  # Daily granularity
            "breakdowns": ["region"],
            "level": "account",
            "limit": 5000,
        }

        try:
            insights = self._account.get_insights(fields=fields, params=params)

            rows = []
            for row in insights:
                region = row.get("region", "Unknown")
                if region == "Unknown":
                    continue

                record = {
                    "date": row.get("date_start"),
                    geo_col: region,
                    "spend": float(row.get("spend", 0)),
                    "impressions": int(row.get("impressions", 0)),
                    "clicks": int(row.get("clicks", 0)),
                }

                # Extract conversions from the 'actions' list
                actions = row.get("actions", [])
                conversions = self._extract_conversions(actions)
                record["conversions"] = conversions

                # CPC / CPM
                record["cpc"] = float(row.get("cpc", 0))
                record["cpm"] = float(row.get("cpm", 0))

                rows.append(record)

            df = pd.DataFrame(rows)

            if df.empty:
                logger.warning("No data returned from Meta Ads for the given range.")
                return pd.DataFrame(columns=["date", geo_col, outcome_col])

            # Set primary outcome
            if outcome_col not in df.columns:
                if "conversions" in df.columns:
                    df[outcome_col] = df["conversions"]
                elif "spend" in df.columns:
                    df[outcome_col] = df["spend"]

            # Keep requested columns
            keep = ["date", geo_col, outcome_col]
            if metric_cols:
                keep += [c for c in metric_cols if c in df.columns]
            df = df[[c for c in keep if c in df.columns]]

            return self._validate_output(df, geo_col, outcome_col)

        except Exception as e:
            logger.error(f"Meta Ads fetch error: {e}")
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _extract_conversions(self, actions: list) -> float:
        """
        Extract total conversions from Meta's 'actions' array.

        Meta returns actions as a list of dicts: [{"action_type": "...", "value": "..."}]
        We sum values across known conversion action types.
        """
        total = 0.0
        if not actions:
            return total

        for action in actions:
            action_type = action.get("action_type", "")
            if action_type in self.CONVERSION_ACTION_TYPES:
                total += float(action.get("value", 0))

        return total
