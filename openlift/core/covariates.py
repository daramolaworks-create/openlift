"""
Automated external covariates for OpenLift experiments.

Provides holiday indicators and historical weather data as additional
covariates for the Bayesian model, helping control for external shocks
that would otherwise pollute the treatment effect estimate.

- Holidays: Uses the ``holidays`` Python package (100+ countries).
- Weather: Uses the Open-Meteo Historical Weather API (free, no key).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Dict, Tuple
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Built-in geo → lat/lon lookup for common cities/regions
# ------------------------------------------------------------------
GEO_COORDINATES: Dict[str, Tuple[float, float]] = {
    # Nigeria
    "lagos": (6.45, 3.40),
    "abuja": (9.06, 7.49),
    "ibadan": (7.38, 3.94),
    "kano": (12.00, 8.52),
    "port harcourt": (4.78, 7.01),
    "benin": (6.34, 5.63),
    "abeokuta": (7.16, 3.35),
    "kaduna": (10.52, 7.43),
    "enugu": (6.44, 7.50),
    "warri": (5.52, 5.76),
    # US major metros
    "new york": (40.71, -74.01),
    "los angeles": (34.05, -118.24),
    "chicago": (41.88, -87.63),
    "houston": (29.76, -95.37),
    "phoenix": (33.45, -112.07),
    "philadelphia": (39.95, -75.17),
    "san antonio": (29.42, -98.49),
    "san diego": (32.72, -117.16),
    "dallas": (32.78, -96.80),
    "miami": (25.76, -80.19),
    "atlanta": (33.75, -84.39),
    "seattle": (47.61, -122.33),
    "denver": (39.74, -104.99),
    "boston": (42.36, -71.06),
    "san francisco": (37.77, -122.42),
    # UK
    "london": (51.51, -0.13),
    "manchester": (53.48, -2.24),
    "birmingham": (52.49, -1.89),
    "leeds": (53.80, -1.55),
    "glasgow": (55.86, -4.25),
    # Europe
    "paris": (48.86, 2.35),
    "berlin": (52.52, 13.41),
    "amsterdam": (52.37, 4.90),
    "madrid": (40.42, -3.70),
    "rome": (41.90, 12.50),
    # Asia / Middle East
    "dubai": (25.20, 55.27),
    "mumbai": (19.08, 72.88),
    "singapore": (1.35, 103.82),
    "tokyo": (35.68, 139.69),
    "sydney": (-33.87, 151.21),
    # South America
    "são paulo": (-23.55, -46.63),
    "sao paulo": (-23.55, -46.63),
    "buenos aires": (-34.60, -58.38),
    "bogota": (4.71, -74.07),
    # Africa
    "johannesburg": (-26.20, 28.05),
    "cairo": (30.04, 31.24),
    "nairobi": (-1.29, 36.82),
    "accra": (5.56, -0.19),
    "cape town": (-33.93, 18.42),
}

# Common country code → name mapping for the UI
SUPPORTED_COUNTRIES = {
    "NG": "Nigeria",
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "DE": "Germany",
    "FR": "France",
    "IN": "India",
    "ZA": "South Africa",
    "KE": "Kenya",
    "GH": "Ghana",
    "BR": "Brazil",
    "AE": "UAE",
    "AU": "Australia",
    "SG": "Singapore",
    "JP": "Japan",
    "ES": "Spain",
    "IT": "Italy",
    "NL": "Netherlands",
    "MX": "Mexico",
    "CO": "Colombia",
    "AR": "Argentina",
    "EG": "Egypt",
}


# ------------------------------------------------------------------
# Holiday Builder
# ------------------------------------------------------------------

class HolidayBuilder:
    """
    Generate holiday indicator covariates for a date range.

    Uses the ``holidays`` Python package.
    """

    def __init__(self, country_code: str = "NG", years: Optional[List[int]] = None):
        self.country_code = country_code.upper()
        self.years = years

    def build(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Build holiday covariates aligned to the given date index.

        Returns a DataFrame with columns:
        - is_holiday: 1 if date is a public holiday
        - near_holiday: 1 if within ±1 day of a holiday
        """
        try:
            import holidays as holidays_pkg
        except ImportError:
            logger.warning("holidays package not installed. Skipping holiday covariates.")
            return pd.DataFrame(index=date_index)

        # Determine years needed
        years = sorted(set(d.year for d in date_index))
        try:
            country_holidays = holidays_pkg.country_holidays(
                self.country_code,
                years=years,
            )
        except Exception as e:
            logger.warning(f"Could not load holidays for {self.country_code}: {e}")
            return pd.DataFrame(index=date_index)

        # Build binary indicators
        holiday_dates = set(country_holidays.keys())
        is_holiday = np.array([1.0 if d.date() in holiday_dates else 0.0 for d in date_index])

        # Near-holiday: ±1 day
        extended_dates = set()
        for hd in holiday_dates:
            extended_dates.add(hd - timedelta(days=1))
            extended_dates.add(hd)
            extended_dates.add(hd + timedelta(days=1))
        near_holiday = np.array([1.0 if d.date() in extended_dates else 0.0 for d in date_index])

        result = pd.DataFrame({
            "is_holiday": is_holiday,
            "near_holiday": near_holiday,
        }, index=date_index)

        n_holidays = int(is_holiday.sum())
        logger.info(f"Found {n_holidays} holiday days in range ({self.country_code}).")
        return result


# ------------------------------------------------------------------
# Weather Builder
# ------------------------------------------------------------------

class WeatherBuilder:
    """
    Fetch historical weather data from the Open-Meteo API.

    Completely free, no API key required.
    """

    API_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def build(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Fetch weather covariates aligned to the given date index.

        Returns a DataFrame with columns:
        - temp_max: daily max temperature (°C), z-scored
        - temp_min: daily min temperature (°C), z-scored
        - precip: daily precipitation sum (mm), z-scored
        """
        import requests

        start_date = date_index.min().strftime("%Y-%m-%d")
        end_date = date_index.max().strftime("%Y-%m-%d")

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "auto",
        }

        try:
            resp = requests.get(self.API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Open-Meteo API failed: {e}. Skipping weather covariates.")
            return pd.DataFrame(index=date_index)

        if "daily" not in data:
            logger.warning("No daily data in Open-Meteo response.")
            return pd.DataFrame(index=date_index)

        daily = data["daily"]
        weather_df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily.get("temperature_2m_max", []),
            "temp_min": daily.get("temperature_2m_min", []),
            "precip": daily.get("precipitation_sum", []),
        }).set_index("date")

        # Fill any missing values
        weather_df = weather_df.fillna(0)

        # Z-score normalization
        for col in ["temp_max", "temp_min", "precip"]:
            if col in weather_df.columns:
                mean = weather_df[col].mean()
                std = weather_df[col].std()
                if std > 0:
                    weather_df[col] = (weather_df[col] - mean) / std
                else:
                    weather_df[col] = 0.0

        # Align to the requested date index
        weather_df = weather_df.reindex(date_index).fillna(0)

        logger.info(
            f"Fetched weather data for ({self.latitude}, {self.longitude}): "
            f"{len(weather_df)} days."
        )
        return weather_df


# ------------------------------------------------------------------
# Combined builder
# ------------------------------------------------------------------

def build_covariates(
    date_index: pd.DatetimeIndex,
    country_code: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    geo_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Build a combined covariate matrix Z for the given date range.

    Parameters
    ----------
    date_index : DatetimeIndex
    country_code : e.g. "NG", "US" (for holidays)
    latitude, longitude : for weather data
    geo_name : if provided, auto-resolves lat/lon from built-in lookup

    Returns
    -------
    np.ndarray of shape (T, n_covariates) or None if no covariates requested.
    """
    frames = []
    covariate_names = []

    # --- Holidays ---
    if country_code:
        hb = HolidayBuilder(country_code)
        hol_df = hb.build(date_index)
        if not hol_df.empty:
            frames.append(hol_df)
            covariate_names.extend(hol_df.columns.tolist())

    # --- Weather ---
    # Auto-resolve coordinates from geo name if not provided
    if latitude is None and longitude is None and geo_name:
        coords = GEO_COORDINATES.get(geo_name.lower().strip())
        if coords:
            latitude, longitude = coords
            logger.info(f"Auto-resolved '{geo_name}' → ({latitude}, {longitude})")

    if latitude is not None and longitude is not None:
        wb = WeatherBuilder(latitude, longitude)
        wx_df = wb.build(date_index)
        if not wx_df.empty:
            frames.append(wx_df)
            covariate_names.extend(wx_df.columns.tolist())

    if not frames:
        return None, []

    combined = pd.concat(frames, axis=1).reindex(date_index).fillna(0)
    logger.info(f"Built covariate matrix: {combined.shape[1]} features × {len(combined)} days.")
    return combined.values, covariate_names


def get_covariate_names(
    country_code: Optional[str] = None,
    has_weather: bool = False,
) -> List[str]:
    """Return the expected covariate column names for display purposes."""
    names = []
    if country_code:
        names.extend(["is_holiday", "near_holiday"])
    if has_weather:
        names.extend(["temp_max", "temp_min", "precip"])
    return names
