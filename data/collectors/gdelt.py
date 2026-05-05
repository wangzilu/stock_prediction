import requests
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class GDELTCollector:
    """Collects geopolitical event data from GDELT Project.

    GDELT monitors news media worldwide and encodes events with
    Goldstein scale scores (-10 to +10, conflict to cooperation).
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "StockPrediction/1.0"
        })

    def fetch_events_by_country(
        self,
        country_code: str = "CH",  # China
        days: int = 7,
        max_records: int = 250,
    ) -> pd.DataFrame:
        """Fetch recent events involving a country.

        Uses GDELT DOC 2.0 API to search for events.

        Args:
            country_code: FIPS country code (CH=China, US=United States)
            days: Number of days to look back
            max_records: Maximum records to return

        Returns:
            DataFrame with columns [date, title, tone, source_country, url].
            Empty DataFrame if fetch fails.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # GDELT DOC 2.0 API
            url = f"{self.BASE_URL}/doc/doc"
            params = {
                "query": f"sourcelang:english sourcecountry:{country_code}",
                "mode": "ArtList",
                "maxrecords": max_records,
                "format": "json",
                "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
                "sort": "DateDesc",
            }

            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"GDELT API returned status {resp.status_code}")
                return pd.DataFrame()

            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                return pd.DataFrame()

            records = []
            for article in articles:
                records.append({
                    "date": article.get("seendate", ""),
                    "title": article.get("title", ""),
                    "tone": float(article.get("tone", 0)),
                    "source_country": article.get("sourcecountry", ""),
                    "domain": article.get("domain", ""),
                    "url": article.get("url", ""),
                })

            df = pd.DataFrame(records)
            if "date" in df.columns and not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

        except Exception as e:
            logger.warning(f"GDELT event fetch failed: {e}")
            return pd.DataFrame()

    def fetch_china_us_relations(self, days: int = 7) -> pd.DataFrame:
        """Fetch events related to China-US relations.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame of articles about China-US relations with tone scores.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            url = f"{self.BASE_URL}/doc/doc"
            params = {
                "query": "(china OR chinese) (united states OR american OR US) (trade OR tariff OR sanction OR military OR diplomacy OR taiwan)",
                "mode": "ArtList",
                "maxrecords": 100,
                "format": "json",
                "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
                "sort": "DateDesc",
                "sourcelang": "english",
            }

            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return pd.DataFrame()

            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                return pd.DataFrame()

            records = []
            for article in articles:
                records.append({
                    "date": article.get("seendate", ""),
                    "title": article.get("title", ""),
                    "tone": float(article.get("tone", 0)),
                    "domain": article.get("domain", ""),
                    "url": article.get("url", ""),
                })

            df = pd.DataFrame(records)
            if "date" in df.columns and not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

        except Exception as e:
            logger.warning(f"GDELT China-US fetch failed: {e}")
            return pd.DataFrame()

    def fetch_geopolitical_conflicts(self, days: int = 7) -> pd.DataFrame:
        """Fetch events related to global conflicts and security.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame of conflict-related articles with tone scores.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            url = f"{self.BASE_URL}/doc/doc"
            params = {
                "query": "(conflict OR war OR military OR attack OR missile OR nuclear OR invasion OR sanction)",
                "mode": "ArtList",
                "maxrecords": 100,
                "format": "json",
                "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
                "sort": "ToneDesc",
                "sourcelang": "english",
            }

            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return pd.DataFrame()

            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                return pd.DataFrame()

            records = []
            for article in articles:
                records.append({
                    "date": article.get("seendate", ""),
                    "title": article.get("title", ""),
                    "tone": float(article.get("tone", 0)),
                    "domain": article.get("domain", ""),
                    "url": article.get("url", ""),
                })

            df = pd.DataFrame(records)
            if "date" in df.columns and not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df

        except Exception as e:
            logger.warning(f"GDELT conflict fetch failed: {e}")
            return pd.DataFrame()

    def fetch_tone_timeseries(self, query: str, days: int = 30) -> pd.DataFrame:
        """Fetch daily average tone timeseries for a query.

        Uses GDELT DOC 2.0 timeline API.

        Args:
            query: Search query string
            days: Number of days

        Returns:
            DataFrame with columns [date, tone] indexed by date.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            url = f"{self.BASE_URL}/doc/doc"
            params = {
                "query": query,
                "mode": "TimelineTone",
                "format": "json",
                "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
                "sourcelang": "english",
            }

            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return pd.DataFrame()

            data = resp.json()
            timeline = data.get("timeline", [])
            if not timeline:
                return pd.DataFrame()

            # Timeline format: list of {series, data: [{date, value}]}
            records = []
            for series in timeline:
                for point in series.get("data", []):
                    records.append({
                        "date": point.get("date", ""),
                        "tone": float(point.get("value", 0)),
                    })

            df = pd.DataFrame(records)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                df = df.set_index("date").sort_index()
            return df

        except Exception as e:
            logger.warning(f"GDELT tone timeseries fetch failed: {e}")
            return pd.DataFrame()
