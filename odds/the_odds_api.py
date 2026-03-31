import os
import requests
from typing import Any, Dict, List, Optional

BASE = "https://api.the-odds-api.com/v4/sports"


class TheOddsAPI:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing ODDS_API_KEY env var")
        self.timeout = timeout
        self.sess = requests.Session()

    def get_moneylines_live(
        self,
        commence_from_iso: Optional[str] = None,
        commence_to_iso: Optional[str] = None,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "american",
        sport: str = "baseball_mlb",
    ) -> List[Dict[str, Any]]:
        url = f"{BASE}/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
        }
        if commence_from_iso:
            params["commenceTimeFrom"] = commence_from_iso
        if commence_to_iso:
            params["commenceTimeTo"] = commence_to_iso

        r = self.sess.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_moneylines_historical(
        self,
        snapshot_iso: str,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "american",
        sport: str = "baseball_mlb",
    ) -> List[Dict[str, Any]]:
        url = f"{BASE}/{sport}/odds-history"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
            "date": snapshot_iso,
        }

        r = self.sess.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()

        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
            if "events" in payload and isinstance(payload["events"], list):
                return payload["events"]

        if isinstance(payload, list):
            return payload

        raise RuntimeError(f"Unexpected historical odds response shape: {type(payload)}")