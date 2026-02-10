import os
import requests
from typing import Any, Dict, List, Optional

BASE = "https://api.the-odds-api.com/v4/sports"

class TheOddsAPI:
    """
    Adapter for The Odds API (v4).
    Docs vary by plan; we keep this flexible and only assume:
      - sport key like 'baseball_mlb'
      - markets like 'h2h' (moneyline)
      - regions like 'us'
      - oddsFormat 'american'
    """
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing ODDS_API_KEY env var")
        self.timeout = timeout
        self.sess = requests.Session()

    def get_moneylines(
        self,
        date_iso: Optional[str] = None,   # some providers support; if not, ignore upstream
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
        }
        # Not all plans support a date filter; harmless if ignored by provider
        if date_iso:
            params["dateFormat"] = "iso"
            params["commenceTimeFrom"] = date_iso

        r = self.sess.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
