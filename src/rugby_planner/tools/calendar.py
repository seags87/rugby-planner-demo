from __future__ import annotations

import json
import os
from datetime import datetime, date as dt_date
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from langsmith import traceable

# Pitchero calendar lookups: fetch JSON for months and search for fixtures.
# Team location is inferred via Google Places and cached on disk.

# Read club configuration from environment (fallback to defaults)
PITCHERO_CLUB_ID = os.getenv("PITCHERO_CLUB_ID", "7732")
DEFAULT_TEAM_NAME = os.getenv("PITCHERO_TEAM_NAME", "Ramsey (IoM)")
PITCHERO_BASE = f"https://www.pitchero.com/data/club/{PITCHERO_CLUB_ID}/calendar"



@traceable(name="calendar.fetch_fixtures", run_type="tool")
def fetch_fixtures(year: int, month: int, *, use_cache_first: bool = True) -> Dict[str, Any]:
    """Fetch fixtures JSON from Pitchero for a given year/month.

    Notes:
    - The API returns a payload with shape: { data: { days: [ { date, fixtures: [...] } ] } }.
    - The use_cache_first flag is accepted for backwards compatibility; no caching is performed here.
    """
    params = {"year": str(year), "month": str(month), "showCompleteWeeks": "1"}
    url = PITCHERO_BASE

    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except Exception:
        return {"key": {"params": params}, "data": {"days": []}}


 


def _iter_year_month(start_year: int, start_month: int, max_months: int = 24):
    """Yield (year, month) pairs moving forward, wrapping Dec -> Jan."""
    year, month = start_year, start_month
    for _ in range(max_months):
        yield year, month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1


@traceable(name="calendar.find_fixture_on_date", run_type="tool")
def find_fixture_on_date(
    target_date: str,
    *,
    team_name: str = DEFAULT_TEAM_NAME,
    max_months: int = 24,
) -> Optional[Dict[str, Any]]:
    """Find the first fixture on a specific calendar date (YYYY-MM-DD)."""
    try:
        start_year, start_month = int(target_date[:4]), int(target_date[5:7])
    except Exception:
        today = dt_date.today()
        start_year, start_month = today.year, today.month
    for year, month in _iter_year_month(start_year, start_month, max_months=max_months):
        payload = fetch_fixtures(year, month)
        for day in payload.get("data", {}).get("days", []):
            if day.get("date") != target_date:
                continue
            for fixture in day.get("fixtures", []) or []:
                return fixture
    return None


@traceable(name="calendar.find_next_fixture_by_opponent", run_type="tool")
def find_next_fixture_by_opponent(
    opponent_substring: str,
    *,
    from_date: Optional[str] = None,
    home_away: Optional[str] = None,
    max_months: int = 24,
) -> Optional[Dict[str, Any]]:
    """Find the next fixture whose opponent name contains the given substring.

    Args:
        opponent_substring: Case-insensitive substring to search in the opponent name.
        from_date: ISO date to start searching from (inclusive). Defaults to today.
        home_away: Optional filter for home/away: 'h' or 'a'.
        max_months: How many month pages to scan forward.
    """
    if not opponent_substring:
        return None
    if from_date is None:
        from_date = dt_date.today().isoformat()
    home_away = (home_away or "").lower() if home_away else None
    try:
        start_year, start_month = int(from_date[:4]), int(from_date[5:7])
    except Exception:
        today = dt_date.today()
        start_year, start_month = today.year, today.month

    needle = opponent_substring.lower()
    for year, month in _iter_year_month(start_year, start_month, max_months=max_months):
        payload = fetch_fixtures(year, month)
        for day in payload.get("data", {}).get("days", []):
            day_date = day.get("date")
            if not day_date or day_date < from_date:
                continue
            for fixture in day.get("fixtures", []) or []:
                if home_away and (fixture.get("ha") or "").lower() != home_away:
                    continue
                opp = (fixture.get("opponent") or "").lower()
                if needle in opp:
                    return fixture
    return None


@traceable(name="calendar.find_next_fixture_across_months", run_type="tool")
def find_next_fixture_across_months(
    *,
    from_date: Optional[str] = None,
    home_away: Optional[str] = None,
    require_club_is_home: bool = False,
    club_name: str = DEFAULT_TEAM_NAME,
    max_months: int = 24,
) -> Optional[Dict[str, Any]]:
    """Scan forward month pages to find the next fixture matching filters.

    Args:
        from_date: ISO date to start searching from (inclusive). Defaults to today.
        home_away: Optional filter for home/away: 'h' or 'a'.
        require_club_is_home: If True, only return fixtures where the homeSide.name
            equals club_name. Used when the query explicitly asks for a "home" fixture.
        club_name: Name of this club as used by the data source.
        max_months: How many month pages to scan forward.
    """
    if from_date is None:
        from_date = dt_date.today().isoformat()
    try:
        start_year, start_month = int(from_date[:4]), int(from_date[5:7])
    except Exception:
        today = dt_date.today()
        start_year, start_month = today.year, today.month

    home_away = (home_away or "").lower() if home_away else None

    for year, month in _iter_year_month(start_year, start_month, max_months=max_months):
        first_window = (year == start_year and month == start_month)
        payload = fetch_fixtures(year, month)
        days = payload.get("data", {}).get("days", [])
        for day in days:
            day_date = day.get("date")
            if not day_date or day_date < from_date:
                continue
            for fixture in day.get("fixtures", []) or []:
                if home_away and (fixture.get("ha") or "").lower() != home_away:
                    continue
                if require_club_is_home:
                    home_name = (fixture.get("homeSide") or {}).get("name")
                    if home_name != club_name:
                        continue
                return fixture
        if not first_window:
            for day in days:
                for fixture in day.get("fixtures", []) or []:
                    if home_away and (fixture.get("ha") or "").lower() != home_away:
                        continue
                    if require_club_is_home:
                        home_name = (fixture.get("homeSide") or {}).get("name")
                        if home_name != club_name:
                            continue
                    return fixture
    return None



def _team_cache_file() -> str:
    root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cache_dir = root / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return str(cache_dir / "team_locations.json")


def _load_team_cache() -> Dict[str, Any]:
    path = _team_cache_file()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_team_cache(cache: Dict[str, Any]) -> None:
    path = _team_cache_file()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _extract_city_country(formatted: str) -> str:
    parts = [p.strip() for p in (formatted or "").split(",") if p.strip()]
    if not parts:
        return formatted or ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}, {parts[-1]}"


def _geocode_team_via_places(team: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return None
    query_variants = [
        team,
        f"{team} rugby club",
        f"{team} RUFC",
    ]
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location",
    }
    url = "https://places.googleapis.com/v1/places:searchText"
    with httpx.Client(timeout=10.0) as client:
        for q in query_variants:
            try:
                r = client.post(url, headers=headers, json={"textQuery": q})
                r.raise_for_status()
                data = r.json()
                places = data.get("places") or []
                if not places:
                    continue
                p0 = places[0]
                return {
                    "displayName": (p0.get("displayName") or {}).get("text"),
                    "formattedAddress": p0.get("formattedAddress"),
                    "location": p0.get("location"),
                }
            except Exception:
                continue
    return None


@traceable(name="calendar.get_team_location_name", run_type="tool")
def get_team_location_name(team: str) -> str:
    """Return a concise "City, Country" location string for a team.

    Uses Google Places for geocoding with a small on-disk cache.
    Returns "Unavailable" if no API key or a lookup fails.
    """
    norm = (team or "").strip()
    if not norm:
        return "Unavailable"
    cache = _load_team_cache()
    key = norm.lower()
    cached = cache.get(key)
    if cached and isinstance(cached, dict) and cached.get("city_country"):
        return cached["city_country"]

    # Lookup via Google Places
    res = _geocode_team_via_places(norm)
    if res and res.get("formattedAddress"):
        city_country = _extract_city_country(res.get("formattedAddress", ""))
        cache[key] = {**res, "city_country": city_country}
        _save_team_cache(cache)
        return city_country

    # No static fallback
    return "Unavailable"


@traceable(name="calendar.infer_location_from_fixture", run_type="tool")
def infer_location_from_fixture(fixture: Dict[str, Any]) -> str:
    """Infer a friendly location string for a fixture using the home team."""
    home_team = (fixture.get("homeSide", {}) or {}).get("name") or ""
    loc = get_team_location_name(home_team)
    return loc or "Unavailable"
