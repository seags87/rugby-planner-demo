from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Dict, Optional

import httpx

GOOGLE_WEATHER_BASE = "https://weather.googleapis.com/v1"


def get_weather(location: str, for_date: Optional[date] = None) -> Dict[str, Any]:
    """Return a compact weather dict for a location and date.

    Shape: {summary, temp_c, precip, wind_kph, source}
    - Uses Google Maps Weather API when GOOGLE_MAPS_API_KEY is set.
    - If missing or requests fail, returns a clear "unavailable" status (no offline fake).
    - Limits forecasts to <= 10 days from today.
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    # Determine the target date and whether a forecast should be available (<= 10 days out)
    today = date.today()
    target_date = for_date or today
    delta_days = (target_date - today).days

    if delta_days > 10:
        # Too far out; let the user know a forecast isn't available yet.
        return {
            "location": location,
            "summary": f"No forecast available yet for {target_date.isoformat()} (more than 10 days out)",
            "temp_c": "N/A",
            "precip": 0,
            "wind_kph": "N/A",
            "source": "unavailable",
        }

    if not api_key:
        return {
            "location": location,
            "summary": "Weather currently unavailable (no Google API key configured)",
            "temp_c": "N/A",
            "precip": 0,
            "wind_kph": "N/A",
            "source": "unavailable",
        }

    # Live call (best-effort). We first geocode via Places Text Search (lat/lng),
    # then request forecast via Google Weather API: /forecast/days:lookup
    try:
        with httpx.Client(timeout=10.0) as client:
            # Geocode via Places Text Search (Maps Places API) to get lat/lng
            place_url = "https://places.googleapis.com/v1/places:searchText"
            headers = {
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.location,places.displayName",
            }
            payload = {"textQuery": location}
            p = client.post(place_url, headers=headers, json=payload)
            try:
                p.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = p.text[:300]
                return {
                    "location": location,
                    "summary": f"Weather currently unavailable (geocoding HTTP {p.status_code})",
                    "temp_c": "N/A",
                    "precip": 0,
                    "wind_kph": "N/A",
                    "source": "unavailable",
                    "_detail": detail,
                }
            pdata = p.json()
            loc = pdata.get("places", [{}])[0].get("location", {})
            lat, lng = loc.get("latitude"), loc.get("longitude")
            if lat is None or lng is None:
                return {
                    "location": location,
                    "summary": "Weather currently unavailable (geocoding returned no results)",
                    "temp_c": "N/A",
                    "precip": 0,
                    "wind_kph": "N/A",
                    "source": "unavailable",
                }

            # Forecast endpoint — use the documented days:lookup pattern with query params
            # Example: https://weather.googleapis.com/v1/forecast/days:lookup
            weather_url = f"{GOOGLE_WEATHER_BASE}/forecast/days:lookup"
            days = max(1, min(10, (delta_days if delta_days >= 0 else 0) + 1))
            wparams = {
                "key": api_key,
                "location.latitude": lat,
                "location.longitude": lng,
                "days": days,
                "languageCode": "en",
            }
            wheaders = {"X-Goog-Api-Key": api_key}
            w = client.get(weather_url, params=wparams, headers=wheaders)
            try:
                w.raise_for_status()
            except httpx.HTTPStatusError:
                # Try again with header-only (without key in query) in case restrictions apply
                w = client.get(weather_url, params={k: v for k, v in wparams.items() if k != "key"}, headers=wheaders)
                try:
                    w.raise_for_status()
                except httpx.HTTPStatusError as e:
                    text = w.text[:300]
                    return {
                        "location": location,
                        "summary": f"Weather currently unavailable (forecast HTTP {w.status_code})",
                        "temp_c": "N/A",
                        "precip": 0,
                        "wind_kph": "N/A",
                        "source": "unavailable",
                        "_detail": text,
                    }
            wj = w.json()
            # Reduce to a small shape; be defensive about response structure
            # Preferred per current API: top-level `forecastDays` array
            days_list = (
                wj.get("forecastDays")
                or wj.get("forecast", {}).get("forecastDays")
                or wj.get("dailyForecasts")
                or wj.get("forecast", {}).get("dailyForecasts")
                or wj.get("days")
                or wj.get("daily")
                or []
            )
            idx = min(max(0, delta_days), max(0, len(days_list) - 1))
            d = days_list[idx] if days_list else {}

            def _get_precip(v: Any) -> Optional[float]:
                try:
                    if isinstance(v, dict):
                        v = v.get("value") or v.get("percentage") or v.get("chance")
                    if v is None:
                        return None
                    v = float(v)
                    return v / 100.0 if v > 1 else v
                except Exception:
                    return None

            # Helpers to pull fields from the documented structure
            daytime = d.get("daytimeForecast", {})
            nighttime = d.get("nighttimeForecast", {})

            # Summary text: prefer daytime description text
            summary = (
                (daytime.get("weatherCondition", {}).get("description", {}) or {}).get("text")
                or (nighttime.get("weatherCondition", {}).get("description", {}) or {}).get("text")
                or d.get("summary")
                or "Unavailable"
            )

            # Temperatures (Celsius degrees on daily object)
            temp_c = None
            try:
                temp_c = (
                    (d.get("maxTemperature") or {}).get("degrees")
                    or (d.get("feelsLikeMaxTemperature") or {}).get("degrees")
                )
            except Exception:
                temp_c = None
            if temp_c is None:
                temp_c = "Unavailable"

            # Precipitation probability percent -> 0..1
            precip = None
            try:
                precip = (
                    ((daytime.get("precipitation") or {}).get("probability") or {}).get("percent")
                    or ((nighttime.get("precipitation") or {}).get("probability") or {}).get("percent")
                )
            except Exception:
                precip = None
            precip = _get_precip(precip)

            # Wind kph: prefer gust, else speed; convert if mph
            def _wind_kph(block: Dict[str, Any]) -> Optional[float]:
                wind = block.get("wind") or {}
                gust = (wind.get("gust") or {}).get("value")
                gust_unit = (wind.get("gust") or {}).get("unit")
                spd = (wind.get("speed") or {}).get("value")
                spd_unit = (wind.get("speed") or {}).get("unit")
                val, unit = (gust, gust_unit) if gust is not None else (spd, spd_unit)
                if val is None:
                    return None
                try:
                    v = float(val)
                except Exception:
                    return None
                unit = (unit or "").upper()
                if unit in ("KILOMETERS_PER_HOUR", "KMH", "KPH"):
                    return v
                if unit in ("MILES_PER_HOUR", "MPH"):
                    return v * 1.60934
                return v  # assume already kph

            wind_kph = _wind_kph(daytime) or _wind_kph(nighttime) or "Unavailable"

            return {
                "location": location,
                "summary": summary,
                "temp_c": temp_c,
                "precip": precip,
                "wind_kph": wind_kph,
                "source": "google",
            }
    except httpx.RequestError as e:
        return {
            "location": location,
            "summary": f"Weather currently unavailable (network error: {e.__class__.__name__})",
            "temp_c": "N/A",
            "precip": 0,
            "wind_kph": "N/A",
            "source": "unavailable",
        }
    except Exception as e:
        return {
            "location": location,
            "summary": f"Weather currently unavailable (failed to retrieve forecast)",
            "temp_c": "N/A",
            "precip": 0,
            "wind_kph": "N/A",
            "source": "unavailable",
        }
