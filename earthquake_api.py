"""
FASTAPI SERVICE: REAL-TIME EARTHQUAKE DATA FROM USGS
====================================================

This FastAPI app exposes simple endpoints to retrieve active earthquake data
from the USGS (United States Geological Survey) feeds.

Run locally:
-----------
1) Install dependencies (PowerShell example):
   pip install "fastapi[standard]" httpx

2) Start the API server from this folder:
   uvicorn earthquake_api:app --reload --port 8000

3) Open the interactive docs in your browser:
   http://127.0.0.1:8000/docs
"""

from datetime import datetime, timedelta
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


USGS_FDSN_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

app = FastAPI(
    title="Earthquake Real‑Time Data API",
    description=(
        "Simple FastAPI wrapper around the USGS Earthquake API.\n\n"
        "Use this service to retrieve recent earthquake events and, if you want, "
        "feed them into your aftershock prediction pipeline."
    ),
    version="1.0.0",
)


class EarthquakeProperties(BaseModel):
    event_id: str = Field(..., description="USGS event ID")
    time: datetime = Field(..., description="Origin time (UTC)")
    magnitude: float = Field(..., description="Magnitude (Mw)")
    place: Optional[str] = Field(None, description="Human‑readable location")
    depth_km: float = Field(..., description="Depth in kilometers")
    latitude: float
    longitude: float
    url: Optional[str] = Field(None, description="USGS detail page URL")


class EarthquakeResponse(BaseModel):
    count: int
    earthquakes: List[EarthquakeProperties]


async def fetch_usgs_events(
    starttime: datetime,
    endtime: datetime,
    min_magnitude: float,
    max_magnitude: Optional[float],
    limit: int,
) -> EarthquakeResponse:
    """Call the USGS FDSN API and normalize the response."""
    params = {
        "format": "geojson",
        "starttime": starttime.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": endtime.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_magnitude,
        "orderby": "time",
        "limit": limit,
    }
    if max_magnitude is not None:
        params["maxmagnitude"] = max_magnitude

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(USGS_FDSN_BASE, params=params)
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Error connecting to USGS service: {exc}",
            ) from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"USGS API error: {resp.text[:300]}",
        )

    data = resp.json()
    features = data.get("features", [])

    earthquakes: List[EarthquakeProperties] = []
    for feature in features:
        try:
            props = feature.get("properties", {})
            geom = feature.get("geometry", {}) or {}
            coords = geom.get("coordinates", [None, None, None])

            quake = EarthquakeProperties(
                event_id=feature.get("id", ""),
                time=datetime.utcfromtimestamp(props.get("time", 0) / 1000.0),
                magnitude=float(props.get("mag")),
                place=props.get("place"),
                depth_km=float(coords[2]) if coords[2] is not None else 0.0,
                longitude=float(coords[0]) if coords[0] is not None else 0.0,
                latitude=float(coords[1]) if coords[1] is not None else 0.0,
                url=props.get("url"),
            )
            earthquakes.append(quake)
        except Exception:
            # Skip malformed entries but continue parsing others
            continue

    return EarthquakeResponse(count=len(earthquakes), earthquakes=earthquakes)


@app.get("/health", summary="Health check")
async def health_check():
    """Simple health endpoint to verify the service is running."""
    return {"status": "ok"}


@app.get(
    "/earthquakes/recent",
    response_model=EarthquakeResponse,
    summary="Get recent earthquakes in a given time window",
)
async def get_recent_earthquakes(
    hours: int = Query(
        24,
        ge=1,
        le=168,
        description="Look‑back window in hours (1–168). Default: last 24 hours.",
    ),
    min_magnitude: float = Query(
        3.0,
        ge=0.0,
        description="Minimum magnitude to include (default 3.0).",
    ),
    max_magnitude: Optional[float] = Query(
        None,
        ge=0.0,
        description="Optional maximum magnitude filter.",
    ),
    limit: int = Query(
        200,
        ge=1,
        le=20000,
        description="Maximum number of events to return (1–20000).",
    ),
):
    """
    Retrieve earthquakes from the USGS catalog in the last N hours.

    Example:
    - Last 24 hours, M≥4.5: `/earthquakes/recent?hours=24&min_magnitude=4.5`
    """
    endtime = datetime.utcnow()
    starttime = endtime - timedelta(hours=hours)

    return await fetch_usgs_events(
        starttime=starttime,
        endtime=endtime,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        limit=limit,
    )


@app.get(
    "/earthquakes/by_time",
    response_model=EarthquakeResponse,
    summary="Get earthquakes between explicit start and end times",
)
async def get_earthquakes_by_time(
    starttime: datetime = Query(
        ...,
        description="Start time (UTC). Example: 2025-01-01T00:00:00",
    ),
    endtime: datetime = Query(
        ...,
        description="End time (UTC). Example: 2025-01-02T00:00:00",
    ),
    min_magnitude: float = Query(
        0.0,
        ge=0.0,
        description="Minimum magnitude to include (default 0.0).",
    ),
    max_magnitude: Optional[float] = Query(
        None,
        ge=0.0,
        description="Optional maximum magnitude filter.",
    ),
    limit: int = Query(
        2000,
        ge=1,
        le=20000,
        description="Maximum number of events to return (1–20000).",
    ),
):
    """
    Retrieve earthquakes between arbitrary start and end times (UTC).

    This is a thin wrapper over the USGS FDSN `/query` endpoint.
    """
    if endtime <= starttime:
        raise HTTPException(status_code=400, detail="endtime must be after starttime")

    return await fetch_usgs_events(
        starttime=starttime,
        endtime=endtime,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        limit=limit,
    )


