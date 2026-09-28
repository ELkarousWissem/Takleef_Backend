import time
import requests
from urllib.parse import urlencode

from django.conf import settings
from django.core.cache import cache
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status


NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"


def _headers(request):
    # IMPORTANT: set a real contact email in settings (see below)
    ua = getattr(settings, "NOMINATIM_USER_AGENT", None)
    if not ua:
        ua = "Mobicrowd/1.0 (contact: you@example.com)"  # replace in settings

    # Referer is optional but helps when running behind some proxies
    referer = getattr(settings, "NOMINATIM_REFERER", None)
    h = {
        "User-Agent": ua,
        "Accept": "application/json",
        "Accept-Language": request.headers.get("Accept-Language", "en"),
    }
    if referer:
        h["Referer"] = referer
    return h


def _global_throttle(min_interval_sec: float = 1.05) -> None:
    """
    Ensure we do NOT hit Nominatim faster than ~1 req/sec *globally*.
    This avoids returning 429 to the frontend while still respecting limits.
    """
    key = "nominatim:last_call_ts"
    now = time.time()
    last = cache.get(key)
    if last:
        wait = min_interval_sec - (now - float(last))
        if wait > 0:
            time.sleep(wait)
    cache.set(key, time.time(), timeout=30)


def _safe_error_payload(r: requests.Response, max_len: int = 300):
    txt = ""
    try:
        txt = (r.text or "")[:max_len]
    except Exception:
        txt = ""
    return {"upstream_status": r.status_code, "upstream_body": txt}


class GeocodeSearchView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        q = (request.query_params.get("q") or "").strip()
        if not q:
            return Response([], status=status.HTTP_200_OK)

        allowed = {
            "format", "q", "limit", "addressdetails", "dedupe",
            "countrycodes", "viewbox", "bounded", "layer"
        }
        params = {k: v for k, v in request.query_params.items() if k in allowed}
        params["q"] = q
        params.setdefault("format", "jsonv2")
        params.setdefault("limit", "8")
        params.setdefault("addressdetails", "1")
        params.setdefault("dedupe", "1")

        cache_key = "nominatim:search:" + urlencode(sorted(params.items()))
        cached = cache.get(cache_key)
        if cached is not None:
            return Response(cached, status=status.HTTP_200_OK)

        _global_throttle()

        try:
            r = requests.get(NOMINATIM_SEARCH, params=params, headers=_headers(request), timeout=10)
        except requests.RequestException as e:
            return Response({"detail": "Upstream request failed", "error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)

        if r.status_code != 200:
            # return a clean error instead of 500
            return Response(
                {"detail": "Upstream geocoder error", **_safe_error_payload(r)},
                status=status.HTTP_502_BAD_GATEWAY
            )

        data = r.json()
        cache.set(cache_key, data, timeout=24 * 3600)
        return Response(data, status=status.HTTP_200_OK)


class GeocodeReverseView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        lat = request.query_params.get("lat")
        lon = request.query_params.get("lon")
        if lat is None or lon is None:
            return Response({"detail": "lat and lon are required"}, status=status.HTTP_400_BAD_REQUEST)

        allowed = {"format", "lat", "lon", "zoom", "addressdetails"}
        params = {k: v for k, v in request.query_params.items() if k in allowed}
        params.setdefault("format", "json")
        params.setdefault("zoom", "18")
        params.setdefault("addressdetails", "1")

        cache_key = "nominatim:reverse:" + urlencode(sorted(params.items()))
        cached = cache.get(cache_key)
        if cached is not None:
            return Response(cached, status=status.HTTP_200_OK)

        _global_throttle()

        try:
            r = requests.get(NOMINATIM_REVERSE, params=params, headers=_headers(request), timeout=10)
        except requests.RequestException as e:
            return Response({"detail": "Upstream request failed", "error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)

        if r.status_code != 200:
            return Response(
                {"detail": "Upstream geocoder error", **_safe_error_payload(r)},
                status=status.HTTP_502_BAD_GATEWAY
            )

        data = r.json()
        cache.set(cache_key, data, timeout=24 * 3600)
        return Response(data, status=status.HTTP_200_OK)