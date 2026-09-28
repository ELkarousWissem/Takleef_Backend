"""Standalone API views for description extraction and processing."""

from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

import mobicrowd.views.apis.description_enh.helpers as h

_HIDDEN_FIELDS = {
    "model",
    "provider_used",
    "provider_used_label",
    "configured_fallback_models",
    "openrouter_key_count",
    "title_backend",
}


def _public_payload(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in _HIDDEN_FIELDS}


def _error(message: str, http_status: int) -> Response:
    return Response(
        {"status": "error", "message": message},
        status=http_status,
    )


class _TextEnhPostAPIView(APIView):
    parser_classes = [JSONParser, FormParser]
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            description = h.parse_description(request.data)
            result = self.run(description, data=request.data)
            return Response(_public_payload(result))
        except ValueError as error:
            return _error(str(error), status.HTTP_400_BAD_REQUEST)
        except Exception as error:
            return _error(str(error), status.HTTP_500_INTERNAL_SERVER_ERROR)

    def run(self, description, *, data):
        raise NotImplementedError


class TextEnhExtractAPIView(_TextEnhPostAPIView):
    def run(self, description, *, data):
        return h.run_extract(description)


class TextEnhProcessAPIView(_TextEnhPostAPIView):
    def run(self, description, *, data):
        return h.run_process(description, data)