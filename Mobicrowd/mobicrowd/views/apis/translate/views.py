"""Standalone API views for text translation."""

from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

import mobicrowd.views.apis.translate.helpers as h



def _attach_provider(result: dict) -> dict:
    provider = result.get("provider_used") or h.DEFAULT_LLM_PROVIDER
    result["provider_used"] = provider
    result["provider_used_label"] = h.provider_used_label(provider)
    return result


def _error(message: str, http_status: int) -> Response:
    return Response(
        _attach_provider({"status": "error", "message": message}),
        status=http_status,
    )


class DetectLanguageAPIView(APIView):
    parser_classes = [JSONParser, FormParser]
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            text = h.parse_text(request.data)
            result = h.run_detect_language(text)
            return Response(_attach_provider(result))
        except ValueError as error:
            return _error(str(error), status.HTTP_400_BAD_REQUEST)
        except Exception as error:
            return _error(str(error), status.HTTP_500_INTERNAL_SERVER_ERROR)


class TranslateAPIView(APIView):
    parser_classes = [JSONParser, FormParser]
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        try:
            text = h.parse_text(request.data)
            source_lang, target_lang = h.parse_translate_languages(request.data)
            result = h.run_translate(
                text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            return Response(_attach_provider(result))
        except ValueError as error:
            return _error(str(error), status.HTTP_400_BAD_REQUEST)
        except Exception as error:
            return _error(str(error), status.HTTP_500_INTERNAL_SERVER_ERROR)