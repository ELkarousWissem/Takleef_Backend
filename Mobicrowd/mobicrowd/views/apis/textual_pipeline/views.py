"""Diagnostic textual relevance + redundancy APIs.

These endpoints are intentionally non-authoritative: they never change Submission.status,
never create a Submission, and redundancy diagnostics do not persist Qdrant points.
The authoritative text decision runs inside SubmissionCreateView.
"""

from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

import mobicrowd.views.apis.textual_pipeline.helpers as h


def _error(message: str, http_status: int) -> Response:
    return Response({"status": "error", "message": message}, status=http_status)


class _TextualPostAPIView(APIView):
    parser_classes = [JSONParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        try:
            payload = self.run(request.data)
            payload["authoritative"] = False
            payload["notice"] = (
                "Diagnostic result only. Submission approval is decided server-side by /submissions/create."
            )
            if payload.get("ok") is False and "Empty" in str(payload.get("reason", "")):
                return Response(payload, status=status.HTTP_400_BAD_REQUEST)
            return Response(payload, status=status.HTTP_200_OK)
        except ValueError as exc:
            return _error(str(exc), status.HTTP_400_BAD_REQUEST)
        except Exception:
            return _error("Text diagnostic service failed.", status.HTTP_500_INTERNAL_SERVER_ERROR)

    def run(self, data):
        raise NotImplementedError


class TextualRelevanceAPIView(_TextualPostAPIView):
    def run(self, data):
        return h.run_relevance(data)


class TextualRedundancyAPIView(_TextualPostAPIView):
    def run(self, data):
        # h.run_redundancy is persist=False by design.
        return h.run_redundancy(data)


__all__ = ["TextualRelevanceAPIView", "TextualRedundancyAPIView"]
