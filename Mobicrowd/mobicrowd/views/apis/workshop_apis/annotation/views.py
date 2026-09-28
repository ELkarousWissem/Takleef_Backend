from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from mobicrowd.views.apis.workshop_apis.annotation.helpers import (
    build_annotation_response,
    parse_bool,
    parse_int,
    run_annotation,
)


DEFAULT_TASK = "Detect and annotate all visible objects in this image."
DEFAULT_MAX_TOKENS = 900


class OpenRouterAnnotationAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        image = request.FILES.get("image")

        task = request.data.get("task", DEFAULT_TASK)
        max_tokens = parse_int(
            request.data.get("max_tokens"),
            default=DEFAULT_MAX_TOKENS,
        )

        include_base64 = parse_bool(
            request.data.get("include_base64"),
            default=True,
        )

        include_summary = parse_bool(
            request.data.get("include_summary"),
            default=True,
        )

        include_raw_model_output = parse_bool(
            request.data.get("include_raw_model_output"),
            default=False,
        )

        try:
            result = run_annotation(
                image=image,
                task=task,
                max_tokens=max_tokens,
            )

            payload = build_annotation_response(
                result,
                include_base64=include_base64,
                include_summary=include_summary,
                include_raw_model_output=include_raw_model_output,
            )

            if result.get("status") != "success":
                return Response(payload, status=status.HTTP_502_BAD_GATEWAY)

            return Response(payload, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
