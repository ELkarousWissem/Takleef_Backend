from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .helpers import (
    check_text_safety,
    parse_bool,
)


class TextualSafetyAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        text = (
            request.data.get("text")
            or request.data.get("message")
            or request.data.get("content")
            or request.data.get("answer")
            or ""
        )

        use_llm = parse_bool(
            request.data.get("use_llm"),
            default=True,
        )

        output_language = request.data.get("output_language", "auto")

        try:
            result = check_text_safety(
                text=text,
                use_llm=use_llm,
                output_language=output_language,
            )

            if not result.get("ok"):
                return Response(result, status=status.HTTP_400_BAD_REQUEST)

            return Response(result, status=status.HTTP_200_OK)

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
