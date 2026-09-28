from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .helpers import run_recaptured_check


class RecapturedCheckAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        uploaded_file = (
            request.FILES.get("file")
            or request.FILES.get("image")
            or request.FILES.get("video")
        )

        media_type = request.data.get(
            "media_type",
            ""
        )

        try:
            result = run_recaptured_check(
                uploaded_file=uploaded_file,
                media_type=media_type,
            )

            # Simple backend console output
            print("\nRECAPTURE RESULT:")
            print(result)
            print()

            if result.get("status") != "success":
                return Response(
                    result,
                    status=status.HTTP_400_BAD_REQUEST
                )

            return Response(
                result,
                status=status.HTTP_200_OK
            )

        except ValueError as e:
            print(
                "RECAPTURE ERROR:",
                str(e)
            )

            return Response(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        except FileNotFoundError as e:
            print(
                "RECAPTURE ERROR:",
                str(e)
            )

            return Response(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        except Exception as e:
            print(
                "RECAPTURE ERROR:",
                str(e)
            )

            return Response(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )