from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .helpers import (
    parse_bool,
    run_background_removal,
)


class BackgroundRemovalAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        image = request.FILES.get("image")

        use_cuda = parse_bool(
            request.data.get("use_cuda"),
            default=False,
        )

        try:
            result = run_background_removal(
                image=image,
                use_cuda=use_cuda,
            )

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
