from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .helpers import (
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_THRESHOLD,
    build_segmentation_response,
    parse_bool,
    parse_float,
    segment_uploaded_image_text,
)


class ImageSegmentationAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        image = request.FILES.get("image")

        text_prompt = (
            request.data.get("text_prompt")
            or request.data.get("prompt")
            or request.data.get("labels")
            or ""
        )

        threshold = parse_float(
            request.data.get("threshold"),
            default=DEFAULT_THRESHOLD,
        )

        mask_threshold = parse_float(
            request.data.get("mask_threshold"),
            default=DEFAULT_MASK_THRESHOLD,
        )

        show_segmentation = parse_bool(
            request.data.get("show_segmentation"),
            default=True,
        )

        show_boxes = parse_bool(
            request.data.get("show_boxes"),
            default=True,
        )

        show_labels = parse_bool(
            request.data.get("show_labels"),
            default=True,
        )

        blur_boxes = parse_bool(
            request.data.get("blur_boxes"),
            default=False,
        )

        include_base64 = parse_bool(
            request.data.get("include_base64"),
            default=True,
        )

        include_summary = parse_bool(
            request.data.get("include_summary"),
            default=True,
        )

        include_device_info = parse_bool(
            request.data.get("include_device_info"),
            default=False,
        )

        try:
            result = segment_uploaded_image_text(
                image=image,
                text_prompt=text_prompt,
                threshold=threshold,
                mask_threshold=mask_threshold,
                show_segmentation=show_segmentation,
                show_boxes=show_boxes,
                show_labels=show_labels,
                blur_boxes=blur_boxes,
            )

            payload = build_segmentation_response(
                result,
                include_base64=include_base64,
                include_summary=include_summary,
                include_device_info=include_device_info,
            )

            if result.get("status") != "success":
                return Response(payload, status=status.HTTP_400_BAD_REQUEST)

            return Response(payload, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
