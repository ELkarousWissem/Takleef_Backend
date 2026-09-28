import json

from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .helpers import (
    ask_story_teller_follow_up,
    run_story_teller_from_uploads,
)


class StoryTellerAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        uploaded_files = request.FILES.getlist("media")

        if not uploaded_files:
            uploaded_files = request.FILES.getlist("files")

        if not uploaded_files:
            single_file = (
                request.FILES.get("file")
                or request.FILES.get("image")
                or request.FILES.get("video")
            )

            if single_file:
                uploaded_files = [single_file]

        recap_prompt = request.data.get("recap_prompt", "")
        output_mode = request.data.get("output_mode", "detailed_summary")
        original_request = request.data.get("original_request", "")

        try:
            result = run_story_teller_from_uploads(
                uploaded_files=uploaded_files,
                recap_prompt=recap_prompt,
                output_mode=output_mode,
                original_request=original_request,
            )

            if result.get("status") != "success":
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


class StoryTellerFollowUpAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        question = request.data.get("question", "")
        story_text = request.data.get("story", "")

        media_metadata_raw = request.data.get("media_metadata", [])
        chat_history_raw = request.data.get("chat_history", [])

        try:
            if isinstance(media_metadata_raw, str):
                media_metadata = json.loads(media_metadata_raw or "[]")
            else:
                media_metadata = media_metadata_raw or []

            if isinstance(chat_history_raw, str):
                chat_history = json.loads(chat_history_raw or "[]")
            else:
                chat_history = chat_history_raw or []

            result = ask_story_teller_follow_up(
                question=question,
                story_text=story_text,
                media_metadata=media_metadata,
                chat_history=chat_history,
            )

            if result.get("status") != "success":
                return Response(result, status=status.HTTP_400_BAD_REQUEST)

            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
