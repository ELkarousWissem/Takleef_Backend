from django.utils import timezone
from rest_framework import serializers

from mobicrowd.models.submisson import Event


class HomeEventSerializer(serializers.ModelSerializer):
    """Small event payload used only by the home-page event card."""

    status = serializers.SerializerMethodField()
    type_label = serializers.SerializerMethodField()
    joined_workers_count = serializers.IntegerField(read_only=True)
    approved_submissions_count = serializers.IntegerField(read_only=True)
    requested_items_count = serializers.SerializerMethodField()
    progress_percent = serializers.SerializerMethodField()

    class Meta:
        model = Event
        fields = (
            "id",
            "title",
            "location",
            "startdate",
            "deadline",
            "created_at",
            "media_types",
            "numberOfPhotos",
            "numberOfVideos",
            "numberOfTexts",
            "status",
            "type_label",
            "joined_workers_count",
            "approved_submissions_count",
            "requested_items_count",
            "progress_percent",
        )

    def _now(self):
        return self.context.get("now") or timezone.now()

    def get_status(self, obj):
        return "active" if obj.startdate <= self._now() else "upcoming"

    def get_type_label(self, obj):
        raw_media_types = obj.media_types or []
        if isinstance(raw_media_types, str):
            raw_media_types = [raw_media_types]

        media_types = [
            str(value).strip().lower()
            for value in raw_media_types
            if str(value).strip()
        ]

        if not media_types:
            return "Event"

        labels = {
            Event.PHOTO: "Photo",
            Event.VIDEO: "Video",
            Event.TEXT: "Text",
        }
        resolved = [labels.get(value, value.title()) for value in media_types]
        return resolved[0] if len(resolved) == 1 else "Mixed"

    def get_requested_items_count(self, obj):
        return sum(
            max(int(value or 0), 0)
            for value in (
                obj.numberOfPhotos,
                obj.numberOfVideos,
                obj.numberOfTexts,
            )
        )

    def get_progress_percent(self, obj):
        requested = self.get_requested_items_count(obj)
        if requested <= 0:
            return 0

        approved = max(int(getattr(obj, "approved_submissions_count", 0) or 0), 0)
        return min(100, round((approved / requested) * 100))