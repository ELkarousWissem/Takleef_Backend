from rest_framework import serializers

from mobicrowd.models.submisson import (
    Event,
    Submission,
    Workshop,
    WorkshopMedia,
    WorkshopActionResult,
)


ALLOWED_WORKSHOP_MEDIA_TYPES = {'photo', 'video' , 'text'}


def is_valid_workshop_event(event):
    media_types = set(event.media_types or [])

    if not media_types:
        return False

    return media_types.issubset(ALLOWED_WORKSHOP_MEDIA_TYPES)


class WorkshopEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = Event
        fields = [
            'id',
            'title',
            'description',
            'location',
            'media_types',
            'numberOfPhotos',
            'numberOfVideos',
            'startdate',
            'deadline',
        ]


class WorkshopMediaSerializer(serializers.ModelSerializer):
    event_title = serializers.CharField(source='event.title', read_only=True)
    photo_url = serializers.SerializerMethodField()
    video_url = serializers.SerializerMethodField()

    class Meta:
        model = WorkshopMedia
        fields = [
            'id',
            'workshop',
            'event',
            'event_title',
            'submission',
            'media_type',
            'photo',
            'photo_url',
            'video',
            'video_url',
            'original_event_id',
            'original_submission_id',
            'original_photo_id',
            'original_video_id',
            'action_status',
            'added_at',
            'updated_at',
        ]
        read_only_fields = [
            'id',
            'workshop',
            'event',
            'event_title',
            'submission',
            'media_type',
            'photo',
            'photo_url',
            'video',
            'video_url',
            'original_event_id',
            'original_submission_id',
            'original_photo_id',
            'original_video_id',
            'added_at',
            'updated_at',
        ]

    def get_photo_url(self, obj):
        request = self.context.get('request')

        if obj.photo and obj.photo.image:
            url = obj.photo.image.url
            return request.build_absolute_uri(url) if request else url

        return None

    def get_video_url(self, obj):
        request = self.context.get('request')

        if obj.video and obj.video.video:
            url = obj.video.video.url
            return request.build_absolute_uri(url) if request else url

        return None


class WorkshopSerializer(serializers.ModelSerializer):
    """
    Detail/create/update serializer.

    Important:
    - It intentionally does NOT include media_items.
    - Media are fetched only through /workshops/<id>/media/.
    - Action outputs are fetched only through /workshops/<id>/action-results/.
    """
    events = WorkshopEventSerializer(many=True, read_only=True)

    event_ids = serializers.PrimaryKeyRelatedField(
        queryset=Event.objects.all(),
        many=True,
        write_only=True,
        required=False,
        source='events'
    )

    requester_id = serializers.IntegerField(read_only=True)

    class Meta:
        model = Workshop
        fields = [
            'id',
            'workshop_name',
            'requester_id',
            'events',
            'event_ids',
            'created_at',
            'updated_at',
        ]

    def get_requester(self):
        request = self.context.get('request')

        if not request or not request.user.is_authenticated:
            raise serializers.ValidationError("Authentication is required.")

        requester = getattr(request.user, 'requester_profile', None)

        if requester is None:
            raise serializers.ValidationError("Only requesters can manage workshops.")

        return requester

    def validate(self, attrs):
        requester = self.get_requester()
        events = attrs.get('events', None)

        if self.instance is None and not events:
            raise serializers.ValidationError({
                'event_ids': 'At least one event is required.'
            })

        if events is not None:
            invalid_owner = []
            invalid_media_type = []

            for event in events:
                if event.requester_id != requester.pk:
                    invalid_owner.append(event.id)

                if not is_valid_workshop_event(event):
                    invalid_media_type.append(event.id)

            if invalid_owner:
                raise serializers.ValidationError({
                    'event_ids': f'These events do not belong to this requester: {invalid_owner}'
                })

            if invalid_media_type:
                raise serializers.ValidationError({
                    'event_ids': (
                        'Workshops can only use events with media_types photo, video, '
                        f'or both. These events are invalid: {invalid_media_type}'
                    )
                })

        return attrs

    def create(self, validated_data):
        requester = self.get_requester()
        events = validated_data.pop('events', [])

        workshop = Workshop.objects.create(
            requester=requester,
            **validated_data
        )

        workshop.events.set(events)
        return workshop

    def update(self, instance, validated_data):
        events = validated_data.pop('events', None)

        instance.workshop_name = validated_data.get(
            'workshop_name',
            instance.workshop_name
        )
        instance.save(update_fields=['workshop_name', 'updated_at'])

        if events is not None:
            instance.events.set(events)

            WorkshopMedia.objects.filter(
                workshop=instance
            ).exclude(
                event__in=events
            ).delete()

        return instance


class WorkshopListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for GET /workshops/.
    It keeps event metadata for the sidebar but never serializes media/action payloads.
    """
    events = WorkshopEventSerializer(many=True, read_only=True)
    requester_id = serializers.IntegerField(read_only=True)
    events_count = serializers.IntegerField(read_only=True)
    media_count = serializers.IntegerField(read_only=True)
    photo_count = serializers.IntegerField(read_only=True)
    video_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Workshop
        fields = [
            'id',
            'workshop_name',
            'requester_id',
            'events',
            'events_count',
            'media_count',
            'photo_count',
            'video_count',
            'created_at',
            'updated_at',
        ]


class WorkshopMediaAddItemSerializer(serializers.Serializer):
    submission_id = serializers.IntegerField()
    media_type = serializers.ChoiceField(choices=['photo', 'video'])


class WorkshopMediaBulkAddSerializer(serializers.Serializer):
    items = WorkshopMediaAddItemSerializer(many=True)

    def validate_items(self, items):
        if not items:
            raise serializers.ValidationError("At least one media item is required.")

        return items


class WorkshopActionResultSerializer(serializers.ModelSerializer):
    media_url = serializers.SerializerMethodField()
    event_title = serializers.SerializerMethodField()

    class Meta:
        model = WorkshopActionResult
        fields = [
            'id',
            'workshop',
            'event',
            'event_title',
            'workshop_media',
            'requester',
            'run_uid',
            'action_key',
            'media_kind',
            'media_name',
            'status',
            'media_url',
            'output_media_url',
            'output_media_storage_key',
            'output_media_kind',
            'output_media_mime_type',
            'output_media_size_bytes',
            'label_colors',
            'request_payload',
            'result_payload',
            'error_message',
            'started_at',
            'completed_at',
            'duration_ms',
            'created_at',
            'updated_at',
        ]
        read_only_fields = [
            'id',
            'workshop',
            'requester',
            'media_url',
            'event_title',
            'created_at',
            'updated_at',
        ]

    def get_media_url(self, obj):
        request = self.context.get('request')
        media = obj.workshop_media

        if not media:
            return None

        url = None

        if media.media_type == 'photo' and media.photo and media.photo.image:
            url = media.photo.image.url

        if media.media_type == 'video' and media.video and media.video.video:
            url = media.video.video.url

        if not url:
            return None

        return request.build_absolute_uri(url) if request else url

    def get_event_title(self, obj):
        if obj.event:
            return obj.event.title

        media = obj.workshop_media
        if media and media.event:
            return media.event.title

        return None