import base64
import io
import mimetypes
import os
import subprocess
import tempfile
import uuid
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image, ImageOps
from django.core.files.base import ContentFile
from rest_framework import serializers
from ultralytics import YOLO

from mobicrowd.models.Users import Worker
from mobicrowd.models.submisson import EventWorker, Photo, Video, Event, Submission, WorkerReward, SubmissionReport, \
    VIDEO, PHOTO
from mobicrowd.serializers.usersSerializers import WorkerSerializer, RequesterSerializer
from mobicrowd.video_tasks import blur_video_faces_sparse
from mobicrowd.views.apis.submission_flow.task_understanding import get_task_keywords

# Serializer for EventWorker
class EventWorkerSerializer(serializers.ModelSerializer):
    event_title = serializers.CharField(source='event.title')
    worker_id = serializers.CharField(source='worker.user.id')
    worker_name = serializers.CharField(source='worker.user.fullName', read_only=True)

    class Meta:
        model = EventWorker
        fields = '__all__'


# Serializer for Event
from rest_framework import serializers

class EventSerializer(serializers.ModelSerializer):
    joined_workers = serializers.SerializerMethodField()
    requester_organization_name = serializers.CharField(source='requester.organization_name', read_only=True)
    requester_location = serializers.CharField(source='requester.location', read_only=True)
    submissions_count = serializers.SerializerMethodField()
    requester_name = serializers.SerializerMethodField()
    organization_requester_name = serializers.SerializerMethodField()
    created_by_display = serializers.SerializerMethodField()

    # Admin-only derived report counters.
    # These are not database columns; list APIs can annotate them efficiently.
    reported_photos_count = serializers.SerializerMethodField()
    reported_videos_count = serializers.SerializerMethodField()

    class Meta:
        model = Event
        fields = '__all__'
        extra_kwargs = {
                            'requester': {'required': False, 'allow_null': True},
                            'organization_membership': {'required': False, 'allow_null': True},
                            'organization': {'required': False, 'allow_null': True},}

    def _is_admin_request(self):
        request = self.context.get('request')
        user = getattr(request, 'user', None)

        return bool(
            user
            and getattr(user, 'is_authenticated', False)
            and (
                getattr(user, 'role', '') == 'Admin'
                or getattr(user, 'is_superuser', False)
            )
        )

    def get_fields(self):
        """
        Report counters are admin-only API attributes.

        For non-admin requests the fields are removed completely from the
        serialized representation instead of being exposed as zero/null.
        """
        fields = super().get_fields()

        if not self._is_admin_request():
            fields.pop('reported_photos_count', None)
            fields.pop('reported_videos_count', None)

        return fields

    def get_reported_photos_count(self, obj):
        annotated_value = getattr(obj, 'reported_photos_count', None)

        if annotated_value is not None:
            return int(annotated_value)

        # Fallback for a single admin retrieve/update response that was not
        # annotated by a list queryset.
        return obj.reports.filter(type=PHOTO).count()

    def get_reported_videos_count(self, obj):
        annotated_value = getattr(obj, 'reported_videos_count', None)

        if annotated_value is not None:
            return int(annotated_value)

        return obj.reports.filter(type=VIDEO).count()

    def _apply_keywords_if_needed(self, validated_data, instance=None):
        """
        Shared keyword logic for create/update.

        Rules:
        - If caller provided non-empty keywords -> keep them.
        - If no description in payload during update -> leave keywords untouched.
        - If description is empty -> set keywords = "".
        - Else extract keywords from description (fail-safe on LLM error).
        """
        # # 1) If client explicitly provided keywords and it's non-empty, keep it
        # provided = (validated_data.get("keywords") or "").strip()
        # if provided:
        #     validated_data["keywords"] = provided
        #     return validated_data

        # 2) On update/PATCH: if description not being updated, don't touch keywords
        if instance is not None and "description" not in validated_data:
            return validated_data

        # 3) Description-driven keyword extraction
        desc = (validated_data.get("description") or "").strip()

        if not desc:
            # create with empty desc OR update setting desc empty
            validated_data["keywords"] = ""
            return validated_data

        # Optional optimization: if update and description unchanged, keep existing keywords
        if instance is not None:
            current_desc = (instance.description or "").strip()
            if desc == current_desc and "keywords" not in validated_data:
                return validated_data

        # 4) LLM extraction (fail-safe)
        try:
            validated_data["keywords"] = (get_task_keywords(desc) or "").strip()
            print("********************************************", validated_data["keywords"])
        except Exception:
            validated_data["keywords"] = ""

        return validated_data

    def create(self, validated_data):
        validated_data = self._apply_keywords_if_needed(validated_data, instance=None)
        return super().create(validated_data)

    def update(self, instance, validated_data):
        print(instance)
        validated_data = self._apply_keywords_if_needed(validated_data, instance=instance)
        print("---------------------------------------------------", validated_data)
        return super().update(instance, validated_data)

    def get_joined_workers(self, obj):
        event_workers = EventWorker.objects.filter(event=obj, status=EventWorker.APPROVED)
        return EventWorkerSerializer(event_workers, many=True).data

    def get_submissions_count(self, obj):
        return obj.event_submissions.count()

    def get_requester_name(self, obj):
            # Event public
            if obj.organization_id is None and obj.requester and getattr(obj.requester, 'user', None):
                return obj.requester.user.fullName
            return None

    def get_requester_organization_name(self, obj):
            # Event organisation
            if obj.organization:
                return getattr(obj.organization, 'name', None) or getattr(obj.organization, 'organisationName', None)
            return None

    def get_organization_requester_name(self, obj):
        if obj.organization_id is None:
            return None

        # requester membre de l'organisation
        if obj.organization_membership and getattr(obj.organization_membership, 'user', None):
            return obj.organization_membership.user.fullName

        # représentant
        if obj.organization and getattr(obj.organization, 'representative', None):
            representative = obj.organization.representative
            if getattr(representative, 'user', None):
                return representative.user.fullName

        return None
    def get_created_by_display(self, obj):
        # Event public
        if obj.organization_id is None:
            if obj.requester and getattr(obj.requester, 'user', None):
                return obj.requester.user.fullName
            return None

        # Event organisation créé par membre requester
        if obj.organization_membership and getattr(obj.organization_membership, 'user', None):
            return obj.organization_membership.user.fullName

        # Event organisation créé par représentant
        if obj.organization and getattr(obj.organization, 'representative', None):
            representative = obj.organization.representative
            if getattr(representative, 'user', None):
                return representative.user.fullName

        return None
    def to_representation(self, instance):
         data = super().to_representation(instance)

         data['requester_name'] = self.get_requester_name(instance)
         data['requester_organization_name'] = self.get_requester_organization_name(instance)
         data['organization_requester_name'] = self.get_organization_requester_name(instance)
         data['created_by_display'] = self.get_created_by_display(instance)

         return data

# Serializer for Photo
class PhotoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Photo
        fields = '__all__'


# Serializer for Photo in Submission for Worker
class PhotoSerializerForSubmissionForWorker(serializers.ModelSerializer):
    class Meta:
        model = Photo
        fields = ['image', 'thumbnail', 'created_at']


# Serializer for Video
class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = '__all__'


# Submission Serializer (includes both photo and video)
class SubmissionSerializer(serializers.ModelSerializer):
    # Nesting the Photo and Video serializers
    photo = PhotoSerializer(read_only=True, required=False)  # Include photo if exists
    video = VideoSerializer(read_only=True, required=False)  # Include video if exists

    class Meta:
        model = Submission
        fields = '__all__' # Include necessary fields

    def validate_text(self, value):
        """Normalize optional text media so statistics never count blank text."""
        if value is None:
            return None
        return value.strip() or None
class SubmissionStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = Submission
        fields = ["id", "status", "message"]  # only what you need to patch/return

# Submission Serializer for Worker (includes only status and media related fields)
class SubmissionSerializerForWorker(serializers.ModelSerializer):
    photos = PhotoSerializerForSubmissionForWorker(many=True, read_only=True)
    video = VideoSerializer(read_only=True)  # Include video as well

    class Meta:
        model = Submission
        fields = ['status', 'photos', 'video', 'text']


class SubmissionReportSerializer(serializers.ModelSerializer):
    worker_name = serializers.CharField(
        source='worker.user.fullName',
        read_only=True
    )
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = SubmissionReport
        fields = [
            'id',
            'event',
            'type',
            'comment',
            'file_url',
            'thumbnail',
            'worker',
            'worker_name',
            'created_at',
            'submission_message'
        ]

    def get_file_url(self, obj):
      request = self.context.get('request')
      if obj.file and hasattr(obj.file, 'url'):
          url = obj.file.url
          return request.build_absolute_uri(url) if request else url
      return None
# Worker Reward Serializer
class WorkerRewardSerializer(serializers.ModelSerializer):
    worker = WorkerSerializer(read_only=True)
    submission = SubmissionSerializer(read_only=True)

    class Meta:
        model = WorkerReward
        fields = '__all__'


# Worker Event Info Serializer (to get the event-specific worker information)
class WorkerEventInfoSerializer(serializers.Serializer):
    worker_id = serializers.CharField(source='worker.user.id')
    full_name = serializers.CharField(source='worker.user.fullName')
    device_specs = serializers.SerializerMethodField()
    approved_submissions_count = serializers.SerializerMethodField()

    def get_device_specs(self, obj):
        event_worker = EventWorker.objects.filter(worker=obj.worker, event=obj.event, status=EventWorker.APPROVED).first()
        if event_worker:
            return event_worker.device_specs
        return {}

    def get_approved_submissions_count(self, obj):
        return Submission.objects.filter(worker=obj.worker, event=obj.event, status=Submission.APPROVED).count()


# Submission Report Serializer (for managing submission reports)