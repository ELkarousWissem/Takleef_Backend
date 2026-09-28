import os
import time
import uuid

from django.core.exceptions import ValidationError
from django.db import models

from mobicrowd.models.Users import Worker, User,Organization,OrganizationMembership
from django.utils import timezone
# Assuming a Photo model exists

class Event(models.Model):
    PHOTO = 'photo'
    VIDEO = 'video'
    TEXT = 'text'

    MEDIA_CHOICES = [
        (PHOTO, 'Photo'),
        (VIDEO, 'Video'),
        (TEXT, 'Text'),
    ]

    requester = models.ForeignKey('Requester', on_delete=models.CASCADE, related_name='events',null=True,blank=True)

    title = models.CharField(max_length=255)
    description = models.TextField()
    keywords = models.TextField(blank=True, default='')
    location = models.JSONField(default=dict, null=True, blank=True)
    CoverageArea = models.FloatField(max_length=255, null=True, blank=True)
    Polygon_area = models.JSONField(null=True, blank=True)

    # 🔄 old “cost” is now split:
    photo_reward = models.FloatField(null=True, blank=True)
    video_reward = models.FloatField(null=True, blank=True)

    numberOfPhotos = models.IntegerField(null=True, blank=True)
    max_photos_per_worker = models.IntegerField(default=5, null=True, blank=True)

    numberOfVideos = models.IntegerField(null=True, blank=True)
    max_videos_per_worker = models.IntegerField(default=5, null=True, blank=True)

    # TEXT
    text_reward = models.FloatField(null=True, blank=True)
    numberOfTexts = models.IntegerField(null=True, blank=True)
    max_texts_per_worker = models.IntegerField(default=5, null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    deadline = models.DateTimeField()
    startdate = models.DateTimeField()
    duration_hours = models.FloatField(null=True, blank=True)

    # 🆕 which media types are enabled
    media_types = models.JSONField(default=list, blank=True)

    joined_workers = models.ManyToManyField('Worker', through='EventWorker', related_name='joined_events', blank=True)
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='events'
    )
    organization_membership = models.ForeignKey(
            OrganizationMembership,
            on_delete=models.CASCADE,
            null=True,
            blank=True,
            related_name='organization_events'
        )
    @property
    def is_public(self):
        return self.organization_id is None

    @property
    def is_organization_event(self):
        return self.organization_id is not None

class EventWorker(models.Model):
    PENDING = 'PENDING'
    APPROVED = 'APPROVED'
    REJECTED = 'REJECTED'
    STATUS_CHOICES = [
        (PENDING, 'PENDING'),
        (APPROVED, 'APPROVED'),
        (REJECTED, 'Rejected'),
    ]

    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    worker = models.ForeignKey(Worker, on_delete=models.CASCADE)
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default=PENDING)

    # When the join request row was created. This is not the approval time.
    joined_at = models.DateTimeField(auto_now_add=True)

    # When the worker was actually approved for this event.
    approved_at = models.DateTimeField(null=True, blank=True, db_index=True)

    device_specs = models.JSONField(default=dict)

    class Meta:
        unique_together = ('event', 'worker')

    def approve(self, *, at=None) -> bool:
        """
        Mark this join request as approved and record the approval time once.

        Returns True when the row changed. Repeated approval calls are
        idempotent and do not replace the original approval timestamp.
        """
        if self.status == self.APPROVED and self.approved_at is not None:
            return False

        self.status = self.APPROVED
        self.approved_at = at or timezone.now()
        self.save(update_fields=['status', 'approved_at'])
        return True

class Photo(models.Model):
    image = models.ImageField(upload_to='photos',max_length=600)
    thumbnail = models.ImageField(upload_to='thumbs', max_length=600, blank=True, null=True)  # New field

    sizeimage = models.FloatField(default=0, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    original_metadata = models.JSONField(default=dict)  # Store metadata as a JSON object
    # submission = models.ForeignKey('Submission', on_delete=models.CASCADE, related_name='photos')
    caption = models.TextField(blank=True)
    extracted_text = models.TextField(blank=True)

class Video(models.Model):
    video = models.FileField(upload_to='videos', max_length=600)  # Store the video file
    duration = models.FloatField()  # Duration of the video in seconds
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp for when the video was created
    metadata = models.JSONField(default=dict)  # Store video metadata (e.g., resolution, format, codec)
    size = models.FloatField(blank=True, null=True)  # Size of the video file in MB
    thumbnail = models.ImageField(upload_to='thumbnails', max_length=600, blank=True, null=True)  # Thumbnail image for the video

    caption = models.TextField(blank=True)

    def get_video_duration(self):
        """
        Helper method to return the duration of the video in a human-readable format (HH:MM:SS).
        """
        seconds = self.duration
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    def save(self, *args, **kwargs):
        """
        Override save method to automatically compute the size of the video file.
        """
        # Compute video size (if not already set)
        if not self.size:
            if self.video:
                self.size = self.video.size / (1024 * 1024)  # Convert from bytes to MB
        super().save(*args, **kwargs)

class WorkerReward(models.Model):
    worker = models.ForeignKey('Worker', on_delete=models.CASCADE, related_name='worker_rewards')
    rewardPerSubmission = models.FloatField()  # ← now a simple float
    submission = models.ForeignKey('Submission', on_delete=models.CASCADE, related_name='worker_rewards')
    awarded_on = models.DateTimeField(auto_now_add=True)

class Submission(models.Model):
    PENDING = 'pending'
    APPROVED = 'approved'
    REFUSED = 'refused'
    STATUS_CHOICES = [
        (PENDING, 'Pending'),
        (APPROVED, 'Approved'),
        (REFUSED, 'Refused'),
    ]

    # Server-authoritative workflow state. Never accept this value from a client.
    FLOW_CREATED = 'created'
    FLOW_DECODING = 'decoding'
    FLOW_DECODED = 'decoded'
    FLOW_VALIDATING = 'validating'
    FLOW_VALIDATED = 'validated'
    FLOW_FINALIZING = 'finalizing'
    FLOW_FINALIZED = 'finalized'
    FLOW_REFUSED = 'refused'
    FLOW_CHOICES = [
        (FLOW_CREATED, 'Created'),
        (FLOW_DECODING, 'Decoding'),
        (FLOW_DECODED, 'Decoded'),
        (FLOW_VALIDATING, 'Validating'),
        (FLOW_VALIDATED, 'Validated'),
        (FLOW_FINALIZING, 'Finalizing'),
        (FLOW_FINALIZED, 'Finalized'),
        (FLOW_REFUSED, 'Refused'),
    ]

    worker = models.ForeignKey('Worker', on_delete=models.CASCADE, related_name='submissions_list')
    event = models.ForeignKey('Event', on_delete=models.CASCADE, related_name='event_submissions')
    status = models.CharField(max_length=100, choices=STATUS_CHOICES, default=PENDING)

    # Security state for ordered API execution and replay prevention.
    flow_stage = models.CharField(
        max_length=32,
        choices=FLOW_CHOICES,
        default=FLOW_CREATED,
        db_index=True,
        editable=False,
    )
    flow_nonce = models.UUIDField(default=uuid.uuid4, editable=False, db_index=True)
    flow_artifact_path = models.TextField(null=True, blank=True, editable=False)
    flow_artifact_digest = models.CharField(max_length=64, null=True, blank=True, editable=False)
    flow_validated_at = models.DateTimeField(null=True, blank=True, editable=False)
    flow_finalized_at = models.DateTimeField(null=True, blank=True, editable=False)

    location = models.JSONField(null=True, blank=True)
    photo = models.OneToOneField('Photo', on_delete=models.CASCADE, null=True, blank=True)
    video = models.OneToOneField('Video', on_delete=models.CASCADE, null=True, blank=True)
    text = models.TextField(null=True, blank=True)
    message = models.TextField(blank=True, null=True)
    worker_message = models.TextField(blank=True, null=True)
    worker_message_read = models.BooleanField(default=False)

    def __str__(self):
        user = getattr(self.worker, 'user', None)
        label = getattr(user, 'email', None) or getattr(user, 'username', None) or self.worker_id
        return f"{label} - {self.event.title} - {self.status}/{self.flow_stage}"

PHOTO = 'photo'
VIDEO = 'video'

def report_upload_to(instance, filename):
    wid = getattr(instance, "worker_id", None) or "anon"
    eid = getattr(instance, "event_id", None) or "none"
    base, ext = os.path.splitext(filename or "")
    ext = ext or ".bin"
    ts = int(time.time() * 1000)
    return f"reports/w{wid}/e{eid}/{ts}{ext}"
def report_upload_to_thumb(instance, filename):
    wid = getattr(instance, "worker_id", None) or "anon"
    eid = getattr(instance, "event_id", None) or "none"
    base, ext = os.path.splitext(filename or "")
    ext = ext or ".bin"
    ts = int(time.time() * 1000)
    return f"thumbreports/w{wid}/e{eid}/{ts}{ext}"

from django.db import models
from django.utils import timezone

class SubmissionReport(models.Model):
    TYPE_CHOICES = (
        (PHOTO, 'photo'),
        (VIDEO, 'video'),
    )

    worker = models.ForeignKey(
        Worker,
        on_delete=models.CASCADE,
        related_name='reports'
    )
    submission = models.ForeignKey(
        Submission,
        on_delete=models.CASCADE,
        related_name='reports',
        null=True,          # keep nullable first if you already have old rows
        blank=True
    )
    event = models.ForeignKey(
        Event,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reports'
    )
    type = models.CharField(max_length=16, choices=TYPE_CHOICES, default=PHOTO)
    comment = models.TextField()
    file = models.FileField(upload_to=report_upload_to, max_length=600, null=True, blank=True)
    thumbnail = models.FileField(upload_to=report_upload_to_thumb, max_length=600, null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    submission_message = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(
                fields=['worker', 'submission'],
                name='uniq_report_per_worker_submission'
            )
        ]

    def __str__(self):
        return f"Report #{self.id} by {self.worker_id} ({self.type})"



class UserUploadLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    submission = models.ForeignKey(Submission, on_delete=models.CASCADE)
    size_mb = models.FloatField(null=True, blank=True)


class Workshop(models.Model):
    requester = models.ForeignKey(
        'Requester',
        on_delete=models.CASCADE,
        related_name='workshops'
    )

    workshop_name = models.CharField(
        max_length=255,
        blank=True,
        default=''
    )

    events = models.ManyToManyField(
        'Event',
        related_name='workshops',
        blank=True
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.workshop_name or f"Workshop #{self.id}"

class WorkshopMedia(models.Model):
    PHOTO = 'photo'
    VIDEO = 'video'

    MEDIA_CHOICES = [
        (PHOTO, 'Photo'),
        (VIDEO, 'Video'),
    ]

    SELECTED = 'selected'
    PROCESSED = 'processed'
    REMOVED = 'removed'

    ACTION_STATUS_CHOICES = [
        (SELECTED, 'Selected'),
        (PROCESSED, 'Processed'),
        (REMOVED, 'Removed'),
    ]

    workshop = models.ForeignKey(
        Workshop,
        on_delete=models.CASCADE,
        related_name='media_items'
    )

    event = models.ForeignKey(
        'Event',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='workshop_media_items'
    )

    submission = models.ForeignKey(
        'Submission',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='workshop_media_items'
    )

    media_type = models.CharField(
        max_length=10,
        choices=MEDIA_CHOICES
    )

    photo = models.ForeignKey(
        'Photo',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='workshop_media_items'
    )

    video = models.ForeignKey(
        'Video',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='workshop_media_items'
    )

    # Snapshots keep the trace even if FK becomes null later.
    original_event_id = models.IntegerField(null=True, blank=True)
    original_submission_id = models.IntegerField(null=True, blank=True)
    original_photo_id = models.IntegerField(null=True, blank=True)
    original_video_id = models.IntegerField(null=True, blank=True)

    action_status = models.CharField(
        max_length=20,
        choices=ACTION_STATUS_CHOICES,
        default=SELECTED
    )

    action_payload = models.JSONField(default=dict, blank=True)

    added_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-added_at']
        constraints = [
            models.UniqueConstraint(
                fields=['workshop', 'submission', 'media_type'],
                name='uniq_workshop_submission_media_type'
            )
        ]

    def clean(self):
        if self.media_type not in [self.PHOTO, self.VIDEO]:
            raise ValidationError("Only photo and video media are allowed in workshops.")

        if self.submission:
            self.event = self.submission.event
            self.original_submission_id = self.submission_id
            self.original_event_id = self.submission.event_id

            if self.media_type == self.PHOTO:
                if not self.submission.photo_id:
                    raise ValidationError("This submission does not contain a photo.")

                self.photo = self.submission.photo
                self.video = None
                self.original_photo_id = self.submission.photo_id
                self.original_video_id = None

            if self.media_type == self.VIDEO:
                if not self.submission.video_id:
                    raise ValidationError("This submission does not contain a video.")

                self.video = self.submission.video
                self.photo = None
                self.original_video_id = self.submission.video_id
                self.original_photo_id = None

        if self.event and self.workshop:
            if self.event.requester_id != self.workshop.requester_id:
                raise ValidationError("This media does not belong to this workshop requester.")

            if not self.workshop.events.filter(id=self.event_id).exists():
                raise ValidationError("This media event is not selected in this workshop.")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Workshop {self.workshop_id} - {self.media_type} - Submission {self.original_submission_id}"

import uuid
from django.db import models
from django.utils import timezone


class WorkshopActionResult(models.Model):
    STATUS_SUCCESS = 'success'
    STATUS_ERROR = 'error'
    STATUS_SKIPPED = 'skipped'

    STATUS_CHOICES = [
        (STATUS_SUCCESS, 'Success'),
        (STATUS_ERROR, 'Error'),
        (STATUS_SKIPPED, 'Skipped'),
    ]

    MEDIA_PHOTO = 'photo'
    MEDIA_VIDEO = 'video'
    MEDIA_MIXED = 'mixed'
    MEDIA_TEXT = 'text'

    MEDIA_KIND_CHOICES = [
        (MEDIA_PHOTO, 'Photo'),
        (MEDIA_VIDEO, 'Video'),
        (MEDIA_MIXED, 'Mixed'),
        (MEDIA_TEXT, 'Text'),
    ]

    workshop = models.ForeignKey(
        Workshop,
        on_delete=models.CASCADE,
        related_name='action_results'
    )

    # New explicit event FK for grouping results by workshop + event.
    # For one image/video action row, this is copied from workshop_media.event.
    # For mixed/global actions, it can stay null.
    event = models.ForeignKey(
        Event,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='workshop_action_results'
    )

    workshop_media = models.ForeignKey(
        WorkshopMedia,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='action_results'
    )

    requester = models.ForeignKey(
        'Requester',
        on_delete=models.CASCADE,
        related_name='workshop_action_results'
    )

    run_uid = models.UUIDField(default=uuid.uuid4, db_index=True)
    action_key = models.CharField(max_length=64, db_index=True)

    media_kind = models.CharField(
        max_length=16,
        choices=MEDIA_KIND_CHOICES,
        default=MEDIA_PHOTO
    )
    media_name = models.CharField(max_length=512, blank=True, default='')

    status = models.CharField(
        max_length=16,
        choices=STATUS_CHOICES,
        default=STATUS_SUCCESS,
        db_index=True
    )

    # Original media URL is still resolved from workshop_media in the serializer.
    # Generated media is stored in S3 and referenced here directly.
    output_media_url = models.URLField(max_length=2048, blank=True, default='')
    output_media_storage_key = models.CharField(max_length=1024, blank=True, default='')
    output_media_kind = models.CharField(max_length=16, blank=True, default='')
    output_media_mime_type = models.CharField(max_length=128, blank=True, default='')
    output_media_size_bytes = models.PositiveIntegerField(default=0)

    # Keep the reusable visual legend directly queryable.
    # Example: [{"label":"laptop","color_hex":"#ffdc00","instance_count":1}, ...]
    label_colors = models.JSONField(default=list, blank=True)

    # Keep these as compact metadata only. Do not store base64 or full raw model output here.
    request_payload = models.JSONField(default=dict, blank=True)
    result_payload = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True, default='')

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_ms = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-completed_at', '-created_at']
        indexes = [
            models.Index(fields=['workshop', 'action_key']),
            models.Index(fields=['workshop', 'run_uid']),
            models.Index(fields=['workshop', 'event', 'action_key']),
            models.Index(fields=['workshop', 'event', 'run_uid']),
            models.Index(fields=['workshop_media', 'action_key']),
        ]

    def __str__(self):
        return f"Workshop {self.workshop_id} - Event {self.event_id} - {self.action_key} - {self.status}"