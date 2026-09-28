import secrets
from datetime import timedelta

from django.conf import settings
from django.db import models
from django.utils import timezone
class DeviceToken(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    device_uid = models.CharField(max_length=64, unique=True)                 # 1 row per device
    token = models.CharField(max_length=512, unique=True, null=True, blank=True)  # push token, may be null at login

    platform = models.CharField(max_length=16, choices=[("ios","iOS"),("android","Android"),("web","Web")])
    timezone = models.CharField(max_length=64, blank=True, default="")
    is_active = models.BooleanField(default=True)

    kind = models.CharField(max_length=16, choices=[("APP","App"),("DASHBOARD","Dashboard")])
    session_active = models.BooleanField(default=False)
    session_sid = models.CharField(max_length=64, blank=True, default="")
    session_last_seen = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(auto_now=True)



class LoginTakeoverChallenge(models.Model):
    METHOD_EMAIL_OTP = "EMAIL_OTP"
    METHOD_ACTIVE_SESSION = "ACTIVE_SESSION"

    STATUS_PENDING = "PENDING"
    STATUS_APPROVED = "APPROVED"
    STATUS_DENIED = "DENIED"
    STATUS_USED = "USED"
    STATUS_EXPIRED = "EXPIRED"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    challenge_id = models.CharField(max_length=96, unique=True, default=secrets.token_urlsafe)
    method = models.CharField(
        max_length=32,
        choices=[
            (METHOD_EMAIL_OTP, "Email OTP"),
            (METHOD_ACTIVE_SESSION, "Active Session Approval"),
        ],
    )

    kind = models.CharField(max_length=16, choices=[("APP", "App"), ("DASHBOARD", "Dashboard")])
    platform = models.CharField(max_length=16, choices=[("ios", "iOS"), ("android", "Android"), ("web", "Web")])

    incoming_device_uid = models.CharField(max_length=128)
    incoming_device_label = models.CharField(max_length=160, blank=True, default="")

    otp_hash = models.CharField(max_length=128, blank=True, default="")
    otp_attempts = models.PositiveSmallIntegerField(default=0)

    status = models.CharField(max_length=16, default=STATUS_PENDING)

    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    decided_at = models.DateTimeField(null=True, blank=True)

    approved_by_device_uid = models.CharField(max_length=128, blank=True, default="")

    class Meta:
        indexes = [
            models.Index(fields=["user", "status", "created_at"]),
            models.Index(fields=["challenge_id"]),
            models.Index(fields=["incoming_device_uid"]),
        ]

    def is_expired(self):
        return timezone.now() >= self.expires_at

    @classmethod
    def new_expiry(cls):
        return timezone.now() + timedelta(minutes=10)
class Notification(models.Model):
    HIGH, NORMAL = "high", "normal"
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="notifications")
    event_type = models.CharField(max_length=64)          # e.g., join_request.approved
    title = models.CharField(max_length=140)
    body = models.TextField(blank=True)
    payload = models.JSONField(default=dict)              # deep-link, context ids, etc.
    priority = models.CharField(max_length=8, default=NORMAL)
    created_at = models.DateTimeField(auto_now_add=True)
    read_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["user","created_at"]),
            models.Index(fields=["user","read_at"]),
        ]
