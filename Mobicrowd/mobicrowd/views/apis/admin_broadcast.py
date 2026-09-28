# mobicrowd/views/apis/admin_broadcast.py
from rest_framework import serializers, permissions, status
from rest_framework.views import APIView
from rest_framework.response import Response

from mobicrowd.models.Users import User
from mobicrowd.tasks import send_broadcast
from mobicrowd.notify import notify_user   # <-- import this
from rest_framework import permissions
from django.utils import timezone
from math import ceil
class BroadcastSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=200)
    body = serializers.CharField(allow_blank=True, default="")
    priority = serializers.ChoiceField(choices=[("normal","normal"),("high","high")], default="high")
    audience = serializers.ChoiceField(choices=[("all","all"),("requesters","requesters"),("workers","workers")], default="all")
    payload = serializers.JSONField(required=False)
    schedule_at = serializers.DateTimeField(required=False, allow_null=True)
# mobicrowd/permissions.py

class IsAdminRole(permissions.BasePermission):
    def has_permission(self, request, view):
        user = request.user
        return bool(user and user.is_authenticated and getattr(user, "role", "") == "Admin")



class BroadcastView(APIView):
    permission_classes = [IsAdminRole,permissions.IsAuthenticated]

    def post(self, request):
        s = BroadcastSerializer(data=request.data)
        s.is_valid(raise_exception=True)

        # audience resolution (unchanged) …
        audience = s.validated_data["audience"]
        qs = User.objects.filter(is_active=True)
        if audience == "requesters":
            qs = qs.filter(role="Requester")
        elif audience == "workers":
            qs = qs.filter(role="Worker")
        user_ids = list(qs.values_list("id", flat=True))
        if not user_ids:
            return Response({"ok": True, "queued": False, "reason": "no recipients"}, status=200)
        body = s.validated_data.get("body", "")
        base_payload = s.validated_data.get("payload") or {"type": "system.maintenance"}
        payload = {
            **base_payload,
            "body_for_ui": body
        }

        kw = dict(
            user_ids=user_ids,
            title=s.validated_data["title"],
            body=body,  # still keep body at top-level for backward compat
            priority=s.validated_data.get("priority", "high"),
            payload=payload,
        )

        # --- Authoritative scheduling here ---
        schedule_at = s.validated_data.get("schedule_at")  # aware dt if USE_TZ=True
        now = timezone.now()
        if schedule_at and schedule_at > now:
            # Use countdown to avoid TZ surprises
            seconds = max(1, ceil((schedule_at - now).total_seconds()))
            task = send_broadcast.apply_async(kwargs=kw, countdown=seconds)
            return Response(
                {"ok": True, "queued": True, "task_id": task.id,
                 "scheduled_for": schedule_at.isoformat(), "countdown": seconds},
                status=200
            )
        else:
            task = send_broadcast.delay(**kw)
            return Response({"ok": True, "queued": True, "task_id": task.id, "scheduled_for": "now"}, status=200)


class BroadcastTestView(APIView):
    permission_classes = [IsAdminRole]

    class TestSerializer(serializers.Serializer):
        title = serializers.CharField(max_length=200)
        body = serializers.CharField(allow_blank=True, default="")
        priority = serializers.ChoiceField(choices=[("normal","normal"),("high","high")], default="high")
        payload = serializers.JSONField(required=False)
        use_celery = serializers.BooleanField(required=False, default=False)
        schedule_at = serializers.DateTimeField(required=False, allow_null=True)

    def post(self, request):
        s = self.TestSerializer(data=request.data)
        s.is_valid(raise_exception=True)

        title     = s.validated_data["title"]
        body      = s.validated_data.get("body", "")
        priority  = s.validated_data.get("priority", "high")
        payload   = s.validated_data.get("payload") or {"type": "system.maintenance"}
        use_celery = s.validated_data.get("use_celery", False)
        eta = s.validated_data.get("schedule_at")  # aware dt if USE_TZ=True
        payload = {
            **payload,
            "body_for_ui": body
        }
        # If we want to exercise the exact same path as the real broadcast,
        # reuse your Celery task with a single-recipient list.
        if use_celery:
            kwargs = dict(
                user_ids=[request.user.id],
                title=title, body=body, payload=payload, priority=priority
            )
            now = timezone.now()
            if eta and eta > now:
                seconds = max(1, ceil((eta - now).total_seconds()))
                task = send_broadcast.apply_async(kwargs=kwargs, countdown=seconds)  # <-- like BroadcastView
                return Response(
                    {"ok": True, "queued": True, "task_id": task.id,
                     "scheduled_for": eta.isoformat(), "countdown": seconds},
                    status=200
                )
            else:
                task = send_broadcast.delay(**kwargs)
                return Response({"ok": True, "queued": True, "task_id": task.id}, status=200)

        # Otherwise send immediately to the admin (no Celery).
        notify_user(
            user_id=request.user.id,
            event_type="system.maintenance.test",
            title=title,
            body=body,
            payload=payload,
            priority=priority,
        )
        return Response({"ok": True, "sent_to": request.user.id}, status=200)