
from rest_framework.decorators import action
from mobicrowd.models.notifications import DeviceToken, Notification
from mobicrowd.serializers.deviceCamSpecsSerializers import DeviceTokenSerializer, NotificationSerializer
from rest_framework.pagination import PageNumberPagination

# views.py
from django.db import transaction
from django.utils import timezone
from rest_framework import mixins, viewsets, permissions, status
from rest_framework.response import Response
from zoneinfo import ZoneInfo

class DeviceTokenViewSet(mixins.ListModelMixin,
                         mixins.DestroyModelMixin,
                         mixins.UpdateModelMixin,
                         viewsets.GenericViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = DeviceTokenSerializer

    ALLOWED_PLATFORMS = {"ios", "android", "web"}

    def get_queryset(self):
        return DeviceToken.objects.filter(user=self.request.user).order_by("-created_at")

    def _normalize_platform(self, v: str) -> str:
        v = (v or "").strip().lower()
        if v in ("ios", "apple", "iphone", "ipad"):
            return "ios"
        if v == "android":
            return "android"
        if v in ("web", "browser"):
            return "web"
        return "android"

    def _valid_tz(self, tzname: str | None) -> str | None:
        tzname = (tzname or "").strip()
        if not tzname:
            return None
        try:
            ZoneInfo(tzname)
            return tzname
        except Exception:
            return None

    def create(self, request, *args, **kwargs):
        token = (request.data.get("token") or "").strip()
        device_uid = (request.data.get("device_uid") or "").strip()
        platform = self._normalize_platform(request.data.get("platform") or "")
        tz = self._valid_tz(request.data.get("timezone"))

        if not token:
            return Response({"token": ["This field is required."]}, status=400)
        if not device_uid:
            return Response({"device_uid": ["This field is required."]}, status=400)

        kind = "DASHBOARD" if platform == "web" else "APP"

        with transaction.atomic():
            obj, created = DeviceToken.objects.update_or_create(
                device_uid=device_uid,
                defaults={
                    "user": request.user,
                    "token": token,
                    "platform": platform,
                    "timezone": tz or "",
                    "is_active": True,
                    "kind": kind,  # ✅ keep kind consistent
                    "last_seen": timezone.now(),
                },
            )

            # keep token globally unique (clean old rows if any)
            DeviceToken.objects.filter(token=token).exclude(pk=obj.pk).delete()

        return Response({"ok": True, "created": created, "device": self.get_serializer(obj).data},
                        status=201 if created else 200)

    def destroy(self, request, *args, **kwargs):
        obj = self.get_object()
        if obj.user_id != request.user.id:
            return Response(status=status.HTTP_404_NOT_FOUND)
        obj.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class TenPerPage(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 50


class NotificationViewSet(mixins.ListModelMixin, viewsets.GenericViewSet):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = NotificationSerializer
    pagination_class = TenPerPage  # <-- add this

    def get_queryset(self):
        return Notification.objects.filter(user=self.request.user).order_by("-created_at")

    @action(detail=False, methods=["get"])
    def unread_count(self, request):
        c = Notification.objects.filter(user=request.user, read_at__isnull=True).count()
        return Response({"unread": c})

    @action(detail=True, methods=["post"])
    def mark_read(self, request, pk=None):
        n = Notification.objects.filter(id=pk, user=request.user).first()
        if not n:
            return Response(status=status.HTTP_404_NOT_FOUND)
        if not n.read_at:
            n.read_at = timezone.now()
            n.save(update_fields=["read_at"])
        return Response({"ok": True})

    @action(detail=False, methods=["post"])
    def mark_all_read(self, request):
        Notification.objects.filter(user=request.user, read_at__isnull=True).update(read_at=timezone.now())
        return Response({"ok": True})