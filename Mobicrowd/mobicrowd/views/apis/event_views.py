from django.db.models import Q, Exists, OuterRef, Subquery, Count
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.timezone import localtime
from rest_framework import generics, permissions, status

from django.conf import settings
from .notify_helpers import make_payload, notify_admins, notify_workers, notify_event_subscribers
from ...authentication.email_sending import send_reject_join_request_email, send_approve_join_request_email
from ...models.submisson import Event, Photo, EventWorker, Submission, SubmissionReport, VIDEO, PHOTO
from ...models.Users import Worker, User, Organization, OrganizationMembership
from datetime import datetime
from rest_framework.response import Response
from rest_framework.views import APIView
from mobicrowd.serializers.submissionSerializers import EventWorkerSerializer, EventSerializer, \
    SubmissionReportSerializer
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from django_filters import rest_framework as filters
from django_filters.rest_framework import DjangoFilterBackend
from mobicrowd.notify_tz import tz_map_for_users, fmt_with_map, iso_utc

from ...notify import notify_user
# views.py
from rest_framework import generics, permissions
from rest_framework.response import Response
from mobicrowd.models.submisson import Event, EventWorker
from mobicrowd.views.apis.notify_helpers import notify_admins, make_payload  # adjust import path
import logging
from mobicrowd.tasks import schedule_event_reminders, schedule_worker_inactivity_check
from zoneinfo import ZoneInfo
logger = logging.getLogger(__name__)
from zoneinfo import ZoneInfo  # For timezone handling
import requests
from django.conf import settings
from django.core.cache import cache
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.response import Response
from rest_framework.throttling import SimpleRateThrottle
from mobicrowd.models.Users import User, Requester, Organization, OrganizationMembership, OrganizationLicenceKey

from django.db.models import OuterRef, Subquery, Value, CharField, Q, F
from django.db.models.functions import Coalesce
from django.utils import timezone
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"


def _is_admin_user(user):
    return bool(
        getattr(user, "role", "") == "Admin"
        or getattr(user, "is_superuser", False)
    )


def _with_admin_report_counts(queryset, user):
    """
    Annotate per-event report counters only for admin event-list requests.

    The values are derived from SubmissionReport through Event.reports and are
    not persisted on Event, so there is no counter state to keep in sync.
    """
    if not _is_admin_user(user):
        return queryset

    return queryset.annotate(
        reported_photos_count=Count(
            "reports",
            filter=Q(reports__type=PHOTO),
            distinct=True,
        ),
        reported_videos_count=Count(
            "reports",
            filter=Q(reports__type=VIDEO),
            distinct=True,
        ),
    )


def can_manage_event(user, event):
    if _is_admin_user(user):
        return True

    # Public event: only its requester can manage it
    if event.organization_id is None:
        return bool(
            event.requester
            and event.requester.user_id == user.id
        )

    # Organization event created by a requester membership
    membership = event.organization_membership

    if (
        membership
        and membership.user_id == user.id
        and membership.role == "requester"
        and membership.status == "active"
    ):
        return True

    # Organization event created by its representative.
    # These events intentionally have organization_membership=NULL.
    requester = Requester.objects.filter(user=user).first()

    return bool(
        requester
        and event.organization
        and event.organization.representative == requester
    )


class EventListView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = EventSerializer

    def get_queryset(self):
        qs = Event.objects.filter(organization__isnull=True)
        return _with_admin_report_counts(qs, self.request.user)



class UpcomingEventsListAPIView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = EventSerializer

    def get_queryset(self):
        qs = Event.objects.filter(
            deadline__gt=timezone.now(),
            organization__isnull=True,
        )
        return _with_admin_report_counts(qs, self.request.user)


class PastEventsListAPIView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = EventSerializer

    def get_queryset(self):
        qs = Event.objects.filter(
            deadline__lt=timezone.now(),
            organization__isnull=True,
        )
        return _with_admin_report_counts(qs, self.request.user)


def fmt_in_tz(dt, tzname: str | None = None) -> str:
    """
    Format a timezone-aware datetime in a given IANA tz name (e.g. 'Africa/Tunis'),
    including the zone abbreviation in the string. Falls back to server localtime.
    """
    tzname = tzname or getattr(settings, "TIME_ZONE", "UTC")
    try:
        return timezone.localtime(dt, ZoneInfo(tzname)).strftime("%b %d, %H:%M %Z")
    except Exception:
        # Fallback to current active timezone if tzname is invalid/missing
        return timezone.localtime(dt).strftime("%b %d, %H:%M %Z")


class EventCreateView(generics.CreateAPIView):
    permission_classes = [permissions.IsAuthenticated]
    queryset = Event.objects.all()
    serializer_class = EventSerializer

    def create(self, request, *args, **kwargs):
        requester = Requester.objects.filter(user=request.user).first()
        if not requester:
            return Response(
                {"error": "Only requester can create public event"},
                status=status.HTTP_403_FORBIDDEN,
            )

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer, requester=requester)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer, requester):
        event: Event = serializer.save(
            requester=requester,
            organization=None,
            organization_membership=None,
        )

        # --- A) Server-authoritative reminders ---
        schedule_event_reminders.delay(event_id=event.id)

        # --- B) Notify Admins if the creator is a Requester ---
        creator_role = getattr(self.request.user, "role", None)
        if creator_role == "Requester":
            rq = event.requester
            requester_name = rq.user.get_full_name()
            requester_email = rq.user.email

            title_admin = "New event awaiting review"
            body_admin = f"{requester_name} ({requester_email}) created “{event.title}”."

            payload_admin = make_payload(
                type="event.created",
                event_id=event.id,
                requester_id=rq.user_id,
                status="created",
                body_for_ui=body_admin,
            )

            notify_admins(
                event_type="event.created",
                title=title_admin,
                body=body_admin,
                payload=payload_admin,
                priority="high",
            )

        # --- C) Notify Workers (discovery feed) ---
        # 1) Gather audience and their timezones
        audience_qs = Worker.objects.select_related("user").filter(user__is_active=True)
        user_ids = list(audience_qs.values_list("user_id", flat=True))
        tzmap = tz_map_for_users(user_ids)

        # 2) UI body (short and sweet; UI will read dates from payload)
        title_workers = f"New task published : {event.title}"

        # Building reward text for UI based on media types
        reward_txts = []
        if 'photo' in event.media_types:
            reward_txts.append(f"${event.photo_reward:,.2f} per photo")
        if 'video' in event.media_types:
            reward_txts.append(f"${event.video_reward:,.2f} per video")
        if 'text' in event.media_types:
            reward_txts.append(f"${event.text_reward:,.2f} per text")

        # Building UI body
        body_for_ui = " • ".join([
            *reward_txts,
            f"{event.max_photos_per_worker or 0} photos",
            f"{event.max_videos_per_worker or 0} videos",
            f"{getattr(event, 'max_texts_per_worker', 0) or 0} texts",
            "Tap to join",
        ])

        # 3) Per-user push body with localized time
        for w in audience_qs:
            uid = w.user_id

            # Resolve timezone info for the worker
            tzname = tzmap.get(uid)
            try:
                tzinfo = ZoneInfo(tzname) if tzname else timezone.get_current_timezone()
            except Exception:
                tzinfo = timezone.get_current_timezone()

            # Localize start and end datetimes
            start_local_dt = localtime(event.startdate, tzinfo) if event.startdate else None
            end_local_dt = localtime(event.deadline, tzinfo)

            # Timezone label (abbreviation)
            tz_label = (start_local_dt or end_local_dt).tzname() if (start_local_dt or end_local_dt) else None

            # Short strings for tray text
            start_local_str = start_local_dt.strftime("%b %d, %H:%M") if start_local_dt else None
            end_local_str = end_local_dt.strftime("%b %d, %H:%M") if end_local_dt else None

            # Building the notification body for push notifications
            tray_bits = [
                f"{event.max_photos_per_worker} photos",
                f"{event.max_videos_per_worker} videos",
                f"{getattr(event, 'max_texts_per_worker', 0) or 0} texts"
            ]
            if start_local_str:
                tray_bits.append(f"Starts {start_local_str}")
            if end_local_str:
                tray_bits.append(f"Ends {end_local_str}" + (f" {tz_label}" if tz_label else ""))
            tray_body = " • ".join(tray_bits)

            # Payload for UI rendering (ISO in UTC + timezone label)
            payload = make_payload(
                type="event.created",
                event_id=event.id,
                reason="new",
                start_iso=iso_utc(event.startdate) if event.startdate else None,
                deadline_iso=iso_utc(event.deadline),
                tz_label=tz_label,  # Show timezone abbreviation
                photos_total=event.numberOfPhotos,
                max_per_worker=event.max_photos_per_worker,
                videos_total=event.numberOfVideos,
                max_videos_per_worker=event.max_videos_per_worker,
                texts_total=getattr(event, "numberOfTexts", None),
                max_texts_per_worker=getattr(event, "max_texts_per_worker", None),
                approx_reward=(
                    f"{event.photo_reward:,.2f}" if 'photo' in event.media_types
                    else f"{event.video_reward:,.2f}" if 'video' in event.media_types
                    else f"{getattr(event, 'text_reward', 0) or 0:,.2f}"
                ),
                cta="join",
                body_for_ui=body_for_ui,
            )

            # Send the push notification to each worker
            notify_user(
                user_id=uid,
                event_type="event.created",
                title=title_workers,
                body=tray_body,
                payload=payload,
                priority="high",
            )

class EventRetrieveView(generics.RetrieveAPIView):
    queryset = Event.objects.all()
    serializer_class = EventSerializer
    permission_classes = [permissions.IsAuthenticated]



class EventUpdateView(generics.UpdateAPIView):
    queryset = Event.objects.all()
    serializer_class = EventSerializer
    permission_classes = [permissions.IsAuthenticated]

    def update(self, request, *args, **kwargs):
        instance: Event = self.get_object()

        if not can_manage_event(request.user, instance):
            return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

        # Fields we want to track for "what changed?"
        tracked = [
            # core
            "title",
            "description",
            "startdate",
            "deadline",
            "keywords",

            # media model
            "media_types",
            "photo_reward",
            "video_reward",
            "text_reward",
            "numberOfPhotos",
            "numberOfVideos",
            "numberOfTexts",
            "max_photos_per_worker",
            "max_videos_per_worker",
            "max_texts_per_worker",

            # geo
            "location",
            "CoverageArea",
            "Polygon_area",
        ]

        # Snapshot BEFORE
        before = {f: getattr(instance, f) for f in tracked}

        # Perform update (uses serializer validation)
        response = super().update(request, *args, **kwargs)

        # Reload instance from DB so AFTER reflects saved values
        instance.refresh_from_db()
        after = {f: getattr(instance, f) for f in tracked}

        # Raw changed field names
        raw_changed = [f for f in tracked if before[f] != after[f]]

        # --- Friendly list for notifications ---------------------------------
        location_keys = {"CoverageArea", "Polygon_area", "location"}
        location_changed = bool(set(raw_changed) & location_keys)

        # Optional label mapping to avoid leaking internal names
        label_map = {
            "title": "title",
            "keywords": "keywords",
            "description": "description",
            "startdate": "start date",
            "deadline": "deadline",
            "media_types": "media types",
            "photo_reward": "photo reward",
            "video_reward": "video reward",
            "text_reward": "text reward",
            "numberOfPhotos": "photo quota",
            "numberOfVideos": "video quota",
            "numberOfTexts": "text quota",
            "max_photos_per_worker": "max photos/worker",
            "max_videos_per_worker": "max videos/worker",
            "max_texts_per_worker": "max texts/worker",
            "location": "location",
            "CoverageArea": "location",
            "Polygon_area": "location",
        }

        friendly_changed = []

        for f in raw_changed:
            # collapse CoverageArea/Polygon_area under one "location" label later
            if f in {"CoverageArea", "Polygon_area"}:
                continue
            friendly_changed.append(label_map.get(f, f))

        if location_changed and "location" not in friendly_changed:
            friendly_changed.append("location")

        # --- Side effects that must use RAW changes --------------------------
        if ("startdate" in raw_changed) or ("deadline" in raw_changed):
            # reschedule reminders if dates changed
            schedule_event_reminders.delay(event_id=instance.id)

        # --- Notifications (only if something meaningful changed) ------------
        if friendly_changed:
          # trim list for short message, keep full list in payload
            changed_txt = ", ".join(friendly_changed[:3]) + (
                "…" if len(friendly_changed) > 3 else ""
            )

            title = "Event updated"
            body = f"“{instance.title}” changed: {changed_txt}"

            payload = make_payload(
                type="event.update",
                event_id=instance.id,
                changed_fields=friendly_changed,
                raw_changed_fields=raw_changed,
                updated_by=request.user.id if request.user.is_authenticated else None,
                body_for_ui=body,
            )

            # notify approved workers (subscribers)
            notify_event_subscribers(
                instance,
                event_type="event.updated",
                title=title,
                body=body,
                payload=payload,
                priority="high",
            )

            # notify admins
            updater_label = (
                request.user.get_full_name()
                if request.user.is_authenticated
                else "Someone"
            )
            notify_admins(
                event_type="event.updated",
                title="Event edited",
                body=f"{updater_label} updated “{instance.title}” ({changed_txt})",
                payload=payload,
                priority="high",
            )

        return response
class UserInfoAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user  # L'utilisateur authentifié
        # Vous pouvez retourner les informations pertinentes sur l'utilisateur ici
        return Response({
            'username': user.username,
            'email': user.email,
            # Ajoutez d'autres informations selon vos besoins
        })
from django.db import transaction
from rest_framework import generics, permissions, status
from rest_framework.response import Response


class EventDeleteView(generics.DestroyAPIView):
    queryset = Event.objects.all()
    serializer_class = EventSerializer
    permission_classes = [permissions.IsAuthenticated]

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()

        if not can_manage_event(request.user, instance):
            return Response(
                {"error": "Not allowed"},
                status=status.HTTP_403_FORBIDDEN,
            )

        with transaction.atomic():
            instance.delete()

        return Response(
            status=status.HTTP_204_NO_CONTENT
        )


class WorkerStatusView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, event_id):
        worker = request.user.worker_profile
        try:
            ew = EventWorker.objects.get(event_id=event_id, worker=worker)
            return Response({'status': ew.status})
        except EventWorker.DoesNotExist:
            return Response({'status': 'NONE'})  # Not joined
class JoinEventView(generics.GenericAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, event_id):
        device_specs = request.data.get("device_specs") or {}

        try:
            with transaction.atomic():
                # Lock event row while validating the join operation.
                try:
                    event = (
                        Event.objects
                        .select_for_update()
                        .get(id=event_id)
                    )
                except Event.DoesNotExist:
                    return Response(
                        {"error": "The event does not exist."},
                        status=status.HTTP_404_NOT_FOUND,
                    )

                now = timezone.now()

                # Never allow joining an event that already ended.
                if event.deadline <= now:
                    return Response(
                        {
                            "error": "This event has already ended.",
                            "status": "expired",
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                try:
                    worker = request.user.worker_profile
                except AttributeError:
                    return Response(
                        {"error": "This user does not have a contributor profile."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                # Get/create one unique worker-event relationship.
                event_worker, created = EventWorker.objects.get_or_create(
                    worker=worker,
                    event=event,
                    defaults={
                        "device_specs": device_specs,
                    },
                )

                # Lock an existing row too, to avoid concurrent join requests
                # changing the same EventWorker simultaneously.
                if not created:
                    event_worker = (
                        EventWorker.objects
                        .select_for_update()
                        .get(pk=event_worker.pk)
                    )

                    event_worker.device_specs = device_specs
                    event_worker.save(
                        update_fields=["device_specs"]
                    )

                # ---------------------------------------------------------
                # AUTOMATIC APPROVAL
                # ---------------------------------------------------------
                #
                # Do NOT simply do:
                #
                #     event_worker.status = EventWorker.APPROVED
                #
                # because the inactivity logic needs approved_at.
                #
                # approve() stores both status and approved_at.
                #
                changed = event_worker.approve(at=now)

                event_worker_id = event_worker.id

                # Start worker-specific inactivity timing only AFTER
                # the database transaction commits successfully.
                if changed:
                    transaction.on_commit(
                        lambda ew_id=event_worker_id:
                            schedule_worker_inactivity_check.delay(
                                event_worker_id=ew_id
                            )
                    )

        except Exception:
            logger.exception(
                "Failed to automatically join event %s for user %s",
                event_id,
                request.user.id,
            )

            return Response(
                {"error": "Unable to join the event."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Already approved => do not restart approved_at or inactivity timer.
        if not changed:
            return Response(
                {
                    "message": "You have already joined this event.",
                    "status": "approved",
                    "event_worker_id": event_worker_id,
                },
                status=status.HTTP_200_OK,
            )

        return Response(
            {
                "message": "Joined successfully.",
                "status": "approved",
                "event_worker_id": event_worker_id,
            },
            status=status.HTTP_200_OK,
        )

class DeleteWorkerJoinView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def delete(self, request, event_id, user_id):
        try:
            user = User.objects.get(id=user_id)
            print(user)
            worker = user.worker_profile
            print(user.worker_profile)

        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        except AttributeError:
            return Response({'error': 'User has no worker profile'}, status=status.HTTP_400_BAD_REQUEST)

        # Validate event
        event = get_object_or_404(Event, id=event_id)

        # Delete the join request
        try:
            worker_join = EventWorker.objects.get(worker=worker, event=event)
            worker_join.delete()
            return Response({'message': 'Join request deleted successfully.'}, status=status.HTTP_204_NO_CONTENT)
        except EventWorker.DoesNotExist:
            return Response({'error': 'Join request not found.'}, status=status.HTTP_404_NOT_FOUND)
from django.db import transaction
from django.utils import timezone
from rest_framework import generics, permissions
from rest_framework.response import Response


class ApproveJoinRequestView(generics.GenericAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, event_worker_id):
        if request.user.role != "Admin":
            return Response(
                {"error": "You are not authorized to approve this join request."},
                status=403,
            )

        try:
            with transaction.atomic():
                event_worker = (
                    EventWorker.objects
                    .select_for_update()
                    .select_related("event", "worker__user")
                    .get(id=event_worker_id)
                )

                approved_at = timezone.now()

                if event_worker.event.deadline <= approved_at:
                    return Response(
                        {"error": "This event has already ended."},
                        status=400,
                    )

                changed = event_worker.approve(at=approved_at)

        except EventWorker.DoesNotExist:
            return Response(
                {"error": "Join request does not exist."},
                status=404,
            )

        # Do not send duplicate notifications when an already-approved
        # request is approved again.
        if not changed:
            return Response(
                {"message": "This join request is already approved."}
            )

        # The database transaction is committed before starting the task.
        schedule_worker_inactivity_check.delay(
            event_worker_id=event_worker.id
        )

        send_approve_join_request_email(
            event_worker.worker.user,
            event_worker.event,
        )

        body = (
            f'You have been approved for “{event_worker.event.title}”. '
            "Tap to see details."
        )

        notify_user(
            event_worker.worker.user_id,
            event_type="join_request.approved",
            title="You're in!",
            body=body,
            payload=make_payload(
                type="join_request",
                event_id=event_worker.event.id,
                status="approved",
                body_for_ui=body,
                event_worker_id=event_worker.id,
            ),
            priority="high",
        )

        return Response({
            "message": "Join request approved and worker added to event."
        })
class PendingInvitationsView(generics.ListAPIView):
    serializer_class = EventWorkerSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user

        # Vérifiez si l'utilisateur est un administrateur
        if user.role == "Admin":
            # Si l'utilisateur est un administrateur, renvoyez toutes les invitations en attente
            return EventWorker.objects.filter(
                status=EventWorker.PENDING
            ).select_related('worker', 'event')
        else:
            # Sinon, renvoyez une liste vide ou générez une exception
            return EventWorker.objects.none()
class RejectJoinRequestView(generics.GenericAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, event_worker_id):
        try:
            event_worker = EventWorker.objects.get(id=event_worker_id)

            # Vérifiez si l'utilisateur est un administrateur
            if request.user.role == "Admin":
                event_worker.status = EventWorker.REJECTED
                event_worker.save()
                send_reject_join_request_email(
                    user=event_worker.worker.user,
                    event=event_worker.event
                )
                notify_user(
                    event_worker.worker.user_id,
                    event_type="join_request.rejected",
                    title="❌ Request declined",
                    body=f"Your request to join “{event_worker.event.title}” was declined.",
                    payload=make_payload(type="join_request",
                                         event_id=event_worker.event.id,
                                         status="rejected",
                                         body_for_ui=f"Your request to join event : “{event_worker.event.title}” was declined.",
                                         event_worker_id=event_worker.id),
                    priority="high",
                )

                return Response({'message': 'Join request rejected.'})
            else:
                return Response({'error': 'You are not authorized to reject this join request.'}, status=403)
        except EventWorker.DoesNotExist:
            return Response({'error': 'Join request does not exist.'}, status=404)
class WorkerJoinedEventsByIdView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, worker_id):
        # Fetch the worker using the provided worker_id or return 404 if not found
        worker = get_object_or_404(Worker, pk=worker_id)

        # Retrieve all EventWorker instances that link to this worker with approved status
        event_workers = EventWorker.objects.filter(worker=worker, status=EventWorker.APPROVED)

        # Now retrieve all associated events from these EventWorker instances
        joined_events = [ew.event for ew in event_workers]

        # Serialize the events
        serializer = EventSerializer(joined_events, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
# class AvailableEventsForWorkerView(APIView):
#     permission_classes = [permissions.IsAuthenticated]
#
#     def get(self, request):
#         user = request.user
#         if not hasattr(user, 'worker_profile'):
#             return Response({'error': 'User is not a worker.'}, status=status.HTTP_400_BAD_REQUEST)
#
#         worker = user.worker_profile
#         now = timezone.now()
#
#         # Get IDs of all events joined by the worker
#         joined_event_ids = EventWorker.objects.filter(worker=worker).values_list('event_id', flat=True)
#
#         # Get IDs of events where the worker's status is 'PENDING'
#         pending_event_ids = EventWorker.objects.filter(
#             worker=worker,
#             status__in=[EventWorker.PENDING, EventWorker.REJECTED]
#         ).values_list('event_id', flat=True)
#
#         # Filter for available events: not joined by the worker or are pending and the deadline has not passed
#         available_events = Event.objects.filter(
#             Q(id__in=pending_event_ids) |  # Events the worker has joined but are pending
#             ~Q(id__in=joined_event_ids)  # Events the worker has not joined
#         ).filter(
#             deadline__gt=now  # Ensuring the event deadline hasn't passed
#         )
#
#         # Debug information (you can comment these out in production)
#         print(f"Joined Event IDs: {list(joined_event_ids)}")
#         print(f"SQL Query: {available_events.query}")
#         print(f"Filtered Events: {list(available_events.values())}")
#
#         # Serialize the events
#         serializer = EventSerializer(available_events, many=True)
#         return Response(serializer.data, status=status.HTTP_200_OK)

class AvailableEventsForWorkerView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user
        if not hasattr(user, 'worker_profile'):
            return Response({'error': 'User is not a worker.'}, status=status.HTTP_400_BAD_REQUEST)

        worker = user.worker_profile
        now = timezone.now()

        # Subquery: status for this worker on each Event row (NULL if no row)
        ew_status_sq = EventWorker.objects.filter(
            worker=worker,
            event_id=OuterRef('pk')
        ).values('status')[:1]

        qs = (
            Event.objects
            .filter(deadline__gt=now,organization__isnull=True)
            .annotate(_ew_status=Subquery(ew_status_sq, output_field=CharField()))
            # Keep events that are NOT joined (NULL) OR are PENDING/REJECTED (your “invitations/declined” tabs)
            .filter(Q(_ew_status__isnull=True) | Q(_ew_status__in=[EventWorker.PENDING, EventWorker.REJECTED]))
            # Expose final worker_status ("NONE" if NULL)
            .annotate(worker_status=Coalesce(F('_ew_status'), Value('NONE'), output_field=CharField()))
            # Optional: if serializer accesses requester fields, avoid extra queries
            .select_related('requester')
        )

        serializer = EventSerializer(qs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
class WorkerJoinedEventsUpcomingView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, worker_id):
        # Use get_object_or_404 to simplify and ensure correct handling of non-existent Worker
        worker = get_object_or_404(Worker, pk=worker_id)

        # Retrieve EventWorker instances that link this worker to upcoming events
        event_workers = EventWorker.objects.filter(worker=worker, event__deadline__gt=timezone.now())
        print(event_workers)

        # Extract the Event instances
        upcoming_events = [ew.event for ew in event_workers]

        # Serialize the events
        serializer = EventSerializer(upcoming_events, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
class WorkerJoinedEventsPastView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, worker_id):
        # Use get_object_or_404 to simplify and ensure correct handling of non-existent Worker
        worker = get_object_or_404(Worker, pk=worker_id)

        # Retrieve EventWorker instances that link this worker to past events
        event_workers = EventWorker.objects.filter(worker=worker, event__deadline__lt=timezone.now())

        # Extract the Event instances
        past_events = [ew.event for ew in event_workers]

        # Serialize the events
        serializer = EventSerializer(past_events, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
class EventRetrieveByIdView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    queryset = Event.objects.all()
    serializer_class = EventSerializer
class EventRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Event.objects.all()
    serializer_class = EventSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_update(self, serializer):
        serializer.save(requester=self.request.user.requester_profile)
class RequesterEventsAllView(generics.ListAPIView):
    serializer_class = EventSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        requester = self.request.user.requester_profile
        return Event.objects.filter(requester=requester,organization__isnull=True)
class RequesterEventsUpcomingView(generics.ListAPIView):
    serializer_class = EventSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        requester = self.request.user.requester_profile
        return Event.objects.filter(requester=requester, deadline__gt=timezone.now())
class RequesterEventsPastView(generics.ListAPIView):
    serializer_class = EventSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        requester = self.request.user.requester_profile
        return Event.objects.filter(requester=requester, deadline__lt=timezone.now())
class TotalUsersAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        total_users = User.objects.filter(is_active=True).count()  # Total users
        total_workers = User.objects.filter(is_worker=True,is_active=True).count()  # Count of workers
        total_requesters = User.objects.filter(is_requester=False,is_active=True).count()  # Count of requesters

        return Response({
            'totalUsers': total_users,
            'totalWorkers': total_workers,
            'totalRequesters': total_requesters
        })
class TotalEventsCreatedTodayAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        today = timezone.now().date()
        start_of_day = timezone.make_aware(datetime.combine(today, datetime.min.time()))
        end_of_day = timezone.make_aware(datetime.combine(today, datetime.max.time()))
        total_events = Event.objects.count()
        total_events_today = Event.objects.filter(
            created_at__range=[start_of_day, end_of_day]).count()  # Total des événements créés aujourd'hui
        return Response({'total_events_today': total_events_today, 'total_events': total_events})
class TotalEventsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, format=None):
        # Calculer le nombre total d'événements
        total_events = Event.objects.count()

        # Retourner la réponse en JSON
        return Response({'total_events': total_events}, status=status.HTTP_200_OK)
class EventFilter(filters.FilterSet):
    permission_classes = [permissions.IsAuthenticated]

    # Frontend uses ?type=photo|video|both
    # We map that to media_types via a custom method.
    type = filters.CharFilter(method='filter_media_type')

    requester = filters.CharFilter(
        field_name="requester__organization_name",
        lookup_expr='exact'
    )

    # Price range -> we treat it as reward range (photo or video)
    cost_min = filters.NumberFilter(method='filter_cost_min')
    cost_max = filters.NumberFilter(method='filter_cost_max')

    # Photos range (used by your frontend)
    numberOfPhotos_min = filters.NumberFilter(
        field_name="numberOfPhotos", lookup_expr='gte'
    )
    numberOfPhotos_max = filters.NumberFilter(
        field_name="numberOfPhotos", lookup_expr='lte'
    )

    # Videos range (for symmetry; only applied if sent)
    numberOfVideos_min = filters.NumberFilter(
        field_name="numberOfVideos", lookup_expr='gte'
    )
    numberOfVideos_max = filters.NumberFilter(
        field_name="numberOfVideos", lookup_expr='lte'
    )

    # Deadline range
    deadline_after = filters.DateFilter(
        field_name="deadline", lookup_expr='gte'
    )
    deadline_before = filters.DateFilter(
        field_name="deadline", lookup_expr='lte'
    )

    # Requester filter by id (you already send requester_id)
    requester_id = filters.NumberFilter(
        field_name="requester__user_id",
        lookup_expr='exact'
    )

    class Meta:
        model = Event
        fields = [
            "type",
            "requester",
            "cost_min", "cost_max",
            "numberOfPhotos_min", "numberOfPhotos_max",
            "numberOfVideos_min", "numberOfVideos_max",
            "deadline_after", "deadline_before",
            "requester_id",
        ]

    # ---------- MEDIA TYPE via media_types ----------

    def filter_media_type(self, queryset, name, value):
        """
        Use media_types ArrayField (e.g. ['photo'], ['video'], ['photo','video']).

        type=photo -> only-photo events
        type=video -> only-video events
        type=both  -> events supporting both
        """
        if not value:
            return queryset

        v = value.lower()

        # NOTE: __contains on ArrayField is "superset of".
        # So we exclude the opposite to ensure "only".
        if v == "photo":
            return (
                queryset
                .filter(media_types__contains=['photo'])
                .exclude(media_types__contains=['video'])
            )

        if v == "video":
            return (
                queryset
                .filter(media_types__contains=['video'])
                .exclude(media_types__contains=['photo'])
            )

        if v == "text":
            return (
                queryset
                .filter(media_types__contains=['text'])
                .exclude(media_types__contains=['photo'])
                .exclude(media_types__contains=['video'])
            )

        if v == "both":
            # must contain both photo and video entries
            return (
                queryset
                .filter(media_types__contains=['photo'])
                .filter(media_types__contains=['video'])
            )

        if v in {"mixed", "multi"}:
            return queryset.filter(media_types__len__gt=1)

        return queryset

    # ---------- COST FILTERS ----------

    def filter_cost_min(self, queryset, name, value):
        # min on either photo_reward or video_reward
        return queryset.filter(
            Q(photo_reward__gte=value) | Q(video_reward__gte=value) | Q(text_reward__gte=value)
        )

    def filter_cost_max(self, queryset, name, value):
        # max on either photo_reward or video_reward
        return queryset.filter(
            Q(photo_reward__lte=value) | Q(video_reward__lte=value) | Q(text_reward__lte=value)
        )
class EventListFilterView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]

    serializer_class = EventSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_class = EventFilter

    def get_queryset(self):
        qs = Event.objects.filter(organization__isnull=True)
        request = self.request

        worker_id = request.query_params.get('worker_id')
        requester_id = request.query_params.get('requester_id')
        tab = request.query_params.get('tab')

        # Only events joined by this worker
        if worker_id:
            qs = qs.filter(
                Exists(
                    EventWorker.objects.filter(
                        worker_id=worker_id,
                        event_id=OuterRef('pk'),
                        status=EventWorker.APPROVED
                    )
                )
            )

        # Limit to events of this requester
        if requester_id:
            qs = qs.filter(requester__user_id=requester_id)

        # Tab-based status filter
        now = timezone.now()
        if tab == 'in-progress':
            qs = qs.filter(deadline__gte=now)
        elif tab == 'finished':
            qs = qs.filter(deadline__lt=now)
        # 'all' => no extra

        qs = qs.distinct()
        return _with_admin_report_counts(qs, request.user)
class PendingWorkersCountAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        pending_workers_count = EventWorker.objects.filter(status='PENDING').count()
        return Response({'pending_workers_count': pending_workers_count})
class NominatimThrottle(SimpleRateThrottle):
    scope = "nominatim"
    def get_cache_key(self, request, view):
        # Single shared bucket for your server. (Swap to per-IP if you prefer.)
        return "nominatim-global"

@api_view(["GET"])
@throttle_classes([NominatimThrottle])
def reverse_geocode(request):
    lat = request.query_params.get("lat")
    lon = request.query_params.get("lon")
    if not lat or not lon:
        return Response({"detail": "lat and lon are required"}, status=400)

    accept_lang = request.query_params.get("accept-language", "en")
    cache_key = f"rev:{lat}:{lon}:{accept_lang}"
    cached = cache.get(cache_key)
    if cached:
        return Response(cached)

    headers = {
        "User-Agent": getattr(settings, "NOMINATIM_UA", "TakleefDashboard/1.0"),
        "Referer": getattr(settings, "NOMINATIM_REFERER", "https://takleef.hackhpc.com"),
        "Accept": "application/json",
        "Accept-Language": accept_lang,
    }
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={"format": "json", "lat": lat, "lon": lon},
            headers=headers,
            timeout=5,
        )
        data = r.json()
    except requests.RequestException:
        return Response({"detail": "Upstream request failed"}, status=502)

    # Cache successes for 10 minutes
    if r.status_code == 200:
        cache.set(cache_key, data, 600)

    return Response(data, status=r.status_code)


class WorkerEventDetailsView(APIView):
    """
    Per (event, worker) stats:
      - submitted_total
      - approved/declined totals
      - approved/declined split by photo/video
      - quotas per media
      - progress %
      - earned per media & total
    """
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, event_id: int):
        worker_id = request.query_params.get("worker_id")

        # if you want to default to current user, uncomment:
        # if not worker_id and request.user.is_authenticated:
        #     worker_id = request.user.id

        if not worker_id:
            return Response(
                {"detail": "worker_id query param is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
          worker_id = int(worker_id)
        except (TypeError, ValueError):
          return Response(
              {"detail": "worker_id must be an integer"},
              status=status.HTTP_400_BAD_REQUEST
          )

        event = get_object_or_404(Event, pk=event_id)

        # Base queryset: submissions of this worker in this event
        qs = Submission.objects.filter(event_id=event.id, worker_id=worker_id)

        # If there are no submissions, this still returns zeros for all counts
        agg = qs.aggregate(
            submitted_total=Count(
                "id",
                filter=Q(status__in=[Submission.APPROVED, Submission.REFUSED]),
                distinct=True,
            ),

            approved_total=Count(
                "id",
                filter=Q(status=Submission.APPROVED),
                distinct=True,
            ),
            approved_photos=Count(
                "id",
                filter=Q(status=Submission.APPROVED, photo__isnull=False),
                distinct=True,
            ),
            approved_videos=Count(
                "id",
                filter=Q(status=Submission.APPROVED, video__isnull=False),
                distinct=True,
            ),
            approved_texts=Count(
                "id",
                filter=(
                    Q(
                        status=Submission.APPROVED,
                        photo__isnull=True,
                        video__isnull=True,
                        text__isnull=False,
                    ) &
                    ~Q(text='')
                ),
                distinct=True,
            ),

            declined_total=Count(
                "id",
                filter=Q(status=Submission.REFUSED),
                distinct=True,
            ),
            declined_photos=Count(
                "id",
                filter=Q(status=Submission.REFUSED, photo__isnull=False),
                distinct=True,
            ),
            declined_videos=Count(
                "id",
                filter=Q(status=Submission.REFUSED, video__isnull=False),
                distinct=True,
            ),
            declined_texts=Count(
                "id",
                filter=(
                    Q(
                        status=Submission.REFUSED,
                        photo__isnull=True,
                        video__isnull=True,
                        text__isnull=False,
                    ) &
                    ~Q(text='')
                ),
                distinct=True,
            ),
        )

        photo_reward = float(event.photo_reward or 0)
        video_reward = float(event.video_reward or 0)
        text_reward = float(event.text_reward or 0)

        # Per-worker quotas, fallback to global event quotas if needed
        quota_photos = int(
            (event.max_photos_per_worker
             or event.numberOfPhotos
             or 0)
        )
        quota_videos = int(
            (event.max_videos_per_worker
             or event.numberOfVideos
             or 0)
        )
        quota_texts = int(
            (event.max_texts_per_worker
             or event.numberOfTexts
             or 0)
        )
        quota_total = quota_photos + quota_videos + quota_texts

        approved_total = agg["approved_total"] or 0
        approved_photos = agg["approved_photos"] or 0
        approved_videos = agg["approved_videos"] or 0
        approved_texts = agg["approved_texts"] or 0

        # Progress based on approved submissions vs available quota
        progress_pct = (approved_total / quota_total * 100) if quota_total else 0.0

        earned_photos = approved_photos * photo_reward
        earned_videos = approved_videos * video_reward
        earned_texts = approved_texts * text_reward
        total_earned = earned_photos + earned_videos + earned_texts

        data = {
            "event_id": event.id,
            "worker_id": worker_id,
            "media_types": event.media_types or [],

            "submitted_total": agg["submitted_total"] or 0,

            "approved_total": approved_total,
            "approved_photos": approved_photos,
            "approved_videos": approved_videos,
            "approved_texts": approved_texts,

            "declined_total": agg["declined_total"] or 0,
            "declined_photos": agg["declined_photos"] or 0,
            "declined_videos": agg["declined_videos"] or 0,
            "declined_texts": agg["declined_texts"] or 0,

            "quota_photos": quota_photos,
            "quota_videos": quota_videos,
            "quota_texts": quota_texts,
            "quota_total": quota_total,
            "progress_pct": round(progress_pct, 1),

            "photo_reward": photo_reward,
            "video_reward": video_reward,
            "text_reward": text_reward,
            "earned_photos": round(earned_photos, 2),
            "earned_videos": round(earned_videos, 2),
            "earned_texts": round(earned_texts, 2),
            "total_earned": round(total_earned, 2),
        }
        return Response(data, status=status.HTTP_200_OK)

class EventProgressView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    """
    GET /api/events/<event_id>/progress/

    Returns a compact summary for the Event Progress card:
      - contributors
      - total approved / declined (overall + per media)
      - acceptance rate
      - progress % (based on existing media types & requested quotas)
      - total paid (computed from Event.photo_reward / video_reward)
    """

    def get(self, request, event_id, *args, **kwargs):
        event = get_object_or_404(Event, pk=event_id)

        # Status label (optional, handy on UI)
        now = timezone.now()
        status_label = "Upcoming"
        if event.startdate and event.deadline:
            if event.startdate <= now <= event.deadline:
                status_label = "In Progress"
            elif event.deadline < now:
                status_label = "Expired"

        # --- Media detection helpers (respecting your new schema) ---
        media_types = list(getattr(event, "media_types", []) or [])

        def has_photo_media(e: Event) -> bool:
            return (
                "photo" in media_types
                or (getattr(e, "numberOfPhotos", 0) or 0) > 0
                or getattr(e, "photo_reward", None) is not None
            )

        def has_video_media(e: Event) -> bool:
            return (
                "video" in media_types
                or (getattr(e, "numberOfVideos", 0) or 0) > 0
                or getattr(e, "video_reward", None) is not None
            )

        def has_text_media(e: Event) -> bool:
            return (
                "text" in media_types
                or (getattr(e, "numberOfTexts", 0) or 0) > 0
                or getattr(e, "text_reward", None) is not None
            )

        has_photo = has_photo_media(event)
        has_video = has_video_media(event)
        has_text = has_text_media(event)

        # Requested quotas (only count media that actually exists)
        photos_req = int(getattr(event, "numberOfPhotos", 0) or 0) if has_photo else 0
        videos_req = int(getattr(event, "numberOfVideos", 0) or 0) if has_video else 0
        texts_req = int(getattr(event, "numberOfTexts", 0) or 0) if has_text else 0
        total_requested = photos_req + videos_req + texts_req

        photo_reward = float(getattr(event, "photo_reward", 0) or 0) if has_photo else 0.0
        video_reward = float(getattr(event, "video_reward", 0) or 0) if has_video else 0.0
        text_reward = float(getattr(event, "text_reward", 0) or 0) if has_text else 0.0

        # --- Contributors: approved joined workers for this event ---
        contributors = (
            EventWorker.objects
            .filter(event=event, status=EventWorker.APPROVED)
            .distinct()
            .count()
        )

        # --- All submissions for this event (from anyone) ---
        subs = Submission.objects.filter(event=event)

        APPROVED = getattr(Submission, "APPROVED", "APPROVED")
        REFUSED = getattr(Submission, "REFUSED", "REFUSED")

        agg = subs.aggregate(
            submitted_total=Count("id", distinct=True),

            approved_total=Count("id", filter=Q(status=APPROVED), distinct=True),
            declined_total=Count("id", filter=Q(status=REFUSED), distinct=True),

            approved_photos=Count(
                "id",
                filter=Q(status=APPROVED, photo__isnull=False),
                distinct=True,
            ),
            approved_videos=Count(
                "id",
                filter=Q(status=APPROVED, video__isnull=False),
                distinct=True,
            ),
            approved_texts=Count(
                "id",
                filter=(
                    Q(
                        status=APPROVED,
                        photo__isnull=True,
                        video__isnull=True,
                        text__isnull=False,
                    ) &
                    ~Q(text='')
                ),
                distinct=True,
            ),

            declined_photos=Count(
                "id",
                filter=Q(status=REFUSED, photo__isnull=False),
                distinct=True,
            ),
            declined_videos=Count(
                "id",
                filter=Q(status=REFUSED, video__isnull=False),
                distinct=True,
            ),
            declined_texts=Count(
                "id",
                filter=(
                    Q(
                        status=REFUSED,
                        photo__isnull=True,
                        video__isnull=True,
                        text__isnull=False,
                    ) &
                    ~Q(text='')
                ),
                distinct=True,
            ),
        )
        # normalize Nones
        agg = {k: int(v or 0) for k, v in agg.items()}

        submitted_total = agg["submitted_total"]
        approved_total = agg["approved_total"]
        declined_total = agg["declined_total"]

        approved_photos = agg["approved_photos"]
        approved_videos = agg["approved_videos"]
        approved_texts = agg["approved_texts"]
        declined_photos = agg["declined_photos"]
        declined_videos = agg["declined_videos"]
        declined_texts = agg["declined_texts"]

        # --- Acceptance rate (based on existing submissions) ---
        acceptance_rate = (
            round((approved_total / submitted_total) * 100, 2)
            if submitted_total > 0 else 0.0
        )

        # --- Progress % (approved vs requested, only existing media) ---
        completed_units = approved_photos + approved_videos + approved_texts
        progress_percent = (
            round((completed_units / total_requested) * 100, 2)
            if total_requested > 0 else 0.0
        )

        # --- Total paid from approved submissions (NO WorkerReward) ---
        total_paid = (
            approved_photos * photo_reward
            + approved_videos * video_reward
            + approved_texts * text_reward
        )



        payload = {
            "event_id": event.id,
            "title": event.title,
            "status": status_label,

            "media": {
                "has_photo": has_photo,
                "has_video": has_video,
                "has_text": has_text,
                "photo_reward": photo_reward,
                "video_reward": video_reward,
                "text_reward": text_reward,
                "photos_requested": photos_req,
                "videos_requested": videos_req,
                "texts_requested": texts_req,
            },

            # top-right pill
            "progress_percent": progress_percent,

            # contributors count (left column)
            "contributors": contributors,

            # submissions & decisions (for your “Photos Approved / Declined” etc.)
            "submissions": {
                "submitted_total": submitted_total,
                "approved_total": approved_total,
                "declined_total": declined_total,
                "approved_photos": approved_photos,
                "approved_videos": approved_videos,
                "approved_texts": approved_texts,
                "declined_photos": declined_photos,
                "declined_videos": declined_videos,
                "declined_texts": declined_texts,
            },

            "acceptance_rate": acceptance_rate,
            "total_paid_usd": round(total_paid, 2),
        }

        return Response(payload, status=status.HTTP_200_OK)

class EventSubmissionReportsAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]  # adjust as needed

    def get(self, request, event_id, *args, **kwargs):
        qs = SubmissionReport.objects.filter(event_id=event_id).select_related(
            'worker__user',
            'event'
        )

        serializer = SubmissionReportSerializer(qs, many=True, context={'request': request})
        return Response(serializer.data, status=status.HTTP_200_OK)

class OrganizationEventListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        return Response(
            {
                "detail": (
                    "Organization-wide event listing is disabled. "
                    "Use the filtered my-events endpoint instead."
                )
            },
            status=status.HTTP_410_GONE
        )
class OrganizationEventCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, org_id):
        organization = get_object_or_404(Organization, id=org_id)
        if organization.status != 'active':
                    return Response(
                        {"error": "Organization must be active before creating events."},
                        status=status.HTTP_403_FORBIDDEN
                    )
        membership = OrganizationMembership.objects.filter(
            user=request.user,
            organization=organization,
            status='active',
            role='requester'
        ).first()

        requester = Requester.objects.filter(user=request.user).first()

        is_org_representative = bool(
            getattr(request.user, "is_representative", False)
            and requester
            and organization.representative == requester
        )

        if not (
            (membership and membership.role == 'requester')
            or is_org_representative
        ):
            return Response(
                {"error": "Only organization requester or representative can create event"},
                status=403
            )

        serializer = EventSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        event = serializer.save(
            organization=organization,
            organization_membership=membership,
            requester=None,
        )
        schedule_event_reminders.delay(event_id=event.id)
        organization_members = (
                    OrganizationMembership.objects
                    .filter(
                        organization=organization,
                        status='active',
                        user__is_active=True
                    )
                    .select_related('user')
                    .distinct()
                )

        title_workers = f"New task published : {event.title}"
        reward_txts = []
        if 'photo' in event.media_types:
            reward_txts.append(f"${event.photo_reward:,.2f} per photo")
        if 'video' in event.media_types:
            reward_txts.append(f"${event.video_reward:,.2f} per video")
        if 'text' in event.media_types:
            reward_txts.append(f"${event.text_reward:,.2f} per text")

        body_for_ui = " • ".join([
            f"{organization.name}",
            *reward_txts,
            f"{event.max_photos_per_worker or 0} photos",
            f"{event.max_videos_per_worker or 0} videos",
            f"{getattr(event, 'max_texts_per_worker', 0) or 0} texts",
            "Tap to join",
        ])
        member_user_ids = list(organization_members.values_list("user_id", flat=True))
        tzmap = tz_map_for_users(member_user_ids)
        for member in organization_members:
            uid = member.user_id

            tzname = tzmap.get(uid)
            try:
                tzinfo = ZoneInfo(tzname) if tzname else timezone.get_current_timezone()
            except Exception:
                tzinfo = timezone.get_current_timezone()

            start_local_dt = localtime(event.startdate, tzinfo) if event.startdate else None
            end_local_dt = localtime(event.deadline, tzinfo)

            tz_label = (start_local_dt or end_local_dt).tzname() if (start_local_dt or end_local_dt) else None

            start_local_str = start_local_dt.strftime("%b %d, %H:%M") if start_local_dt else None
            end_local_str = end_local_dt.strftime("%b %d, %H:%M") if end_local_dt else None

            tray_bits = [
                f"{organization.name}",
                f"{event.max_photos_per_worker} photos",
                f"{event.max_videos_per_worker} videos",
                f"{getattr(event, 'max_texts_per_worker', 0) or 0} texts",
            ]

            if start_local_str:
                tray_bits.append(f"Starts {start_local_str}")
            if end_local_str:
                tray_bits.append(f"Ends {end_local_str}" + (f" {tz_label}" if tz_label else ""))

            tray_body = " • ".join(tray_bits)
            payload = make_payload(
                    type="event.created",
                    event_id=event.id,
                    event_name=event.title,
                    reason="new",
                    start_iso=iso_utc(event.startdate) if event.startdate else None,
                    deadline_iso=iso_utc(event.deadline),
                    tz_label=tz_label,
                    photos_total=event.numberOfPhotos,
                    max_per_worker=event.max_photos_per_worker,
                    videos_total=event.numberOfVideos,
                    max_videos_per_worker=event.max_videos_per_worker,
                    texts_total=getattr(event, "numberOfTexts", None),
                    max_texts_per_worker=getattr(event, "max_texts_per_worker", None),
                    approx_reward=(
                        f"{event.photo_reward:,.2f}" if 'photo' in event.media_types
                        else f"{event.video_reward:,.2f}" if 'video' in event.media_types
                        else f"{getattr(event, 'text_reward', 0) or 0:,.2f}"
                    ),
                    cta="join",
                    body_for_ui=body_for_ui,
                    organization_id=organization.id,
                    organization_name=organization.name,
                    mode="organization",
                    scope="organization",
            )
            notify_user(
                    user_id=uid,
                    event_type="event.created",
                    title=title_workers,
                    body=tray_body,
                    payload=payload,
                    priority="high",
            )


        return Response(EventSerializer(event).data, status=status.HTTP_201_CREATED)


class MyOrganizationEventsListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        organization = get_object_or_404(Organization, id=org_id)

        memberships = OrganizationMembership.objects.filter(
            user=request.user,
            organization=organization,
            status='active'
        )

        requester_membership = memberships.filter(role='requester').first()
        contributor_membership = memberships.filter(role='contributor').first()

        requester = Requester.objects.filter(user=request.user).first()
        is_org_representative = bool(
            getattr(request.user, "is_representative", False)
            and requester
            and organization.representative == requester
        )

        if not memberships.exists() and not is_org_representative:
            return Response(
                {"error": "Not allowed"},
                status=status.HTTP_403_FORBIDDEN
            )

        base_qs = Event.objects.filter(
            organization=organization
        ).select_related(
            'organization',
            'organization_membership__user',
            'organization__representative__user',
            'requester__user'
        ).distinct().order_by('-id')

        # representative => events created by him= membership NULL
        if is_org_representative:
            events = base_qs.filter(
                organization_membership__isnull=True
            )

            serializer = EventSerializer(events, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        # requester membre => his events
        if requester_membership:
            events = base_qs.filter(
                organization_membership=requester_membership
            )

            serializer = EventSerializer(events, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        # contributor => only joined approved
        if contributor_membership:
            if not hasattr(request.user, 'worker_profile'):
                return Response(
                    {"error": "Worker profile not found"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            worker = request.user.worker_profile

            events = base_qs.filter(
                eventworker__worker=worker,
                eventworker__status=EventWorker.APPROVED
            )

            serializer = EventSerializer(events, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response([], status=status.HTTP_200_OK)
class OrganizationAvailableEventsForWorkerView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        organization = get_object_or_404(Organization, id=org_id)

        contributor_membership = OrganizationMembership.objects.filter(
            user=request.user,
            organization=organization,
            status='active',
            role='contributor'
        ).first()

        requester = Requester.objects.filter(user=request.user).first()
        is_org_representative = bool(
            getattr(request.user, "is_representative", False)
            and requester
            and organization.representative == requester
        )

        if not contributor_membership and not is_org_representative:
            return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)


        now = timezone.now()

        events = (
                     Event.objects
                     .filter(
                         organization=organization,
                         deadline__gt=now
                     )
                     .select_related(
                         'organization',
                         'organization_membership__user',
                         'organization__representative__user',
                         'requester__user'
                     )
                     .distinct()
                     .order_by('-id')
                 )

        serializer = EventSerializer(events, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class OrganizationEventsListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
            return Response({"detail": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

        events = (
            Event.objects
            .filter(organization__isnull=False)
            .annotate(
                reported_photos_count=Count(
                    "reports",
                    filter=Q(reports__type=PHOTO),
                    distinct=True,
                ),
                reported_videos_count=Count(
                    "reports",
                    filter=Q(reports__type=VIDEO),
                    distinct=True,
                ),
            )
            .select_related(
                'organization',
                'requester',
                'organization_membership__user',
                'organization__representative__user',
            )
            .prefetch_related('joined_workers')
            .order_by('-startdate')
        )

        data = []
        for event in events:
            org_requester_name = None

            if event.organization_membership and event.organization_membership.user:
                org_requester_name = event.organization_membership.user.fullName
            elif event.organization and event.organization.representative and event.organization.representative.user:
                org_requester_name = event.organization.representative.user.fullName

            data.append({
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "startdate": event.startdate,
                "deadline": event.deadline,
                "numberOfPhotos": event.numberOfPhotos,
                "numberOfVideos": getattr(event, 'numberOfVideos', 0),
                "numberOfTexts": getattr(event, 'numberOfTexts', 0),
                "max_photos_per_worker": event.max_photos_per_worker,
                "max_videos_per_worker": getattr(event, 'max_videos_per_worker', 0),
                "max_texts_per_worker": getattr(event, 'max_texts_per_worker', 0),
                "photo_reward": event.photo_reward,
                "video_reward": getattr(event, 'video_reward', 0),
                "text_reward": getattr(event, 'text_reward', 0),
                "media_types": getattr(event, 'media_types', []),
                "location": event.location,
                "CoverageArea": event.CoverageArea,
                "organization_id": event.organization_id,
                "Polygon_area": event.Polygon_area,
                "requester_organization_name": event.organization.name if event.organization else None,
                "organization_requester_name": org_requester_name,
                "joined_workers_count": event.joined_workers.count() if hasattr(event, 'joined_workers') else 0,
                "submissions_count": Submission.objects.filter(event=event, status='accepted').count(),
                "reported_photos_count": int(event.reported_photos_count or 0),
                "reported_videos_count": int(event.reported_videos_count or 0),
            })
        return Response(data)

class OrganizationUpcomingEventsListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
                    return Response({"detail": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

        now = timezone.now()
        events = (
            Event.objects
            .filter(
                organization__isnull=False,
                deadline__gte=now,
            )
            .annotate(
                reported_photos_count=Count(
                    "reports",
                    filter=Q(reports__type=PHOTO),
                    distinct=True,
                ),
                reported_videos_count=Count(
                    "reports",
                    filter=Q(reports__type=VIDEO),
                    distinct=True,
                ),
            )
            .select_related(
                'organization',
                'requester',
                'organization_membership__user',
                'organization__representative__user',
            )
            .prefetch_related('joined_workers')
            .order_by('-startdate')
        )

        data = []
        for event in events:
            org_requester_name = None

            if event.organization_membership and event.organization_membership.user:
                org_requester_name = event.organization_membership.user.fullName
            elif event.organization and event.organization.representative and event.organization.representative.user:
                org_requester_name = event.organization.representative.user.fullName

            data.append({
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "startdate": event.startdate,
                "deadline": event.deadline,
                "numberOfPhotos": event.numberOfPhotos,
                "numberOfVideos": getattr(event, 'numberOfVideos', 0),
                "numberOfTexts": getattr(event, 'numberOfTexts', 0),
                "max_photos_per_worker": event.max_photos_per_worker,
                "max_videos_per_worker": getattr(event, 'max_videos_per_worker', 0),
                "max_texts_per_worker": getattr(event, 'max_texts_per_worker', 0),
                "photo_reward": event.photo_reward,
                "video_reward": getattr(event, 'video_reward', 0),
                "text_reward": getattr(event, 'text_reward', 0),
                "media_types": getattr(event, 'media_types', []),
                "location": event.location,
                "CoverageArea": event.CoverageArea,
                "organization_id": event.organization_id,
                "Polygon_area": event.Polygon_area,
                "requester_organization_name": event.organization.name if event.organization else None,
                "organization_requester_name": org_requester_name,
                "joined_workers_count": event.joined_workers.count() if hasattr(event, 'joined_workers') else 0,
                "submissions_count": Submission.objects.filter(event=event, status='accepted').count(),
                "reported_photos_count": int(event.reported_photos_count or 0),
                "reported_videos_count": int(event.reported_videos_count or 0),
            })
        return Response(data)
class OrganizationPastEventsListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
                    return Response({"detail": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

        now = timezone.now()
        events = (
            Event.objects
            .filter(
                organization__isnull=False,
                deadline__lt=now,
            )
            .annotate(
                reported_photos_count=Count(
                    "reports",
                    filter=Q(reports__type=PHOTO),
                    distinct=True,
                ),
                reported_videos_count=Count(
                    "reports",
                    filter=Q(reports__type=VIDEO),
                    distinct=True,
                ),
            )
            .select_related(
                'organization',
                'requester',
                'organization_membership__user',
                'organization__representative__user',
            )
            .prefetch_related('joined_workers')
            .order_by('-startdate')
        )

        data = []
        for event in events:
            org_requester_name = None

            if event.organization_membership and event.organization_membership.user:
                org_requester_name = event.organization_membership.user.fullName
            elif event.organization and event.organization.representative and event.organization.representative.user:
                org_requester_name = event.organization.representative.user.fullName

            data.append({
                "id": event.id,
                "title": event.title,
                "description": event.description,
                "startdate": event.startdate,
                "deadline": event.deadline,
                "numberOfPhotos": event.numberOfPhotos,
                "numberOfVideos": getattr(event, 'numberOfVideos', 0),
                "numberOfTexts": getattr(event, 'numberOfTexts', 0),
                "max_photos_per_worker": event.max_photos_per_worker,
                "max_videos_per_worker": getattr(event, 'max_videos_per_worker', 0),
                "max_texts_per_worker": getattr(event, 'max_texts_per_worker', 0),
                "photo_reward": event.photo_reward,
                "video_reward": getattr(event, 'video_reward', 0),
                "text_reward": getattr(event, 'text_reward', 0),
                "media_types": getattr(event, 'media_types', []),
                "location": event.location,
                "CoverageArea": event.CoverageArea,
                "organization_id": event.organization_id,
                "Polygon_area": event.Polygon_area,
                "requester_organization_name": event.organization.name if event.organization else None,
                "organization_requester_name": org_requester_name,
                "joined_workers_count": event.joined_workers.count() if hasattr(event, 'joined_workers') else 0,
                "submissions_count": Submission.objects.filter(event=event, status='accepted').count(),
                "reported_photos_count": int(event.reported_photos_count or 0),
                "reported_videos_count": int(event.reported_videos_count or 0),
            })
        return Response(data)

