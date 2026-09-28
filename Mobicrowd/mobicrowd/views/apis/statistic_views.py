from datetime import datetime

from rest_framework.permissions import IsAuthenticated

from ...models.submisson import Submission, Photo, Event, EventWorker, WorkerReward, UserUploadLog
from django.db.models import ExpressionWrapper, When, Case, Value
from ...models.Users import Worker, User, Requester, Organization, OrganizationMembership
from django.utils import timezone
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.db.models import Count, Q, Sum, F, FloatField
from django.db.models.functions import Coalesce
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from django.db.models import (
    Count, Sum, Q, F, FloatField, Case, When
)
from datetime import datetime
from django.db.models import Sum
from django.db.models.functions import Coalesce
from django.utils import timezone
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView


def _contract_block_response(organization):
    return Response(
        {
            "error": organization.contract_block_message,
            "code": f"contract_{organization.contract_status}",
            "contractStatus": organization.contract_status,
        },
        status=status.HTTP_403_FORBIDDEN
    )


class EventStatisticsView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, format=None):
        now = timezone.now()
        total_users = User.objects.filter(is_active=True).count()  # Total users
        total_workers = User.objects.filter(is_worker=True, is_active=True).count()  # Count of workers
        total_requesters = User.objects.filter(is_requester=True, is_active=True).count()  # Count of requesters

        # Filter events that are currently active
        events = Event.objects.filter(startdate__lte=now, deadline__gte=now).annotate(
            total_workers=Count('joined_workers', distinct=True),
            total_submissions=Count('event_submissions', distinct=True),

            participation_rate=ExpressionWrapper(
                Coalesce(Count('event_submissions', distinct=True), 1) /
                Coalesce(Count('joined_workers', distinct=True), 1),
                output_field=FloatField()
            ),

            approval_rate=ExpressionWrapper(
                Coalesce(Count('event_submissions', filter=Q(event_submissions__status=Submission.APPROVED)), 0) * 100.0 /
                Coalesce(Count('event_submissions'), 1),
                output_field=FloatField()
            ),

            acceptance_ratio=ExpressionWrapper(  # Optional: same as approval_rate
                Coalesce(Count('event_submissions', filter=Q(event_submissions__status=Submission.APPROVED)), 0) * 100.0 /
                Coalesce(Count('event_submissions'), 1),
                output_field=FloatField()
            )
        ).values(
            'id', 'title', 'total_workers', 'total_submissions',
            'participation_rate', 'approval_rate', 'acceptance_ratio'
        )

        return Response(events)


class AdminDashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        now = timezone.now()

        # =========================================================
        # 1) GLOBAL USER / EVENT STATS
        # =========================================================
        total_users = User.objects.filter(is_active=True).count()
        total_workers = User.objects.filter(is_worker=True, is_active=True).count()
        total_requesters = User.objects.filter(is_requester=True, is_active=True).count()

        all_events = Event.objects.all()
        total_events = all_events.count()

        today = now.date()
        start_of_day = timezone.make_aware(datetime.combine(today, datetime.min.time()))
        end_of_day = timezone.make_aware(datetime.combine(today, datetime.max.time()))
        total_events_today = Event.objects.filter(
            created_at__range=[start_of_day, end_of_day]
        ).count()

        # =========================================================
        # 2) GLOBAL MEDIA / REWARD / JOIN STATS
        # =========================================================
        g_total_photos_requested = 0
        g_total_photos_received = 0
        g_total_photos_accepted = 0
        g_total_photos_rejected = 0

        g_total_videos_requested = 0
        g_total_videos_received = 0
        g_total_videos_accepted = 0
        g_total_videos_rejected = 0

        g_total_texts_requested = 0
        g_total_texts_received = 0
        g_total_texts_accepted = 0
        g_total_texts_rejected = 0

        g_total_requested = 0
        g_total_received = 0
        g_total_accepted = 0
        g_total_rejected = 0

        g_total_contributors = 0
        g_total_paid = 0.0

        # =========================================================
        # 3) PER-EVENT STATS
        # =========================================================
        events_stats = []
        events = Event.objects.filter(
            startdate__lte=now,
            deadline__gte=now
        ).order_by('-created_at')
        for e in events:
            ev_subs = Submission.objects.filter(event=e)

            # approved joined contributors for this event
            approved_joined_workers = EventWorker.objects.filter(
                event=e,
                status=EventWorker.APPROVED
            ).count()

            # -------------------------
            # Photos
            # -------------------------
            photos_requested = e.numberOfPhotos or 0

            photos_accepted = ev_subs.filter(
                photo__isnull=False,
                status=Submission.APPROVED
            ).count()

            photos_rejected = ev_subs.filter(
                photo__isnull=False,
                status=Submission.REFUSED
            ).count()

            photos_received = ev_subs.filter(photo__isnull=False).count()

            # -------------------------
            # Videos
            # -------------------------
            videos_requested = getattr(e, "numberOfVideos", None) or 0

            videos_accepted = ev_subs.filter(
                video__isnull=False,
                status=Submission.APPROVED
            ).count()

            videos_rejected = ev_subs.filter(
                video__isnull=False,
                status=Submission.REFUSED
            ).count()

            videos_received = ev_subs.filter(video__isnull=False).count()

            # -------------------------
            # Texts
            # -------------------------
            texts_requested = getattr(e, "numberOfTexts", None) or 0
            text_subs = ev_subs.filter(
                photo__isnull=True,
                video__isnull=True,
                text__isnull=False,
            ).exclude(text='')

            texts_accepted = text_subs.filter(
                status=Submission.APPROVED
            ).count()

            texts_rejected = text_subs.filter(
                status=Submission.REFUSED
            ).count()

            texts_received = text_subs.count()

            # -------------------------
            # Totals
            # -------------------------
            total_requested = photos_requested + videos_requested + texts_requested
            total_received = photos_received + videos_received + texts_received
            total_accepted = photos_accepted + videos_accepted + texts_accepted
            total_rejected = photos_rejected + videos_rejected + texts_rejected

            # -------------------------
            # Participation rate
            # unique submitters / approved joined workers
            # -------------------------
            unique_submitters = ev_subs.values('worker').distinct().count()
            participation_rate = (
                (unique_submitters / approved_joined_workers) * 100
                if approved_joined_workers > 0 else 0.0
            )

            # -------------------------
            # Acceptance ratio
            # accepted / received
            # -------------------------
            acceptance_ratio = (
                (total_accepted / total_received) * 100
                if total_received > 0 else 0.0
            )

            # -------------------------
            # Progress
            # accepted / requested
            # -------------------------
            progress_pct = (
                (total_accepted / total_requested) * 100
                if total_requested > 0 else 0.0
            )

            # -------------------------
            # Total paid
            # -------------------------
            photo_reward = float(getattr(e, "photo_reward", None) or 0.0)
            video_reward = float(getattr(e, "video_reward", None) or 0.0)
            text_reward = float(getattr(e, "text_reward", None) or 0.0)

            total_paid = (
                (photos_accepted * photo_reward) +
                (videos_accepted * video_reward) +
                (texts_accepted * text_reward)
            )

            # -------------------------
            # Time left
            # -------------------------
            delta = e.deadline - now
            if delta.total_seconds() > 0:
                days = delta.days
                hours = delta.seconds // 3600
                minutes = (delta.seconds % 3600) // 60
                time_left_str = f"{days}d {hours}h {minutes}m"
            else:
                time_left_str = "Expired"

            # -------------------------
            # Status
            # -------------------------
            if e.startdate and e.startdate > now:
                status_label = "Upcoming"
            elif e.deadline and e.deadline < now:
                status_label = "Expired"
            else:
                status_label = "Ongoing"

            events_stats.append({
                "id": e.id,
                "title": e.title,
                "status": status_label,

                "total_photos_requested": photos_requested,
                "total_photos_received": photos_received,
                "total_photos_accepted": photos_accepted,
                "total_photos_rejected": photos_rejected,

                "total_videos_requested": videos_requested,
                "total_videos_received": videos_received,
                "total_videos_accepted": videos_accepted,
                "total_videos_rejected": videos_rejected,

                "total_texts_requested": texts_requested,
                "total_texts_received": texts_received,
                "total_texts_accepted": texts_accepted,
                "total_texts_rejected": texts_rejected,

                "total_requested": total_requested,
                "total_received": total_received,
                "total_accepted": total_accepted,
                "total_rejected": total_rejected,

                "progress_pct": round(progress_pct, 2),
                "acceptance_ratio": round(acceptance_ratio, 2),
                "participation_rate": round(participation_rate, 2),

                "contributors": approved_joined_workers,
                "time_left": time_left_str,
                "total_paid": round(total_paid, 2),

                "startdate": e.startdate,
                "deadline": e.deadline,
            })

            # -------------------------
            # accumulate globals
            # -------------------------
            g_total_photos_requested += photos_requested
            g_total_photos_received += photos_received
            g_total_photos_accepted += photos_accepted
            g_total_photos_rejected += photos_rejected

            g_total_videos_requested += videos_requested
            g_total_videos_received += videos_received
            g_total_videos_accepted += videos_accepted
            g_total_videos_rejected += videos_rejected

            g_total_texts_requested += texts_requested
            g_total_texts_received += texts_received
            g_total_texts_accepted += texts_accepted
            g_total_texts_rejected += texts_rejected

            g_total_requested += total_requested
            g_total_received += total_received
            g_total_accepted += total_accepted
            g_total_rejected += total_rejected

            g_total_contributors += approved_joined_workers
            g_total_paid += total_paid

        # =========================================================
        # 4) GLOBAL RATIOS
        # =========================================================
        global_acceptance_ratio = (
            (g_total_accepted / g_total_received) * 100
            if g_total_received > 0 else 0.0
        )

        global_progress_pct = (
            (g_total_accepted / g_total_requested) * 100
            if g_total_requested > 0 else 0.0
        )

        # =========================================================
        # 5) RESPONSE
        # =========================================================
        response_data = {
            "global_statistics": {
                "total_users": total_users,
                "total_workers": total_workers,
                "total_requesters": total_requesters,
                "total_events": total_events,
                "total_events_today": total_events_today,

                "total_photos_requested": g_total_photos_requested,
                "total_photos_received": g_total_photos_received,
                "total_photos_accepted": g_total_photos_accepted,
                "total_photos_rejected": g_total_photos_rejected,

                "total_videos_requested": g_total_videos_requested,
                "total_videos_received": g_total_videos_received,
                "total_videos_accepted": g_total_videos_accepted,
                "total_videos_rejected": g_total_videos_rejected,

                "total_texts_requested": g_total_texts_requested,
                "total_texts_received": g_total_texts_received,
                "total_texts_accepted": g_total_texts_accepted,
                "total_texts_rejected": g_total_texts_rejected,

                "total_requested": g_total_requested,
                "total_received": g_total_received,
                "total_accepted": g_total_accepted,
                "total_rejected": g_total_rejected,

                "acceptance_ratio": round(global_acceptance_ratio, 2),
                "progress_pct": round(global_progress_pct, 2),

                "total_contributors": g_total_contributors,
                "total_paid": round(g_total_paid, 2),
            },
            "events_statistics": events_stats,
        }

        return Response(response_data, status=status.HTTP_200_OK)
class WorkersStatisticsDashboardView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, *args, **kwargs):
        worker_id = kwargs.get('worker_id')
        if not worker_id:
            return Response({"error": "worker_id missing"}, status=status.HTTP_400_BAD_REQUEST)

        # 1ï¸âƒ£ Verify worker exists
        try:
            worker = Worker.objects.get(pk=worker_id)
        except Worker.DoesNotExist:
            return Response({"error": "Worker not found"}, status=status.HTTP_404_NOT_FOUND)

        # 2ï¸âƒ£ Total events joined (APPROVED via EventWorker)
        total_events_joined = (
            EventWorker.objects
            .filter(worker_id=worker_id, status=EventWorker.APPROVED)
            .count()
        )

        # 3ï¸âƒ£ All submissions by this worker
        all_subs = Submission.objects.filter(worker_id=worker_id)

        total_photos_submitted = all_subs.filter(photo__isnull=False).count()
        total_videos_submitted = all_subs.filter(video__isnull=False).count()
        total_texts_submitted = all_subs.filter(
            photo__isnull=True,
            video__isnull=True,
            text__isnull=False,
        ).exclude(text='').count()

        # 4ï¸âƒ£ Approved submissions by this worker
        approved_subs = all_subs.filter(status=Submission.APPROVED)

        total_photos_approved = approved_subs.filter(photo__isnull=False).count()
        total_videos_approved = approved_subs.filter(video__isnull=False).count()
        total_texts_approved = approved_subs.filter(
            photo__isnull=True,
            video__isnull=True,
            text__isnull=False,
        ).exclude(text='').count()

        # 5ï¸âƒ£ Overall acceptance ratio (photos + videos + texts)
        total_submitted = total_photos_submitted + total_videos_submitted + total_texts_submitted
        total_approved = total_photos_approved + total_videos_approved + total_texts_approved
        acceptance_ratio = round(
            (total_approved / total_submitted) * 100, 2
        ) if total_submitted > 0 else 0.0

        # 6ï¸âƒ£ Total data consumed
        total_data_consumed = (
            UserUploadLog.objects
            .filter(submission__worker_id=worker_id)
            .aggregate(
                total=Coalesce(Sum('size_mb'), 0.0)
            )['total'] or 0.0
        )

        # 7ï¸âƒ£ Total earnings (all events) computed from:
        #     - approved photo submissions * event.photo_reward
        #     - approved video submissions * event.video_reward
        total_earnings = (
            approved_subs.aggregate(
                total=Coalesce(
                    Sum(
                        Case(
                            # Approved photo submissions â†’ use event.photo_reward
                            When(
                                photo__isnull=False,
                                then=F('event__photo_reward')
                            ),
                            # Approved video submissions â†’ use event.video_reward
                            When(
                                video__isnull=False,
                                then=F('event__video_reward')
                            ),
                            # Approved text submissions â†’ use event.text_reward
                            When(
                                Q(photo__isnull=True) &
                                Q(video__isnull=True) &
                                Q(text__isnull=False) &
                                ~Q(text=''),
                                then=F('event__text_reward')
                            ),
                            default=0,
                            output_field=FloatField(),
                        )
                    ),
                    0.0,
                    output_field=FloatField(),
                )
            )['total'] or 0.0
        )

        # 8ï¸âƒ£ Joined events: per-event stats for THIS worker
        # Uses reverse relation 'event_submissions' from Event â†’ Submission
        joined_events = (
            Event.objects
            .filter(
                eventworker__worker_id=worker_id,
                eventworker__status=EventWorker.APPROVED
            )
            .annotate(
                # Total submissions (photo+video) by this worker for this event
                submissions_count=Count(
                    'event_submissions',
                    filter=Q(event_submissions__worker_id=worker_id),
                    distinct=True,
                ),

                # Accepted photos by this worker for this event
                accepted_photos=Count(
                    'event_submissions',
                    filter=Q(
                        event_submissions__worker_id=worker_id,
                        event_submissions__status=Submission.APPROVED,
                        event_submissions__photo__isnull=False,
                    ),
                    distinct=True,
                ),

                # Accepted videos by this worker for this event
                accepted_videos=Count(
                    'event_submissions',
                    filter=Q(
                        event_submissions__worker_id=worker_id,
                        event_submissions__status=Submission.APPROVED,
                        event_submissions__video__isnull=False,
                    ),
                    distinct=True,
                ),

                # Accepted texts by this worker for this event
                accepted_texts=Count(
                    'event_submissions',
                    filter=(
                        Q(
                            event_submissions__worker_id=worker_id,
                            event_submissions__status=Submission.APPROVED,
                            event_submissions__photo__isnull=True,
                            event_submissions__video__isnull=True,
                            event_submissions__text__isnull=False,
                        ) & ~Q(event_submissions__text='')
                    ),
                    distinct=True,
                ),

                # ðŸ’° Per-event earnings for THIS worker (no WorkerReward):
                # sum over this event's approved submissions of:
                #   photo  â†’ event.photo_reward
                #   video  â†’ event.video_reward
                worker_earnings_usd=Coalesce(
                    Sum(
                        Case(
                            When(
                                event_submissions__worker_id=worker_id,
                                event_submissions__status=Submission.APPROVED,
                                event_submissions__photo__isnull=False,
                                then=F('photo_reward'),
                            ),
                            When(
                                event_submissions__worker_id=worker_id,
                                event_submissions__status=Submission.APPROVED,
                                event_submissions__video__isnull=False,
                                then=F('video_reward'),
                            ),
                            When(
                                Q(event_submissions__worker_id=worker_id) &
                                Q(event_submissions__status=Submission.APPROVED) &
                                Q(event_submissions__photo__isnull=True) &
                                Q(event_submissions__video__isnull=True) &
                                Q(event_submissions__text__isnull=False) &
                                ~Q(event_submissions__text=''),
                                then=F('text_reward'),
                            ),
                            default=0,
                            output_field=FloatField(),
                        )
                    ),
                    0.0,
                    output_field=FloatField(),
                ),
            )
            .values(
                'id',
                'title',
                'startdate',
                'deadline',
                'location',
                'photo_reward',
                'video_reward',
                'text_reward',
                'max_photos_per_worker',
                'max_videos_per_worker',
                'max_texts_per_worker',
                'submissions_count',
                'accepted_photos',
                'accepted_videos',
                'accepted_texts',
                'worker_earnings_usd',
            )
        )

        # 9ï¸âƒ£ Final payload
        worker_stats = {
            "worker_id": worker_id,
            "total_events_joined": total_events_joined,
            "total_photos_submitted": total_photos_submitted,
            "total_videos_submitted": total_videos_submitted,
            "total_texts_submitted": total_texts_submitted,
            "total_photos_approved": total_photos_approved,
            "total_videos_approved": total_videos_approved,
            "total_texts_approved": total_texts_approved,
            "total_data_consumed_mb": round(total_data_consumed, 2),
            "total_earnings_usd": round(total_earnings, 2),
            "acceptance_ratio": acceptance_ratio,
            "joined_events": list(joined_events),
        }

        return Response(worker_stats, status=status.HTTP_200_OK)

from django.db.models import Sum
from django.db.models.functions import Coalesce
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

class RequesterEventStatisticsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        # Scope every row and aggregate to the authenticated requester profile.
        # Requester.user is the profile primary key, but resolving the profile
        # explicitly avoids ever falling back to a database-wide queryset.
        requester = Requester.objects.filter(user=request.user).first()
        if requester is None:
            return Response(
                {"error": "Requester profile not found for the authenticated user"},
                status=status.HTTP_403_FORBIDDEN,
            )

        events = Event.objects.filter(requester=requester)

        event_data = []

        # -------- global aggregates --------
        total_photos_requested = 0
        total_photos_received = 0
        total_photos_accepted = 0
        total_photos_rejected = 0

        total_videos_requested = 0
        total_videos_received = 0
        total_videos_accepted = 0
        total_videos_rejected = 0

        total_texts_requested = 0
        total_texts_received = 0
        total_texts_accepted = 0
        total_texts_rejected = 0

        total_requested = 0
        total_received = 0
        total_accepted = 0
        total_rejected = 0

        total_workers_joined = 0
        total_rewards = 0.0
        total_data_saving = 0.0

        now = timezone.now()

        for event in events:
            # -------- status --------
            if event.startdate and event.deadline:
                if event.startdate <= now <= event.deadline:
                    status = "In Progress"
                elif event.deadline < now:
                    status = "Expired"
                else:
                    status = "Upcoming"
            else:
                status = "Upcoming"

            # -------- requested --------
            photos_req = event.numberOfPhotos or 0
            videos_req = getattr(event, "numberOfVideos", None) or 0
            texts_req = getattr(event, "numberOfTexts", None) or 0

            # -------- submissions base queryset --------
            ev_subs = Submission.objects.filter(event=event)

            # -------- photos --------
            photos_acc = ev_subs.filter(
                photo__isnull=False,
                status=Submission.APPROVED
            ).count()

            photos_rej = ev_subs.filter(
                photo__isnull=False,
                status=Submission.REFUSED
            ).count()

            photos_recv = ev_subs.filter(photo__isnull=False).count()

            # -------- videos --------
            videos_acc = ev_subs.filter(
                video__isnull=False,
                status=Submission.APPROVED
            ).count()

            videos_rej = ev_subs.filter(
                video__isnull=False,
                status=Submission.REFUSED
            ).count()

            videos_recv = ev_subs.filter(video__isnull=False).count()

            # -------- texts --------
            # The frontend renders text metrics as NA when texts_req is zero.
            # Exclude those events from the requester summary as well, so the
            # card is exactly the sum of applicable rows in this response.
            if texts_req > 0:
                text_subs = ev_subs.filter(
                    photo__isnull=True,
                    video__isnull=True,
                    text__isnull=False,
                ).exclude(text='')
                texts_acc = text_subs.filter(status=Submission.APPROVED).count()
                texts_rej = text_subs.filter(status=Submission.REFUSED).count()
                texts_recv = text_subs.count()
            else:
                texts_acc = 0
                texts_rej = 0
                texts_recv = 0

            # -------- totals --------
            event_total_requested = photos_req + videos_req + texts_req
            event_total_received = photos_recv + videos_recv + texts_recv
            event_total_accepted = photos_acc + videos_acc + texts_acc
            event_total_rejected = photos_rej + videos_rej + texts_rej

            # -------- joined workers --------
            workers_joined = EventWorker.objects.filter(
                event=event,
                status=EventWorker.APPROVED
            ).count()

            # -------- rewards --------
            photo_reward = float(getattr(event, "photo_reward", None) or 0.0)
            video_reward = float(getattr(event, "video_reward", None) or 0.0)
            text_reward = float(getattr(event, "text_reward", None) or 0.0)

            event_rewards = (
                (photos_acc * photo_reward) +
                (videos_acc * video_reward) +
                (texts_acc * text_reward)
            )

            # -------- data saving --------
            data_saving = (
                UserUploadLog.objects
                .filter(submission__event=event)
                .aggregate(total=Coalesce(Sum('size_mb'), 0.0))['total'] or 0.0
            )

            # -------- acceptance ratio --------
            event_acceptance_ratio = (
                round((event_total_accepted / event_total_received) * 100, 2)
                if event_total_received > 0 else 0.0
            )

            # -------- progress --------
            event_progress_pct = (
                round((event_total_accepted / event_total_requested) * 100, 2)
                if event_total_requested > 0 else 0.0
            )

            # -------- row --------
            event_data.append({
                "event_id": event.id,
                "event_title": event.title,
                "status": status,

                "total_photos_requested": photos_req,
                "total_photos_received": photos_recv,
                "total_photos_accepted": photos_acc,
                "total_photos_rejected": photos_rej,

                "total_videos_requested": videos_req,
                "total_videos_received": videos_recv,
                "total_videos_accepted": videos_acc,
                "total_videos_rejected": videos_rej,

                "total_texts_requested": texts_req,
                "total_texts_received": texts_recv,
                "total_texts_accepted": texts_acc,
                "total_texts_rejected": texts_rej,

                "total_requested": event_total_requested,
                "total_received": event_total_received,
                "total_accepted": event_total_accepted,
                "total_rejected": event_total_rejected,

                "acceptance_ratio": event_acceptance_ratio,
                "progress_pct": event_progress_pct,

                "total_workers_joined": workers_joined,
                "total_rewards": round(event_rewards, 2),
                "total_data_saving": round(data_saving, 2),
            })

            # -------- accumulate globals --------
            total_photos_requested += photos_req
            total_photos_received += photos_recv
            total_photos_accepted += photos_acc
            total_photos_rejected += photos_rej

            total_videos_requested += videos_req
            total_videos_received += videos_recv
            total_videos_accepted += videos_acc
            total_videos_rejected += videos_rej

            total_texts_requested += texts_req
            total_texts_received += texts_recv
            total_texts_accepted += texts_acc
            total_texts_rejected += texts_rej

            total_requested += event_total_requested
            total_received += event_total_received
            total_accepted += event_total_accepted
            total_rejected += event_total_rejected

            total_workers_joined += workers_joined
            total_rewards += event_rewards
            total_data_saving += data_saving

        total_events = events.count()

        global_acceptance_ratio = (
            round((total_accepted / total_received) * 100, 2)
            if total_received > 0 else 0.0
        )

        global_progress_pct = (
            round((total_accepted / total_requested) * 100, 2)
            if total_requested > 0 else 0.0
        )

        return Response({
            "event_statistics": event_data,
            "global_statistics": {
                "total_events": total_events,
                "total_workers_joined": total_workers_joined,

                "total_photos_requested": total_photos_requested,
                "total_photos_received": total_photos_received,
                "total_photos_accepted": total_photos_accepted,
                "total_photos_rejected": total_photos_rejected,

                "total_videos_requested": total_videos_requested,
                "total_videos_received": total_videos_received,
                "total_videos_accepted": total_videos_accepted,
                "total_videos_rejected": total_videos_rejected,

                "total_texts_requested": total_texts_requested,
                "total_texts_received": total_texts_received,
                "total_texts_accepted": total_texts_accepted,
                "total_texts_rejected": total_texts_rejected,

                "total_requested": total_requested,
                "total_received": total_received,
                "total_accepted": total_accepted,
                "total_rejected": total_rejected,

                "acceptance_ratio": global_acceptance_ratio,
                "progress_pct": global_progress_pct,

                "total_rewards": round(total_rewards, 2),
                "total_data_saving": round(total_data_saving, 2),
            }
        })
class AllWorkersStatisticsView(APIView):
    permission_classes = [IsAuthenticated]
    """
    Returns all workers who joined at least one approved event,
    with their total joined events count and acceptance ratio.
    """

    def get(self, request, *args, **kwargs):
        # get all workers who have joined at least one approved event
        approved_event_workers = (
            EventWorker.objects.filter(status=EventWorker.APPROVED)
            .select_related("worker__user")
            .values("worker_id", "worker__user__fullName")
            .distinct()
        )

        results = []

        for ew in approved_event_workers:
            worker_id = ew["worker_id"]
            full_name = ew["worker__user__fullName"]

            # total joined events (approved only)
            joined_events_count = EventWorker.objects.filter(
                worker_id=worker_id, status=EventWorker.APPROVED
            ).count()

            # submissions for acceptance ratio
            all_subs = Submission.objects.filter(worker_id=worker_id)
            total_submitted = all_subs.count()
            total_approved = all_subs.filter(status=Submission.APPROVED).count()

            acceptance_ratio = (
                round((total_approved / total_submitted) * 100, 2)
                if total_submitted > 0
                else 0.0
            )

            results.append({
                "worker_id": worker_id,
                "full_name": full_name,
                "joined_events_count": joined_events_count,
                "acceptance_ratio": acceptance_ratio,
            })

        return Response({"workers": results}, status=status.HTTP_200_OK)
class CurrentEventsWithContributorsView(APIView):
    permission_classes = [IsAuthenticated]
    """
    Returns all current (ongoing/upcoming) events with their contributors (workers).
    Each event includes event_id, event_title, and a list of joined workers.
    """

    def get(self, request, *args, **kwargs):
        now = timezone.now()

        # Only ongoing or upcoming events
        current_events = Event.objects.filter(
            deadline__gte=now  # still active or future
        ).select_related("requester").order_by("deadline")

        response_data = []

        for event in current_events:
            # Get approved workers for this event
            event_workers = (
                EventWorker.objects
                .filter(event=event, status=EventWorker.APPROVED)
                .select_related("worker__user")
                .values(
                    "worker__user__id",
                    "worker__user__fullName",
                    "joined_at"
                )
            )

            workers_data = [
                {
                    "worker_id": w["worker__user__id"],
                    "worker_name": w["worker__user__fullName"],
                    "joined_at": w["joined_at"]
                }
                for w in event_workers
            ]

            response_data.append({
                "event_id": event.id,
                "event_title": event.title,
                "workers": workers_data
            })

        return Response(response_data, status=status.HTTP_200_OK)


class OrganizationRequesterDashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        organization = get_object_or_404(Organization, id=org_id)

        membership = (
            OrganizationMembership.objects
            .filter(
                user=request.user,
                organization=organization,
                status='active',
                role='requester'
            )
            .select_related('organization')
            .first()
        )

        if not membership:
            return Response(
                {"error": "Active organization requester membership not found for this organization"},
                status=status.HTTP_403_FORBIDDEN
            )

        if organization.contract_status != "active":
            return _contract_block_response(organization)

        events = (
            Event.objects
            .filter(
                organization=organization,
                organization_membership=membership
            )
            .order_by('-id')
        )

        rows = []

        total_events = 0
        total_workers_joined = 0

        total_photos_requested = 0
        total_photos_received = 0
        total_photos_accepted = 0
        total_photos_rejected = 0

        total_videos_requested = 0
        total_videos_received = 0
        total_videos_accepted = 0
        total_videos_rejected = 0

        total_texts_requested = 0
        total_texts_received = 0
        total_texts_accepted = 0
        total_texts_rejected = 0

        total_requested = 0
        total_received = 0
        total_accepted = 0
        total_rejected = 0

        total_rewards = 0.0
        total_data_saving = 0.0

        now = timezone.now()

        for event in events:
            approved_workers = EventWorker.objects.filter(
                event=event,
                status=EventWorker.APPROVED
            ).count()

            submissions = Submission.objects.filter(event=event)

            photos_requested = int(event.numberOfPhotos or 0)
            videos_requested = int(event.numberOfVideos or 0)
            texts_requested = int(event.numberOfTexts or 0)

            photos_received = submissions.filter(photo__isnull=False).count()
            photos_accepted = submissions.filter(
                photo__isnull=False,
                status=Submission.APPROVED
            ).count()
            photos_rejected = submissions.filter(
                photo__isnull=False,
                status__in=[Submission.REFUSED]
            ).count()

            videos_received = submissions.filter(video__isnull=False).count()
            videos_accepted = submissions.filter(
                video__isnull=False,
                status=Submission.APPROVED
            ).count()
            videos_rejected = submissions.filter(
                video__isnull=False,
                status__in=[Submission.REFUSED]
            ).count()

            if texts_requested > 0:
                text_submissions = submissions.filter(
                    photo__isnull=True,
                    video__isnull=True,
                    text__isnull=False,
                ).exclude(text='')
                texts_received = text_submissions.count()
                texts_accepted = text_submissions.filter(status=Submission.APPROVED).count()
                texts_rejected = text_submissions.filter(status=Submission.REFUSED).count()
            else:
                texts_received = 0
                texts_accepted = 0
                texts_rejected = 0

            requested = photos_requested + videos_requested + texts_requested
            received = photos_received + videos_received + texts_received
            accepted = photos_accepted + videos_accepted + texts_accepted
            rejected = photos_rejected + videos_rejected + texts_rejected

            acceptance_ratio = round((accepted / received) * 100, 2) if received > 0 else 0.0
            progress_pct = round((accepted / requested) * 100, 2) if requested > 0 else 0.0

            rewards = (
                photos_accepted * float(event.photo_reward or 0) +
                videos_accepted * float(event.video_reward or 0) +
                texts_accepted * float(event.text_reward or 0)
            )

            if event.deadline and event.deadline < now:
                event_status = "Expired"
            elif event.startdate and event.startdate > now:
                event_status = "Upcoming"
            else:
                event_status = "In Progress"

            row = {
                "event_id": event.id,
                "event_title": event.title,
                "status": event_status,

                "total_photos_requested": photos_requested,
                "total_photos_received": photos_received,
                "total_photos_accepted": photos_accepted,
                "total_photos_rejected": photos_rejected,

                "total_videos_requested": videos_requested,
                "total_videos_received": videos_received,
                "total_videos_accepted": videos_accepted,
                "total_videos_rejected": videos_rejected,

                "total_texts_requested": texts_requested,
                "total_texts_received": texts_received,
                "total_texts_accepted": texts_accepted,
                "total_texts_rejected": texts_rejected,

                "total_requested": requested,
                "total_received": received,
                "total_accepted": accepted,
                "total_rejected": rejected,

                "acceptance_ratio": acceptance_ratio,
                "progress_pct": progress_pct,

                "total_workers_joined": approved_workers,
                "total_rewards": round(rewards, 2),
                "total_data_saving": 0.0,
            }
            rows.append(row)

            total_events += 1
            total_workers_joined += approved_workers

            total_photos_requested += photos_requested
            total_photos_received += photos_received
            total_photos_accepted += photos_accepted
            total_photos_rejected += photos_rejected

            total_videos_requested += videos_requested
            total_videos_received += videos_received
            total_videos_accepted += videos_accepted
            total_videos_rejected += videos_rejected

            total_texts_requested += texts_requested
            total_texts_received += texts_received
            total_texts_accepted += texts_accepted
            total_texts_rejected += texts_rejected

            total_requested += requested
            total_received += received
            total_accepted += accepted
            total_rejected += rejected

            total_rewards += rewards

        global_statistics = {
            "total_events": total_events,
            "total_workers_joined": total_workers_joined,

            "total_photos_requested": total_photos_requested,
            "total_photos_received": total_photos_received,
            "total_photos_accepted": total_photos_accepted,
            "total_photos_rejected": total_photos_rejected,

            "total_videos_requested": total_videos_requested,
            "total_videos_received": total_videos_received,
            "total_videos_accepted": total_videos_accepted,
            "total_videos_rejected": total_videos_rejected,

            "total_texts_requested": total_texts_requested,
            "total_texts_received": total_texts_received,
            "total_texts_accepted": total_texts_accepted,
            "total_texts_rejected": total_texts_rejected,

            "total_requested": total_requested,
            "total_received": total_received,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,

            "acceptance_ratio": round((total_accepted / total_received) * 100, 2) if total_received > 0 else 0.0,
            "progress_pct": round((total_accepted / total_requested) * 100, 2) if total_requested > 0 else 0.0,

            "total_rewards": round(total_rewards, 2),
            "total_data_saving": round(total_data_saving, 2),
        }

        return Response({
            "global_statistics": global_statistics,
            "event_statistics": rows
        }, status=status.HTTP_200_OK)
class OrganizationContributorDashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        organization = get_object_or_404(Organization, id=org_id)

        membership = (
            OrganizationMembership.objects
            .filter(
                user=request.user,
                organization=organization,
                status='active',
                role='contributor'
            )
            .select_related('organization', 'user')
            .first()
        )

        if not membership:
            return Response(
                {"error": "Active organization contributor membership not found for this organization"},
                status=status.HTTP_403_FORBIDDEN
            )

        if organization.contract_status != "active":
            return _contract_block_response(organization)

        if not hasattr(request.user, 'worker_profile'):
            return Response(
                {"error": "User has no worker profile"},
                status=status.HTTP_400_BAD_REQUEST
            )

        worker = request.user.worker_profile

        joined_event_workers = (
            EventWorker.objects
            .filter(
                worker=worker,
                status=EventWorker.APPROVED,
                event__organization=organization
            )
            .select_related('event')
            .order_by('event__deadline')
        )

        joined_events_payload = []

        total_events_joined = 0
        total_photos_submitted = 0
        total_videos_submitted = 0
        total_texts_submitted = 0
        total_photos_approved = 0
        total_videos_approved = 0
        total_texts_approved = 0
        total_earnings_usd = 0.0
        total_data_consumed_mb = 0.0

        for ew in joined_event_workers:
            event = ew.event

            submissions = Submission.objects.filter(
                event=event,
                worker=worker
            )

            submissions_count = submissions.filter(
                status__in=[Submission.APPROVED, Submission.REFUSED]
            ).count()

            photos_submitted = submissions.filter(photo__isnull=False).count()
            videos_submitted = submissions.filter(video__isnull=False).count()
            texts_submitted = submissions.filter(
                photo__isnull=True,
                video__isnull=True,
                text__isnull=False,
            ).exclude(text='').count()

            accepted_photos = submissions.filter(
                status=Submission.APPROVED,
                photo__isnull=False
            ).count()

            accepted_videos = submissions.filter(
                status=Submission.APPROVED,
                video__isnull=False
            ).count()

            accepted_texts = submissions.filter(
                status=Submission.APPROVED,
                photo__isnull=True,
                video__isnull=True,
                text__isnull=False
            ).exclude(text='').count()

            worker_earnings_usd = (
                accepted_photos * float(event.photo_reward or 0) +
                accepted_videos * float(event.video_reward or 0) +
                accepted_texts * float(event.text_reward or 0)
            )

            total_events_joined += 1
            total_photos_submitted += photos_submitted
            total_videos_submitted += videos_submitted
            total_texts_submitted += texts_submitted
            total_photos_approved += accepted_photos
            total_videos_approved += accepted_videos
            total_texts_approved += accepted_texts
            total_earnings_usd += worker_earnings_usd

            joined_events_payload.append({
                "id": event.id,
                "title": event.title,
                "startdate": event.startdate,
                "deadline": event.deadline,
                "max_photos_per_worker": event.max_photos_per_worker or 0,
                "max_videos_per_worker": event.max_videos_per_worker or 0,
                "max_texts_per_worker": event.max_texts_per_worker or 0,
                "photo_reward": float(event.photo_reward or 0),
                "video_reward": float(event.video_reward or 0),
                "text_reward": float(event.text_reward or 0),
                "submissions_count": submissions_count,
                "accepted_photos": accepted_photos,
                "accepted_videos": accepted_videos,
                "accepted_texts": accepted_texts,
                "worker_earnings_usd": round(worker_earnings_usd, 2),
                "location": getattr(event, "location", None),
            })

        total_submitted = total_photos_submitted + total_videos_submitted + total_texts_submitted
        total_approved = total_photos_approved + total_videos_approved + total_texts_approved

        acceptance_ratio = (
            round((total_approved / total_submitted) * 100, 2)
            if total_submitted > 0 else 0.0
        )

        return Response({
            "total_events_joined": total_events_joined,
            "total_photos_submitted": total_photos_submitted,
            "total_videos_submitted": total_videos_submitted,
            "total_texts_submitted": total_texts_submitted,
            "total_photos_approved": total_photos_approved,
            "total_videos_approved": total_videos_approved,
            "total_texts_approved": total_texts_approved,
            "total_earnings_usd": round(total_earnings_usd, 2),
            "total_data_consumed_mb": round(total_data_consumed_mb, 2),
            "acceptance_ratio": acceptance_ratio,
            "joined_events": joined_events_payload
        }, status=status.HTTP_200_OK)
class OrganizationAdminDashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        if not getattr(request.user, "is_representative", False):
            return Response(
                {"error": "Only representative can access organization admin dashboard"},
                status=status.HTTP_403_FORBIDDEN
            )

        requester = Requester.objects.filter(user=request.user).first()
        if not requester:
            return Response(
                {"error": "Requester profile not found for representative user"},
                status=status.HTTP_404_NOT_FOUND
            )

        organization = get_object_or_404(Organization, id=org_id)

        if organization.representative != requester:
            return Response(
                {"error": "You are not the representative of this organization"},
                status=status.HTTP_403_FORBIDDEN
            )

        if organization.contract_status != "active":
            return _contract_block_response(organization)

        now = timezone.now()

        memberships_qs = OrganizationMembership.objects.filter(
            organization=organization,
            status='active'
        ).select_related('user')

        total_users = memberships_qs.values('user_id').distinct().count()
        total_requesters = memberships_qs.filter(role='requester').values('user_id').distinct().count()
        total_workers = memberships_qs.filter(role='contributor').values('user_id').distinct().count()

        events = Event.objects.filter(organization=organization).order_by('-id')
        total_events = events.count()

        events_statistics = []
        workers_statistics = []
        current_events_with_contributors = []

        total_photos_requested = 0
        total_photos_received = 0
        total_photos_accepted = 0
        total_photos_rejected = 0

        total_videos_requested = 0
        total_videos_received = 0
        total_videos_accepted = 0
        total_videos_rejected = 0

        total_texts_requested = 0
        total_texts_received = 0
        total_texts_accepted = 0
        total_texts_rejected = 0

        total_requested = 0
        total_received = 0
        total_accepted = 0
        total_rejected = 0
        total_paid = 0.0
        total_contributors = 0

        for event in events:
            contributors = EventWorker.objects.filter(
                event=event,
                status=EventWorker.APPROVED
            ).values('worker_id').distinct().count()

            submissions = Submission.objects.filter(event=event)

            photos_requested = int(event.numberOfPhotos or 0)
            videos_requested = int(event.numberOfVideos or 0)
            texts_requested = int(event.numberOfTexts or 0)

            photos_received = submissions.filter(photo__isnull=False).count()
            photos_accepted = submissions.filter(photo__isnull=False, status=Submission.APPROVED).count()
            photos_rejected = submissions.filter(photo__isnull=False, status__in=[Submission.REFUSED]).count()

            videos_received = submissions.filter(video__isnull=False).count()
            videos_accepted = submissions.filter(video__isnull=False, status=Submission.APPROVED).count()
            videos_rejected = submissions.filter(video__isnull=False, status__in=[Submission.REFUSED]).count()

            text_submissions = submissions.filter(
                photo__isnull=True,
                video__isnull=True,
                text__isnull=False,
            ).exclude(text='')
            texts_received = text_submissions.count()
            texts_accepted = text_submissions.filter(status=Submission.APPROVED).count()
            texts_rejected = text_submissions.filter(status=Submission.REFUSED).count()

            requested = photos_requested + videos_requested + texts_requested
            received = photos_received + videos_received + texts_received
            accepted = photos_accepted + videos_accepted + texts_accepted
            rejected = photos_rejected + videos_rejected + texts_rejected

            progress_pct = round((accepted / requested) * 100, 2) if requested > 0 else 0.0
            acceptance_ratio = round((accepted / received) * 100, 2) if received > 0 else 0.0
            participation_rate = round((contributors / total_workers) * 100, 2) if total_workers > 0 else 0.0

            paid = (
                photos_accepted * float(event.photo_reward or 0) +
                videos_accepted * float(event.video_reward or 0) +
                texts_accepted * float(event.text_reward or 0)
            )

            if event.deadline and event.deadline < now:
                status_label = "Expired"
                time_left = "â€”"
            elif event.startdate and event.startdate > now:
                status_label = "Upcoming"
                time_left = "Not started"
            else:
                status_label = "Ongoing"
                if event.deadline:
                    diff = event.deadline - now
                    days = diff.days
                    hours = diff.seconds // 3600
                    if days > 0:
                        time_left = f"{days}d left"
                    elif hours > 0:
                        time_left = f"{hours}h left"
                    else:
                        time_left = "Less than 1h"
                else:
                    time_left = "â€”"

            events_statistics.append({
                "id": event.id,
                "title": event.title,
                "status": status_label,
                "total_photos_requested": photos_requested,
                "total_photos_received": photos_received,
                "total_photos_accepted": photos_accepted,
                "total_photos_rejected": photos_rejected,
                "total_videos_requested": videos_requested,
                "total_videos_received": videos_received,
                "total_videos_accepted": videos_accepted,
                "total_videos_rejected": videos_rejected,
                "total_texts_requested": texts_requested,
                "total_texts_received": texts_received,
                "total_texts_accepted": texts_accepted,
                "total_texts_rejected": texts_rejected,
                "total_requested": requested,
                "total_received": received,
                "total_accepted": accepted,
                "total_rejected": rejected,
                "progress_pct": progress_pct,
                "acceptance_ratio": acceptance_ratio,
                "participation_rate": participation_rate,
                "contributors": contributors,
                "time_left": time_left,
                "total_paid": round(paid, 2),
            })

            total_photos_requested += photos_requested
            total_photos_received += photos_received
            total_photos_accepted += photos_accepted
            total_photos_rejected += photos_rejected

            total_videos_requested += videos_requested
            total_videos_received += videos_received
            total_videos_accepted += videos_accepted
            total_videos_rejected += videos_rejected

            total_texts_requested += texts_requested
            total_texts_received += texts_received
            total_texts_accepted += texts_accepted
            total_texts_rejected += texts_rejected

            total_requested += requested
            total_received += received
            total_accepted += accepted
            total_rejected += rejected
            total_paid += paid
            total_contributors += contributors

        contributor_memberships = memberships_qs.filter(role='contributor')

        for m in contributor_memberships:
            user = m.user
            if not hasattr(user, 'worker_profile'):
                continue

            worker = user.worker_profile

            joined_events_count = EventWorker.objects.filter(
                worker=worker,
                status=EventWorker.APPROVED,
                event__organization=organization
            ).count()

            worker_submissions = Submission.objects.filter(
                worker=worker,
                event__organization=organization
            )

            submitted_total = worker_submissions.count()
            approved_total = worker_submissions.filter(status=Submission.APPROVED).count()
            acceptance_ratio_worker = round((approved_total / submitted_total) * 100, 2) if submitted_total > 0 else 0.0

            workers_statistics.append({
                "worker_id": user.id,
                "full_name": user.get_full_name() or user.email,
                "joined_events_count": joined_events_count,
                "acceptance_ratio": acceptance_ratio_worker
            })

        current_events = events.filter(deadline__gte=now)

        for event in current_events:
            joined = EventWorker.objects.filter(
                event=event,
                status=EventWorker.APPROVED
            ).select_related('worker__user')

            workers_payload = []
            for ew in joined:
                workers_payload.append({
                    "worker_id": ew.worker.user.id,
                    "worker_name": ew.worker.user.get_full_name() or ew.worker.user.email,
                    "joined_at": getattr(ew, "joined_at", None)
                })

            current_events_with_contributors.append({
                "event_id": event.id,
                "event_title": event.title,
                "workers": workers_payload
            })

        global_statistics = {
            "total_users": total_users,
            "total_workers": total_workers,
            "total_requesters": total_requesters,
            "total_events": total_events,

            "total_photos_requested": total_photos_requested,
            "total_photos_received": total_photos_received,
            "total_photos_accepted": total_photos_accepted,
            "total_photos_rejected": total_photos_rejected,

            "total_videos_requested": total_videos_requested,
            "total_videos_received": total_videos_received,
            "total_videos_accepted": total_videos_accepted,
            "total_videos_rejected": total_videos_rejected,

            "total_texts_requested": total_texts_requested,
            "total_texts_received": total_texts_received,
            "total_texts_accepted": total_texts_accepted,
            "total_texts_rejected": total_texts_rejected,

            "total_requested": total_requested,
            "total_received": total_received,
            "total_accepted": total_accepted,
            "total_rejected": total_rejected,

            "acceptance_ratio": round((total_accepted / total_received) * 100, 2) if total_received > 0 else 0.0,
            "progress_pct": round((total_accepted / total_requested) * 100, 2) if total_requested > 0 else 0.0,
            "total_paid": round(total_paid, 2),
            "total_contributors": total_contributors,
        }

        return Response({
            "global_statistics": global_statistics,
            "events_statistics": events_statistics,
            "workers_statistics": workers_statistics,
            "current_events_with_contributors": current_events_with_contributors
        }, status=status.HTTP_200_OK)
