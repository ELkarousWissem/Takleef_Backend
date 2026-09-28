from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from mobicrowd.models.Users import (
    Organization,
    OrganizationMembership,
    Requester,
    Worker,
)
from mobicrowd.models.submisson import Event, EventWorker, Submission
from mobicrowd.serializers.home_event_serializer import HomeEventSerializer


HOME_EVENT_LIMIT = 3


def _home_event_queryset(queryset):
    """Add only the counts required by the compact home serializer."""

    return (
        queryset
        .select_related(
            "requester__user",
            "organization",
            "organization_membership__user",
        )
        .annotate(
            joined_workers_count=Count(
                "eventworker",
                filter=Q(eventworker__status=EventWorker.APPROVED),
                distinct=True,
            ),
            approved_submissions_count=Count(
                "event_submissions",
                filter=Q(event_submissions__status=Submission.APPROVED),
                distinct=True,
            ),
        )
        .distinct()
    )


def _top_active_and_upcoming(queryset, now, limit=HOME_EVENT_LIMIT):
    """
    Return at most ``limit`` rows without loading the user's complete event list.

    Priority:
    1. Ongoing events, most recently started first.
    2. Upcoming events, nearest start date first.
    """

    valid = queryset.filter(deadline__gte=now)

    ongoing = list(
        valid
        .filter(startdate__lte=now)
        .order_by("-startdate", "deadline", "-pk")[:limit]
    )

    remaining = limit - len(ongoing)
    if remaining <= 0:
        return ongoing

    upcoming = list(
        valid
        .filter(startdate__gt=now)
        .order_by("startdate", "deadline", "pk")[:remaining]
    )
    return ongoing + upcoming


def _active_organization(organization_id):
    organization = get_object_or_404(Organization, pk=organization_id)

    if organization.status != "active":
        raise PermissionDenied("This organization is not active.")

    if organization.contract_status != "active":
        raise PermissionDenied(organization.contract_block_message)

    return organization


def _requester_profile(user):
    requester = Requester.objects.filter(user=user).first()
    if requester is None:
        raise PermissionDenied("The current user does not have requester access.")
    return requester


def _worker_profile(user):
    worker = Worker.objects.filter(user=user).first()
    if worker is None:
        raise PermissionDenied("The current user does not have contributor access.")
    return worker


class BaseHomeEventsView(APIView):
    permission_classes = [IsAuthenticated]
    mode = None
    profile = None

    def get_scoped_queryset(self, request, organization_id=None):
        raise NotImplementedError

    def get(self, request, organization_id=None):
        now = timezone.now()
        scoped = _home_event_queryset(
            self.get_scoped_queryset(request, organization_id)
        )
        events = _top_active_and_upcoming(scoped, now)
        serializer = HomeEventSerializer(
            events,
            many=True,
            context={"request": request, "now": now},
        )

        return Response(
            {
                "mode": self.mode,
                "profile": self.profile,
                "organization_id": organization_id,
                "count": len(events),
                "results": serializer.data,
            }
        )


class PublicRequesterHomeEventsView(BaseHomeEventsView):
    mode = "public"
    profile = "requester"

    def get_scoped_queryset(self, request, organization_id=None):
        requester = _requester_profile(request.user)
        return Event.objects.filter(
            organization__isnull=True,
            requester=requester,
        )


class PublicContributorHomeEventsView(BaseHomeEventsView):
    mode = "public"
    profile = "contributor"

    def get_scoped_queryset(self, request, organization_id=None):
        worker = _worker_profile(request.user)
        return Event.objects.filter(
            organization__isnull=True,
            eventworker__worker=worker,
            eventworker__status=EventWorker.APPROVED,
        )


class OrganizationRequesterHomeEventsView(BaseHomeEventsView):
    mode = "organization"
    profile = "requester"

    def get_scoped_queryset(self, request, organization_id=None):
        organization = _active_organization(organization_id)
        user = request.user

        is_representative = organization.representative.user_id == user.id
        has_requester_membership = OrganizationMembership.objects.filter(
            organization=organization,
            user=user,
            role="requester",
            status="active",
        ).exists()

        if not is_representative and not has_requester_membership:
            raise PermissionDenied(
                "The current user does not have requester access to this organization."
            )

        # Current schema stores the creator in organization_membership. Public-style
        # requester links and legacy representative events are covered explicitly.
        created_by_user = Q(organization_membership__user=user)

        requester = Requester.objects.filter(user=user).first()
        if requester is not None:
            created_by_user |= Q(requester=requester)

        if is_representative:
            created_by_user |= Q(
                organization_membership__isnull=True,
                requester__isnull=True,
            )

        return Event.objects.filter(
            Q(organization=organization) & created_by_user
        )


class OrganizationContributorHomeEventsView(BaseHomeEventsView):
    mode = "organization"
    profile = "contributor"

    def get_scoped_queryset(self, request, organization_id=None):
        organization = _active_organization(organization_id)
        user = request.user

        is_representative = organization.representative.user_id == user.id
        has_contributor_access = OrganizationMembership.objects.filter(
            organization=organization,
            user=user,
            role__in=("requester", "contributor"),
            status="active",
        ).exists()

        if not is_representative and not has_contributor_access:
            raise PermissionDenied(
                "The current user does not have contributor access to this organization."
            )

        worker = _worker_profile(user)
        return Event.objects.filter(
            organization=organization,
            eventworker__worker=worker,
            eventworker__status=EventWorker.APPROVED,
        )