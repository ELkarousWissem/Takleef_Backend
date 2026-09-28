from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from mobicrowd.serializers.organizationSerializer import (
    OrganizationSerializer,
    MyOrganizationSerializer,
    OrganizationDetailSerializer,
    OrganizationMemberSerializer,
    OrganizationInviteSerializer,
    OrganizationMembershipCreateSerializer,
)
from mobicrowd.models.Users import (
    DEFAULT_INVITATION_DURATION_HOURS,
    User,
    Organization,
    Requester,
    OrganizationMembership,
    OrganizationInvitation,
    OrganizationLicenceKey,
)
from django.utils import timezone
from django.conf import settings
from django.shortcuts import get_object_or_404
from mobicrowd.authentication.email_sending import (
    send_organization_invitation_cancelled_email,
    send_organization_invitation_email,
)
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import ProtectedError
from django.utils.dateparse import parse_date
from dateutil.relativedelta import relativedelta

def _is_admin_user(user):
    return bool(
        getattr(user, "role", "") == "Admin"
        or getattr(user, "is_superuser", False)
    )

def calculate_expiration_date(duration, starting_date=None):
    start_date = starting_date

    if isinstance(start_date, str):
        start_date = parse_date(start_date)

    if not start_date:
        start_date = timezone.now().date()

    if duration == "yearly":
        return start_date + relativedelta(years=1)

    return start_date + relativedelta(months=1)


def _contract_block_response(org):
    return Response(
        {
            "detail": org.contract_block_message,
            "code": f"contract_{org.contract_status}",
            "contractStatus": org.contract_status,
        },
        status=status.HTTP_403_FORBIDDEN
    )


class OrganizationCreateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        if not _is_admin_user(request.user):
            return Response(
                {"detail": "Only admin can create an organization."},
                status=status.HTTP_403_FORBIDDEN
            )

        duration = request.data.get("licence_duration", "monthly")

        if duration not in ["monthly", "yearly"]:
            return Response(
                {
                    "licence_duration":
                        "Invalid duration. Use monthly or yearly."
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        starting_date = request.data.get("licence_starting_date")
        expiration_date = calculate_expiration_date(duration, starting_date)

        serializer = OrganizationSerializer(
            data=request.data,
            context={}
        )

        serializer.is_valid(raise_exception=True)

        organization = serializer.save(
            licence_duration=duration,
            licence_expiration_date=expiration_date
        )

        organization.refresh_from_db()

        response_serializer = OrganizationSerializer(organization)
        response_data = response_serializer.data

        generated_password = serializer.context.get("generated_password")

        if generated_password:
            response_data["generated_password"] = generated_password

        return Response(
            response_data,
            status=status.HTTP_201_CREATED
        )

class RequesterEmailListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
            return Response({"error": "Admin only"}, status=status.HTTP_403_FORBIDDEN)

        emails = (
            User.objects
            .filter(is_requester=True, is_representative=False)
            .exclude(email__isnull=True)
            .exclude(email__exact='')
            .values_list('email', flat=True)
            .distinct()
            .order_by('email')
        )
        return Response(list(emails))

class OrganizationListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
            return Response(
                {"detail": "Only admin can view all organizations."},
                status=status.HTTP_403_FORBIDDEN
            )
        organizations = Organization.objects.select_related(
            'representative',
            'representative__user'
        ).order_by('-created_at')

        serializer = OrganizationSerializer(organizations, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class OrganizationDeleteAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, org_id):
        if not _is_admin_user(request.user):
            return Response(
                {"detail": "Only admin can delete an organization."},
                status=status.HTTP_403_FORBIDDEN
            )
        try:
            org = Organization.objects.select_related(
                'representative',
                'representative__user'
            ).get(id=org_id)
        except Organization.DoesNotExist:
            return Response({"error": "Organization not found"}, status=status.HTTP_404_NOT_FOUND)

        representative = org.representative
        representative_user = representative.user if representative and representative.user else None

        try:
            with transaction.atomic():
                OrganizationInvitation.objects.filter(organization=org).delete()
                OrganizationMembership.objects.filter(organization=org).delete()
                OrganizationLicenceKey.objects.filter(organization=org).delete()

                org.delete()

        except ProtectedError as e:
            print("PROTECTED ERROR =>", e)
            print("BLOCKING OBJECTS =>", e.protected_objects)
            return Response(
                {
                    "error": "Cannot delete organization",
                    "details": str(e)
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        if representative_user and representative:
            still_represents_other_org = Organization.objects.filter(representative=representative).exists()
            if not still_represents_other_org:
                representative_user.is_representative = False
                representative_user.save(update_fields=["is_representative"])

        return Response({"message": "Deleted successfully"}, status=status.HTTP_204_NO_CONTENT)

class OrganizationUpdateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    @transaction.atomic
    def put(self, request, org_id):
        if not _is_admin_user(request.user):
            return Response(
                {"detail": "Only admin can update an organization."},
                status=status.HTTP_403_FORBIDDEN
            )
        try:
            org = Organization.objects.get(id=org_id)
        except Organization.DoesNotExist:
            return Response({"error": "Organization not found"}, status=status.HTTP_404_NOT_FOUND)

        duration = request.data.get("licence_duration")
        starting_date = request.data.get("licence_starting_date")
        save_kwargs = {}
        if duration is not None:
            if duration not in ["monthly", "yearly"]:
                return Response(
                    {"licence_duration": "Invalid duration. Use monthly or yearly."},
                    status=status.HTTP_400_BAD_REQUEST
                )
        if starting_date is not None:
            parsed_starting_date = parse_date(starting_date) if isinstance(starting_date, str) else starting_date
            if not parsed_starting_date:
                return Response(
                    {"licence_starting_date": "Invalid date."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            if parsed_starting_date < org.created_at.date():
                return Response(
                    {
                        "licence_starting_date":
                            "License starting date cannot be before the organization creation date."
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        if duration is not None or starting_date is not None:
            effective_duration = duration or org.licence_duration
            effective_starting_date = starting_date or org.licence_starting_date
            save_kwargs["licence_expiration_date"] = calculate_expiration_date(
                effective_duration,
                effective_starting_date
            )

        serializer = OrganizationSerializer(org, data=request.data, partial=True, context={})
        serializer.is_valid(raise_exception=True)

        organization = serializer.save(**save_kwargs)
        organization.refresh_from_db()

        response_serializer = OrganizationSerializer(organization)
        response_data = response_serializer.data

        generated_password = serializer.context.get('generated_password')
        if generated_password:
            response_data['generated_password'] = generated_password

        return Response(response_data, status=status.HTTP_200_OK)

class RepresentativeEmailCheckAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
            return Response({"error": "Admin only"}, status=status.HTTP_403_FORBIDDEN)

        email = (request.GET.get("email") or "").strip().lower()
        org_id = (request.GET.get("org_id") or "").strip()

        if not email:
            return Response({
                "exists_in_user": False,
                "is_requester": False,
                "requester_is_activated": False,
                "already_representative_elsewhere": False,
            }, status=status.HTTP_200_OK)

        user = User.objects.filter(email__iexact=email).first()
        if not user:
            return Response({
                "exists_in_user": False,
                "is_requester": False,
                "requester_is_activated": False,
                "already_representative_elsewhere": False,
            }, status=status.HTTP_200_OK)

        already_representative_elsewhere = False

        requester = Requester.objects.filter(user=user).first()
        if requester:
            qs = Organization.objects.filter(representative=requester)
            if org_id:
                qs = qs.exclude(id=org_id)
            already_representative_elsewhere = qs.exists()

        return Response({
            "exists_in_user": True,
            "is_requester": bool(user.is_requester),
            "requester_is_activated": bool(user.is_active),
            "already_representative_elsewhere": already_representative_elsewhere,
        }, status=status.HTTP_200_OK)


class RoleRepresentativeCheckAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        can_activate = False
        activation_block_reason = None
        activation_block_message = None

        if getattr(user, "role", "") == "Requester" and getattr(user, "is_representative", False):
            requester = Requester.objects.filter(user=user).first()

            if requester:
                org = Organization.objects.filter(representative=requester).first()

                if org and org.status == "pending":
                    if org.is_contract_not_started:
                        activation_block_reason = "contract_not_started"
                        activation_block_message = org.contract_block_message
                    elif org.is_contract_expired:
                        activation_block_reason = "contract_expired"
                        activation_block_message = org.contract_block_message
                    else:
                        can_activate = True

        return Response({
            "id": user.id,
            "email": user.email,
            "role": getattr(user, "role", ""),
            "is_representative": bool(getattr(user, "is_representative", False)),
            "can_activate_organization": can_activate,
            "activation_block_reason": activation_block_reason,
            "activation_block_message": activation_block_message,
        })

class MyRepresentativeOrganizationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        if not getattr(user, "is_representative", False):
            return Response(
                {"detail": "You are not a representative."},
                status=status.HTTP_403_FORBIDDEN
            )

        requester = Requester.objects.filter(user=user).first()
        if not requester:
            return Response(
                {"detail": "Requester profile not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        organization = Organization.objects.filter(representative=requester).first()
        if not organization:
            return Response(
                {"detail": "No organization linked to this representative."},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = OrganizationSerializer(organization)
        return Response(serializer.data, status=status.HTTP_200_OK)





class OrganizationActivateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request, org_id):
        try:
            org = Organization.objects.get(id=org_id)
        except Organization.DoesNotExist:
            return Response(
                {"error": "Organization not found"},
                status=404
            )

        if org.representative.user != request.user:
            return Response(
                {"error": "Not allowed"},
                status=403
            )

        if org.status != "pending":
            return Response(
                {"error": "Organization already activated"},
                status=400
            )

        # Check contract validity
        today = timezone.localdate()

        if (
            org.licence_starting_date
            and today < org.licence_starting_date
        ):
            return Response(
                {
                    "error": "The organization contract has not started yet.",
                    "code": "contract_not_started",
                },
                status=400
            )

        if (
            org.licence_expiration_date
            and today >= org.licence_expiration_date
        ):
            return Response(
                {
                    "error": (
                        "The organization contract has expired. "
                        "Please renew the contract before activating the organization."
                    ),
                    "code": "contract_expired",
                },
                status=400
            )

        activation_code = (
            request.data.get("activation_code") or ""
        ).strip()

        if not activation_code:
            return Response(
                {"error": "activation_code is required"},
                status=400
            )

        description = (
            request.data.get("description") or ""
        ).strip()

        if description:
            org.description = description
            org.save(update_fields=["description"])

        try:
            org.activate(code=activation_code)
        except ValidationError as e:
            return Response(
                {
                    "error": (
                        e.messages
                        if hasattr(e, "messages")
                        else str(e)
                    )
                },
                status=400
            )

        org.refresh_from_db()

        return Response({
            "message": "Organization activated successfully",
            "status": org.status,
            "activated_at": org.activated_at,
            "description": org.description,
        })
class MyOrganizationsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user

        representative_orgs = Organization.objects.filter(
            representative__user=user
        )

        member_org_ids = OrganizationMembership.objects.filter(
            user=user,
            status='active'
        ).values_list('organization_id', flat=True)

        member_orgs = Organization.objects.filter(id__in=member_org_ids)

        organizations = (representative_orgs | member_orgs).distinct().order_by('name')

        data = []
        for org in organizations:
            membership = OrganizationMembership.objects.filter(
                organization=org,
                user=user,
                status='active'
            ).first()

            is_representative = bool(
                org.representative and org.representative.user_id == user.id
            )

            if is_representative:
                my_role = 'Requester'
            elif membership and membership.is_requester:
                my_role = 'Requester'
            else:
                my_role = 'Contributor'

            data.append({
                "id": org.id,
                "organisationName": org.name,
                "uniqueCode": org.id,
                "description": org.description,
                "status": org.status,
                "contractStatus": org.contract_status,
                "canAccess": org.status == "active" and org.contract_status == "active",
                "accessBlockReason": (
                    f"contract_{org.contract_status}"
                    if org.contract_status != "active"
                    else None
                ),
                "accessBlockMessage": org.contract_block_message,
                "myRole": my_role,
                "isRepresentative": is_representative,
                "country": getattr(org, "country", "") or "",
                "industrySector": getattr(org, "industry_sector", "") or "",
            })

        return Response(data, status=status.HTTP_200_OK)


class OrganizationDetailAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        org = get_object_or_404(Organization, id=org_id)

        if org.status != 'active':
            return Response(
                {'detail': 'Organization is not active yet.'},
                status=status.HTTP_403_FORBIDDEN
            )

        if org.representative and org.representative.user_id == request.user.id:
            if org.contract_status != "active":
                return _contract_block_response(org)
            serializer = OrganizationDetailSerializer(org, context={'request': request})
            return Response(serializer.data)

        membership = OrganizationMembership.objects.filter(
            organization=org,
            user=request.user,
            status='active'
        ).first()

        if not membership:
            return Response(
                {'detail': 'Access denied'},
                status=status.HTTP_403_FORBIDDEN
            )

        if org.contract_status != "active":
            return _contract_block_response(org)

        serializer = OrganizationDetailSerializer(org, context={'request': request})
        return Response(serializer.data)

class OrganizationMembersAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        org = get_object_or_404(Organization, id=org_id)

        is_representative = org.representative.user_id == request.user.id
        if not is_representative:
            return Response(
                {'detail': 'Only the representative can view organization members.'},
                status=status.HTTP_403_FORBIDDEN
            )

        if org.contract_status != "active":
            return _contract_block_response(org)

        members = OrganizationMembership.objects.filter(
            organization=org
        ).select_related('user').order_by('-joined_at')

        serializer = OrganizationMemberSerializer(members, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class OrganizationLicenseSummaryAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        org = get_object_or_404(Organization, id=org_id)

        is_representative = org.representative.user_id == request.user.id
        membership = OrganizationMembership.objects.filter(
            organization=org,
            user=request.user,
            status='active'
        ).first()

        if not is_representative and not membership:
            return Response({'detail': 'Access denied'}, status=status.HTTP_403_FORBIDDEN)

        if org.contract_status != "active":
            return _contract_block_response(org)

        requester_used = OrganizationMembership.objects.filter(
            organization=org,
            role='requester',
            status='active'
        ).count()

        contributor_used = OrganizationMembership.objects.filter(
            organization=org,
            role='contributor',
            status='active'
        ).count()

        return Response({
            'requesterTotal': org.licence_requester,
            'requesterUsed': requester_used,
            'contributorTotal': org.licence_contributor,
            'contributorUsed': contributor_used,
        }, status=status.HTTP_200_OK)

class OrganizationInviteMembersAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, org_id):
        org = get_object_or_404(Organization, id=org_id)

        if org.representative.user_id != request.user.id:
            return Response(
                {'detail': 'Only the representative can invite members.'},
                status=status.HTTP_403_FORBIDDEN
            )

        if org.status != 'active':
            return Response(
                {'detail': 'Organization must be active before sending invitations.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if org.contract_status != "active":
            return _contract_block_response(org)

        serializer = OrganizationInviteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        emails = [e.strip().lower() for e in serializer.validated_data['emails']]
        role = serializer.validated_data['role']
        duration_hours = DEFAULT_INVITATION_DURATION_HOURS
        custom_message = serializer.validated_data.get('message') or ''

        created = []
        errors = []
        invitations_to_email = []

        for email in emails:

            if email == org.representative.user.email.strip().lower():
                errors.append({
                    'email': email,
                    'error': 'The organization representative cannot invite themselves.'
                })
                continue

            if OrganizationMembership.objects.filter(
                organization=org,
                user__email__iexact=email
            ).exists():
                errors.append({
                    'email': email,
                    'error': 'This user already belongs to this organization.'
                })
                continue

            invitation = None
            invitation_url = None

            with transaction.atomic():
                existing_pending = (
                    OrganizationInvitation.objects
                    .select_for_update()
                    .filter(
                        organization=org,
                        email__iexact=email,
                        status='sent'
                    )
                    .first()
                )

                if existing_pending:
                    if existing_pending.is_expired():
                        existing_pending.status = 'expired'
                        existing_pending.responded_at = timezone.now()
                        existing_pending.save(update_fields=['status', 'responded_at'])
                        existing_pending.licence_key.reset_if_expired()
                    else:
                        errors.append({
                            'email': email,
                            'error': 'Pending invitation already exists.'
                        })
                        continue

                licence_key = (
                    OrganizationLicenceKey.objects
                    .select_for_update()
                    .filter(
                        organization=org,
                        role=role,
                        status='idle'
                    )
                    .order_by('id')
                    .first()
                )

                if not licence_key:
                    errors.append({
                        'email': email,
                        'error': f'No available {role} licence.'
                    })
                    continue

                invitation = OrganizationInvitation(
                    organization=org,
                    licence_key=licence_key,
                    email=email,
                    role=role,
                    duration_hours=duration_hours
                )
                invitation.send_invitation()

            invitation_url = (
                f"{settings.FRONTEND_URL.rstrip('/')}/organization-invitation"
                f"?token={invitation.token}"
            )

            invitations_to_email.append((email, invitation, invitation_url))

            created.append({
                'id': invitation.id,
                'email': invitation.email,
                'role': invitation.role,
                'status': invitation.status,
                'expires_at': invitation.expires_at,
                'invitation_url': invitation_url,
            })

        for email, invitation, invitation_url in invitations_to_email:
            send_organization_invitation_email(
                invited_email=email,
                organization=org,
                role=role,
                invitation=invitation,
                invited_by_name=request.user.fullName,
                invitation_url=invitation_url,
                custom_message=custom_message,
            )

        return Response({
            'created': created,
            'errors': errors,
        }, status=status.HTTP_200_OK)


class OrganizationOverviewAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        org = get_object_or_404(Organization, id=org_id)

        is_representative = org.representative.user_id == request.user.id
        membership = OrganizationMembership.objects.filter(
            organization=org,
            user=request.user,
            status='active'
        ).first()

        if not is_representative and not membership:
            return Response({'detail': 'Access denied'}, status=status.HTTP_403_FORBIDDEN)

        if org.contract_status != "active":
            return _contract_block_response(org)

        total_participants = OrganizationMembership.objects.filter(
            organization=org,
            status='active'
        ).count()

        # Placeholder backend stable for now
        total_events = 0
        published_events = 0
        draft_events = 0
        upcoming_events = 0
        past_events = 0
        events_this_month = 0

        avg_participants = 0 if total_events == 0 else round(total_participants / total_events)

        return Response({
            'totalEvents': total_events,
            'publishedEvents': published_events,
            'draftEvents': draft_events,
            'upcomingEvents': upcoming_events,
            'pastEvents': past_events,
            'totalParticipants': total_participants,
            'eventsTrend': 0,
            'avgParticipantsPerEvent': avg_participants,
            'completionRate': 0,
            'eventsThisMonth': events_this_month,
            'recentActivities': [],
        }, status=status.HTTP_200_OK)

class OrganizationInvitationCancelAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, invitation_id):
        invitation = get_object_or_404(OrganizationInvitation, id=invitation_id)

        if invitation.organization.representative.user_id != request.user.id:
            return Response(
                {"detail": "Only the representative can cancel this invitation."},
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            invitation.cancel()
        except ValidationError as e:
            return Response(
                {"detail": e.messages if hasattr(e, "messages") else str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )


        send_organization_invitation_cancelled_email(
            invited_email=invitation.email,
            organization=invitation.organization,
            role=invitation.role,
            cancelled_by_name=request.user.fullName,
        )

        return Response(
            {"message": "Invitation cancelled successfully."},
            status=status.HTTP_200_OK
        )


class OrganizationInvitationUpdateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, invitation_id):
        allowed_fields = {'role'}
        if set(request.data.keys()) - allowed_fields:
            return Response(
                {"detail": "Only role can be updated."},
                status=status.HTTP_400_BAD_REQUEST
            )

        role = request.data.get("role")
        if role not in ["requester", "contributor"]:
            return Response(
                {"detail": "Invalid role."},
                status=status.HTTP_400_BAD_REQUEST
            )

        role_changed = False
        with transaction.atomic():
            invitation = get_object_or_404(
                OrganizationInvitation.objects
                .select_for_update()
                .select_related("organization__representative__user", "licence_key"),
                id=invitation_id,
            )

            if invitation.organization.representative.user_id != request.user.id:
                return Response(
                    {"detail": "You are not allowed to update this invitation."},
                    status=status.HTTP_403_FORBIDDEN
                )

            if invitation.status != "sent":
                return Response(
                    {"detail": "Only pending invitations can be updated."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if invitation.role != role:
                new_licence_key = (
                    OrganizationLicenceKey.objects
                    .select_for_update()
                    .filter(
                        organization=invitation.organization,
                        role=role,
                        status='idle',
                    )
                    .order_by('id')
                    .first()
                )

                if not new_licence_key:
                    return Response(
                        {"detail": f"No available {role} licence."},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                old_licence_key = invitation.licence_key
                old_licence_key.status = 'idle'
                old_licence_key.invited_email = None
                old_licence_key.sent_at = None
                old_licence_key.expires_at = None
                old_licence_key.assigned_user = None
                old_licence_key.activated_at = None
                old_licence_key.save(update_fields=[
                    'status', 'invited_email', 'sent_at',
                    'expires_at', 'assigned_user', 'activated_at'
                ])

                new_licence_key.status = 'pending_activation'
                new_licence_key.invited_email = invitation.email
                new_licence_key.sent_at = invitation.sent_at
                new_licence_key.expires_at = invitation.expires_at
                new_licence_key.activated_at = None
                new_licence_key.assigned_user = None
                new_licence_key.save(update_fields=[
                    'status', 'invited_email', 'sent_at',
                    'expires_at', 'activated_at', 'assigned_user'
                ])

                invitation.role = role
                invitation.licence_key = new_licence_key
                invitation.save(update_fields=["role", "licence_key"])
                role_changed = True

        if role_changed:
            invitation_url = (
                f"{settings.FRONTEND_URL}/organization-invitation"
                f"?token={invitation.token}"
            )
            send_organization_invitation_email(
                invited_email=invitation.email,
                organization=invitation.organization,
                role=invitation.role,
                invitation=invitation,
                invited_by_name=request.user.fullName,
                invitation_url=invitation_url,
                custom_message='',
            )

        return Response({
            "id": invitation.id,
            "email": invitation.email,
            "role": invitation.role,
            "status": invitation.status,
            "sent_at": invitation.sent_at,
            "expires_at": invitation.expires_at,
        }, status=status.HTTP_200_OK)


class OrganizationInvitationsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, org_id):
        org = get_object_or_404(Organization, id=org_id)

        if org.representative.user_id != request.user.id:
            return Response(
                {"detail": "Only the representative can view invitations."},
                status=status.HTTP_403_FORBIDDEN
            )

        invitations = OrganizationInvitation.objects.filter(
            organization=org,
            status='sent'
        ).order_by('-sent_at')

        data = [
            {
                "id": inv.id,
                "email": inv.email,
                "role": "Requester" if inv.role == "requester" else "Contributor",
                "status": inv.status,
                "sent_at": inv.sent_at,
                "expires_at": inv.expires_at,
            }
            for inv in invitations
        ]

        return Response(data, status=status.HTTP_200_OK)

class OrganizationInvitationAcceptAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        token = (request.data.get('token') or '').strip()
        if not token:
            return Response(
                {'detail': 'token is required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        invitation = get_object_or_404(OrganizationInvitation, token=token)

        if invitation.email.strip().lower() != request.user.email.strip().lower():
            return Response(
                {'detail': 'This invitation does not belong to the current user.'},
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            invitation.accept(request.user)
        except ValidationError as e:
            return Response(
                {'detail': e.messages if hasattr(e, 'messages') else str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

        return Response({
            'message': 'Invitation accepted successfully.',
            'organization_code': invitation.organization.id
        }, status=status.HTTP_200_OK)

class OrganizationMembershipCreateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, org_id):
        return Response(
            {
                "detail": (
                    "Direct membership creation is disabled. "
                    "Use the organization invitation flow instead."
                )
            },
            status=status.HTTP_410_GONE
        )


class OrganizationInvitationDetailByTokenAPIView(APIView):
    permission_classes = []

    def get(self, request):
        token = (request.GET.get('token') or '').strip()
        if not token:
            return Response(
                {'detail': 'token is required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        invitation = get_object_or_404(OrganizationInvitation, token=token)

        if invitation.status == 'sent' and invitation.is_expired():
            invitation.status = 'expired'
            invitation.responded_at = timezone.now()
            invitation.save(update_fields=['status', 'responded_at'])
            invitation.licence_key.reset_if_expired()

        is_authenticated = bool(request.user and request.user.is_authenticated)
        email_matches_current_user = bool(
            is_authenticated and
            request.user.email.strip().lower() == invitation.email.strip().lower()
        )

        data = {
            'organizationName': invitation.organization.name,
            'organizationCode': invitation.organization.id,
            'role': invitation.role,
            'status': invitation.status,
            'expiresAt': invitation.expires_at,
            'isExpired': invitation.status == 'expired',
            'isAuthenticated': is_authenticated,
        }

        if is_authenticated:
            user = User.objects.filter(email__iexact=invitation.email).first()
            data.update({
                'email': invitation.email if email_matches_current_user else None,
                'emailMatchesCurrentUser': email_matches_current_user,
                'accountExists': email_matches_current_user and bool(user),
                'accountActive': email_matches_current_user and bool(user and user.is_active),
            })

        return Response(data, status=status.HTTP_200_OK)

class OrganizationInvitationDeclineAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        token = (request.data.get('token') or '').strip()
        if not token:
            return Response(
                {'detail': 'token is required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        invitation = get_object_or_404(OrganizationInvitation, token=token)

        if invitation.email.strip().lower() != request.user.email.strip().lower():
            return Response(
                {'detail': 'This invitation does not belong to the current user.'},
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            invitation.decline()
        except ValidationError as e:
            return Response(
                {'detail': e.messages if hasattr(e, 'messages') else str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )

        return Response({
            'message': 'Invitation declined successfully.'
        }, status=status.HTTP_200_OK)


class MyRequesterOrganizationsAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        requester = Requester.objects.filter(user=user).first()

        membership_orgs = Organization.objects.filter(
            memberships__user=user,
            memberships__role='requester',
            memberships__status='active'
        )

        representative_orgs = Organization.objects.none()
        if requester:
            representative_orgs = Organization.objects.filter(
                representative=requester
            )

        organizations = (membership_orgs | representative_orgs).distinct()

        data = [
            {
                "id": org.id,
                "name": org.name
            }
            for org in organizations
        ]

        return Response(data, status=status.HTTP_200_OK)
