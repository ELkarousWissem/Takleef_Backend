import logging
import re
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db.models import Q
from django.shortcuts import get_object_or_404
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import generics, status

from mobicrowd.models.Users import Worker, User, Requester
from mobicrowd.serializers.usersSerializers import WorkerSerializer, UserSerializer, RequesterSerializer, \
    ProfilePhotoUpdateSerializer

from django_filters import rest_framework as filters

from rest_framework import permissions
logger = logging.getLogger(__name__)


def _is_admin_user(user):
    return bool(
        getattr(user, "role", "") == "Admin"
        or getattr(user, "is_superuser", False)
    )


def _error_response(message, status_code):
    return Response({"error": message, "message": message}, status=status_code)


class UserRetrieveAPIView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = UserSerializer
    lookup_field = 'id' # Defines the name of the URL parameter

    def get_queryset(self):
        if _is_admin_user(self.request.user):
            return User.objects.all()
        return User.objects.filter(id=self.request.user.id)

class RequesterRetrieveAPIView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = RequesterSerializer
    lookup_field = 'user_id'  # Lookup by 'user_id' which is the primary key of the Requester model

    def get_queryset(self):
        if _is_admin_user(self.request.user):
            return Requester.objects.all()
        return Requester.objects.filter(user=self.request.user)

class ApprovedRequestersNamesView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request, *args, **kwargs):
        try:
            approved_requesters = Requester.objects.filter(approved=True).values_list('organization_name', flat=True)
            return Response(list(approved_requesters), status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ListRequestersView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request):
        Requesters = Requester.objects.filter(user__is_active=True)
        serializer = RequesterSerializer(Requesters, many=True)
        return Response(serializer.data)

class ListPendingRequestersView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
            return Response({"error": "Admin only"}, status=status.HTTP_403_FORBIDDEN)

        # 1) New requester accounts waiting for admin approval
        pending_requesters = Requester.objects.filter(approved=False).select_related("user")

        requester_rows = [
            {
                "request_type": "new_account",
                "organization_name": requester.organization_name,
                "location": requester.location,
                "user": {
                    "email": requester.user.email,
                    "mobile_phone": requester.user.mobile_phone,
                }
            }
            for requester in pending_requesters
        ]

        # 2) Existing users requesting requester access
        pending_access_users = User.objects.filter(
            requester_request_pending=True,
            is_requester=False
        )

        access_rows = [
            {
                "request_type": "access_request",
                "organization_name": "-",
                "location": getattr(user, "location", "") or "-",
                "user": {
                    "email": user.email,
                    "mobile_phone": user.mobile_phone,
                }
            }
            for user in pending_access_users
        ]

        return Response(requester_rows + access_rows, status=status.HTTP_200_OK)

class RequestersFilter(filters.FilterSet):
    permission_classes = [permissions.IsAuthenticated]
    name = filters.CharFilter(method='filter_name')

    def filter_name(self, queryset, name, value):
        return queryset.filter(
            Q(user__fullName__iregex=r'\b%s' % re.escape(value))
        )

    class Meta:
        model = Requester
        fields = ['name']


class RequesterFilterAPIView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]
    queryset = Requester.objects.filter(user__is_active=True, approved=True)
    serializer_class = RequesterSerializer
    filter_backends = [filters.DjangoFilterBackend]
    filterset_class = RequestersFilter



class WorkerDetailView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]
    queryset = Worker.objects.all()
    serializer_class = WorkerSerializer
    lookup_url_kwarg = 'id'
    lookup_field = 'user_id'

class RequesterLocationView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, user_id: int = None):
        if user_id is None:
            user_id = request.user.id

        requester = get_object_or_404(
            Requester.objects.select_related("user"),
            user_id=user_id
        )

        data = RequesterSerializer(requester).data

        # return only what you need (location), but still “using the serializer”
        return Response(
            {
                "user_id": data["user"],
                "location": data["location"],
            },
            status=status.HTTP_200_OK
        )

    def post(self, request, user_id: int = None):
        if user_id is not None and user_id != request.user.id and not _is_admin_user(request.user):
            return _error_response("Not allowed", status.HTTP_403_FORBIDDEN)

        requester = get_object_or_404(
            Requester.objects.select_related("user"),
            user_id=request.user.id if user_id is None else user_id,
        )

        location = request.data.get("location")
        if location is None or not str(location).strip():
            return _error_response("location is required.", status.HTTP_400_BAD_REQUEST)

        requester.location = str(location).strip()
        requester.save(update_fields=["location"])

        return Response(
            {
                "user_id": requester.user_id,
                "location": requester.location,
            },
            status=status.HTTP_200_OK,
        )


class UserPhoneView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        mobile_phone = request.data.get("mobile_phone")
        if mobile_phone is None or not str(mobile_phone).strip():
            return _error_response("mobile_phone is required.", status.HTTP_400_BAD_REQUEST)

        request.user.mobile_phone = str(mobile_phone).strip()
        request.user.save(update_fields=["mobile_phone"])

        return Response(
            {
                "user_id": request.user.id,
                "mobile_phone": request.user.mobile_phone,
            },
            status=status.HTTP_200_OK,
        )


class UserEmailView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        email = request.data.get("email")
        if email is None or not str(email).strip():
            return _error_response("email is required.", status.HTTP_400_BAD_REQUEST)

        email = str(email).strip().lower()
        try:
            validate_email(email)
        except ValidationError:
            return _error_response("Invalid email address.", status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(email__iexact=email).exclude(pk=request.user.pk).exists():
            return _error_response("This email is already in use.", status.HTTP_400_BAD_REQUEST)

        request.user.email = email
        request.user.save(update_fields=["email"])

        return Response(
            {
                "user_id": request.user.id,
                "email": request.user.email,
            },
            status=status.HTTP_200_OK,
        )


class ProfilePhotoAPIView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request):
        user = request.user

        profile_photo_url = None

        if user.profile_photo:
            try:
                profile_photo_url = user.profile_photo.url
            except (ValueError, AttributeError):
                profile_photo_url = None

        return Response(
            {
                "profile_photo": profile_photo_url
            },
            status=status.HTTP_200_OK
        )

    def patch(self, request):
        user = request.user

        previous_photo_name = None
        previous_photo_storage = None

        if user.profile_photo:
            previous_photo_name = user.profile_photo.name
            previous_photo_storage = user.profile_photo.storage

        serializer = ProfilePhotoUpdateSerializer(
            user,
            data=request.data,
            partial=True
        )
        serializer.is_valid(raise_exception=True)

        updated_user = serializer.save()

        new_photo_name = (
            updated_user.profile_photo.name
            if updated_user.profile_photo
            else None
        )

        # Remove the previous S3 object after the new photo is saved.
        if (
            previous_photo_name
            and previous_photo_storage
            and previous_photo_name != new_photo_name
        ):
            try:
                previous_photo_storage.delete(previous_photo_name)
            except Exception:
                logger.exception(
                    "Could not delete previous profile photo: %s",
                    previous_photo_name
                )

        profile_photo_url = (
            updated_user.profile_photo.url
            if updated_user.profile_photo
            else None
        )

        return Response(
            {
                "message": "Profile photo updated successfully.",
                "profile_photo": profile_photo_url
            },
            status=status.HTTP_200_OK
        )
