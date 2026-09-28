import secrets

from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import transaction
from django.utils import timezone
from django.utils.crypto import constant_time_compare
from rest_framework.permissions import IsAuthenticated
from django.http import JsonResponse, HttpResponse
from django.shortcuts import redirect, get_object_or_404
from urllib.parse import quote
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.shortcuts import redirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken
from django.conf import settings

from mobicrowd.authentication.cookie_auth import refresh_cookie_name, clear_auth_cookies
from mobicrowd.authentication.email_sending import send_password_changed, \
    send_forget_password_email, send_requester_access_approved_email, send_requester_access_rejected_email
from mobicrowd.models.Users import User, Token, Requester, Worker, PasswordResetToken
from mobicrowd.models.notifications import DeviceToken
from mobicrowd.views.apis.notify_helpers import make_payload, notify_admins, notify_workers
import logging

logger = logging.getLogger(__name__)

def _is_admin_user(user):
    return bool(
        getattr(user, "role", "") == "Admin"
        or getattr(user, "is_superuser", False)
    )


def _redirect_page(target_url: str, title: str, message: str) -> HttpResponse:
    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta http-equiv="refresh" content="1;url={target_url}">
        <title>{title}</title>
        <style>
          body{{font-family:Arial, sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;background:#f7f7f7;margin:0}}
          .box{{background:#fff;padding:24px 28px;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,.08);max-width:560px}}
          h2{{margin:0 0 10px;font-size:20px}}
          p{{margin:0;color:#555;line-height:1.45}}
          .hint{{margin-top:10px;font-size:12px;color:#777}}
        </style>
      </head>
      <body>
        <div class="box">
          <h2>{title}</h2>
          <p>{message}</p>
          <p class="hint">Redirecting…</p>
        </div>
        <script>
          setTimeout(() => window.location.href = "{target_url}", 20000);
        </script>
      </body>
    </html>
    """
    return HttpResponse(html)


class ConfirmEmailAPIView(APIView):
    """
    Email button should point to:
      GET /api/confirm-email/<pk>/<token>/

    This endpoint:
      - validates token
      - activates the user
      - deletes the stored token
      - shows a friendly confirmation page
      - redirects to frontend /login with a status flag
    """

    authentication_classes = []  # allow public access
    permission_classes = []      # allow public access

    def get(self, request, pk, token):
        # Where to send the user after confirmation
        frontend_login_ok = f"{settings.FRONTEND_URL}/login?verified=1"
        frontend_login_fail = f"{settings.FRONTEND_URL}/login?verified=0"

        try:
            user = User.objects.get(pk=pk)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            return _redirect_page(
                frontend_login_fail,
                "Account not found",
                "This confirmation link is invalid. Please try signing up again or contact support."
            )

        token_obj = Token.objects.filter(user=user).first()
        token_generator = PasswordResetTokenGenerator()

        # Validate both token schemes (your current logic)
        is_valid = bool(token_obj) and token_generator.check_token(user, token) and token_obj.token == token

        if not is_valid:
            return _redirect_page(
                frontend_login_fail,
                "Link invalid or expired",
                "This confirmation link is invalid or has expired. Please request a new verification email."
            )

        # Activate + invalidate token
        user.is_active = True
        user.save(update_fields=["is_active"])
        token_obj.delete()

        return _redirect_page(
            frontend_login_ok,
            "Email verified ✅",
            "Your account has been confirmed successfully. You can log in now."
        )


class RequesterApprovalAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, email):
        if not _is_admin_user(request.user):
            return Response(
                {"error": "Admin only"},
                status=status.HTTP_403_FORBIDDEN,
            )

        email = str(email or "").strip().lower()

        logger.info(
            "Requester approval started email=%s admin_id=%s",
            email,
            getattr(request.user, "id", None),
        )

        user = User.objects.filter(email__iexact=email).first()

        if not user:
            logger.warning(
                "Requester approval failed: user not found email=%s",
                email,
            )
            return Response(
                {"error": "User not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        requester_profile = Requester.objects.filter(user=user).first()

        logger.info(
            (
                "Requester approval state user_id=%s email=%s "
                "is_worker=%s is_requester=%s "
                "requester_request_pending=%s requester_profile=%s "
                "requester_approved=%s"
            ),
            user.id,
            user.email,
            user.is_worker,
            user.is_requester,
            user.requester_request_pending,
            bool(requester_profile),
            getattr(requester_profile, "approved", None),
        )

        # ============================================================
        # CASE 1 -- EXISTING CONTRIBUTOR REQUESTING REQUESTER ACCESS
        #
        # This MUST be checked before Requester.approved=False.
        # An existing contributor can already have a Requester row, and
        # checking that row first would incorrectly classify the request
        # as a new requester account and skip the contributor notification.
        # ============================================================
        if user.requester_request_pending and not user.is_requester:
            logger.info(
                "Approving contributor requester access user_id=%s email=%s",
                user.id,
                user.email,
            )

            with transaction.atomic():
                locked_user = (
                    User.objects
                    .select_for_update()
                    .get(pk=user.pk)
                )

                # Recheck under the lock so a duplicated admin action does
                # not process the same pending request twice.
                if not locked_user.requester_request_pending:
                    return Response(
                        {
                            "success": "Requester access was already processed.",
                            "request_type": "already_processed",
                            "user_id": locked_user.id,
                            "email": locked_user.email,
                        },
                        status=status.HTTP_200_OK,
                    )

                locked_user.is_requester = True
                locked_user.is_worker = True
                locked_user.requester_request_pending = False
                locked_user.save(
                    update_fields=[
                        "is_requester",
                        "is_worker",
                        "requester_request_pending",
                    ]
                )

                requester_obj, created = Requester.objects.get_or_create(
                    user=locked_user,
                    defaults={
                        "approved": True,
                        "organization_name": "",
                        "location": "",
                    },
                )

                if not requester_obj.approved:
                    requester_obj.approved = True
                    requester_obj.save(update_fields=["approved"])

                user = locked_user

            logger.info(
                (
                    "Requester capability granted user_id=%s email=%s "
                    "requester_created=%s"
                ),
                user.id,
                user.email,
                created,
            )

            # --------------------------------------------------------
            # Approval email -- NOT a verification email.
            # --------------------------------------------------------
            try:
                logger.info(
                    "Sending requester access approval email user_id=%s email=%s",
                    user.id,
                    user.email,
                )
                send_requester_access_approved_email(user)
                logger.info(
                    "Requester access approval email sent user_id=%s email=%s",
                    user.id,
                    user.email,
                )
            except Exception:
                logger.exception(
                    (
                        "Requester access approval email FAILED "
                        "user_id=%s email=%s"
                    ),
                    user.id,
                    user.email,
                )

            # --------------------------------------------------------
            # Persistent + realtime/push notification to contributor.
            # notify_workers() internally targets each Worker.user_id.
            # --------------------------------------------------------
            workers = Worker.objects.filter(user_id=user.id)
            worker_count = workers.count()

            logger.info(
                (
                    "Requester approval notification preparation "
                    "user_id=%s worker_profiles=%s"
                ),
                user.id,
                worker_count,
            )

            if worker_count == 0:
                logger.error(
                    (
                        "Requester approval notification NOT sent: "
                        "no Worker profile exists user_id=%s email=%s"
                    ),
                    user.id,
                    user.email,
                )
            else:
                try:
                    notify_workers(
                        workers=workers,
                        event_type="requester.access.approved",
                        title="Requester access approved",
                        body=(
                            "Your request to become a Requester has been "
                            "approved. You can now switch to Requester Mode."
                        ),
                        payload=make_payload(
                            type="requester.access",
                            status="approved",
                            user_id=user.id,
                        ),
                        priority="high",
                    )
                    logger.info(
                        "Requester approval notification created user_id=%s",
                        user.id,
                    )
                except Exception:
                    logger.exception(
                        (
                            "Requester approval notification FAILED "
                            "user_id=%s email=%s"
                        ),
                        user.id,
                        user.email,
                    )

            return Response(
                {
                    "success": "Requester access approved successfully.",
                    "request_type": "access_request",
                    "user_id": user.id,
                    "email": user.email,
                },
                status=status.HTTP_200_OK,
            )

        # ============================================================
        # CASE 2 -- NEW REQUESTER ACCOUNT WAITING FOR APPROVAL
        #
        # There is deliberately NO verification email here. The account
        # receives the requester-access approval email instead.
        # ============================================================
        requester = (
            Requester.objects
            .filter(user=user, approved=False)
            .select_related("user")
            .first()
        )

        if requester:
            logger.info(
                "Approving new requester account user_id=%s email=%s",
                user.id,
                user.email,
            )

            with transaction.atomic():
                requester = (
                    Requester.objects
                    .select_for_update()
                    .select_related("user")
                    .get(pk=requester.pk)
                )
                user = requester.user

                requester.approved = True
                requester.save(update_fields=["approved"])

                user.is_requester = True
                user.is_worker = True
                user.requester_request_pending = False
                user.save(
                    update_fields=[
                        "is_requester",
                        "is_worker",
                        "requester_request_pending",
                    ]
                )

            # Approval email only; no send_verification_email().
            try:
                logger.info(
                    "Sending requester account approval email user_id=%s email=%s",
                    user.id,
                    user.email,
                )
                send_requester_access_approved_email(user)
                logger.info(
                    "Requester account approval email sent user_id=%s email=%s",
                    user.id,
                    user.email,
                )
            except Exception:
                logger.exception(
                    (
                        "Requester account approval email FAILED "
                        "user_id=%s email=%s"
                    ),
                    user.id,
                    user.email,
                )

            # If this account has a Worker profile, notify it as well.
            workers = Worker.objects.filter(user_id=user.id)
            worker_count = workers.count()

            logger.info(
                (
                    "Requester account approval notification preparation "
                    "user_id=%s worker_profiles=%s"
                ),
                user.id,
                worker_count,
            )

            if worker_count > 0:
                try:
                    notify_workers(
                        workers=workers,
                        event_type="requester.access.approved",
                        title="Requester access approved",
                        body=(
                            "Your Requester account has been approved. "
                            "You can now use Requester Mode."
                        ),
                        payload=make_payload(
                            type="requester.access",
                            status="approved",
                            user_id=user.id,
                        ),
                        priority="high",
                    )
                    logger.info(
                        "Requester account approval notification created user_id=%s",
                        user.id,
                    )
                except Exception:
                    logger.exception(
                        (
                            "Requester account approval notification FAILED "
                            "user_id=%s email=%s"
                        ),
                        user.id,
                        user.email,
                    )
            else:
                logger.warning(
                    (
                        "Requester account approved without Worker notification: "
                        "no Worker profile user_id=%s email=%s"
                    ),
                    user.id,
                    user.email,
                )

            return Response(
                {
                    "success": "Requester account approved successfully.",
                    "request_type": "new_account",
                    "user_id": user.id,
                    "email": user.email,
                },
                status=status.HTTP_200_OK,
            )

        # ============================================================
        # ALREADY APPROVED
        # ============================================================
        if user.is_requester:
            logger.info(
                "Requester approval ignored: already approved user_id=%s email=%s",
                user.id,
                user.email,
            )
            return Response(
                {
                    "success": "User already has requester access.",
                    "request_type": "already_approved",
                    "user_id": user.id,
                    "email": user.email,
                },
                status=status.HTTP_200_OK,
            )

        # ============================================================
        # NO PENDING REQUEST
        # ============================================================
        logger.warning(
            "Requester approval failed: no pending request user_id=%s email=%s",
            user.id,
            user.email,
        )
        return Response(
            {"error": "Pending requester request not found."},
            status=status.HTTP_404_NOT_FOUND,
        )


class ActiveWorkersListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
            return Response({"error": "Admin only"}, status=status.HTTP_403_FORBIDDEN)

        # Filter users who are active and are workers
        active_users = User.objects.filter(is_active=True, is_worker=True)

        # Fetch related Worker data
        workers = Worker.objects.filter(user__in=active_users)

        # Prepare the data to be returned
        data = [
            {
                'id': worker.user.id,
                'fullName': worker.user.fullName,
                'email': worker.user.email,
                'location': worker.location,
                'device_specs': worker.device_specs,
                'mobile_phone': worker.user.mobile_phone,
                'is_requester': worker.user.is_requester,
                # Include other fields as needed
            }
            for worker in workers
        ]

        return Response(data, status=status.HTTP_200_OK)


class LogoutAPIView(APIView):
    permission_classes = [
        IsAuthenticated
    ]

    def post(self, request):
        tok = request.auth

        device_uid = (
            tok.get("device_uid")
            if tok
            else None
        )

        sid = (
            tok.get("sid")
            if tok
            else None
        )

        kind = (
            tok.get("kind")
            if tok
            else None
        )

        # Backward-compatible native fallback.
        if not device_uid:
            device_uid = (
                request.data.get(
                    "device_uid"
                )
                or ""
            ).strip()

        if not device_uid:
            return Response(
                {
                    "msg": (
                        "device_uid required"
                    )
                },
                status=(
                    status
                    .HTTP_400_BAD_REQUEST
                ),
            )

        # -------------------------------------------------------------
        # Revoke Takleef server-side session.
        # -------------------------------------------------------------

        session_qs = (
            DeviceToken.objects
            .filter(
                user=request.user,
                device_uid=device_uid,
            )
        )

        if sid:
            session_qs = (
                session_qs.filter(
                    session_sid=sid
                )
            )

        if kind in (
            "APP",
            "DASHBOARD",
        ):
            session_qs = (
                session_qs.filter(
                    kind=kind
                )
            )

        session_qs.update(
            session_active=False,
            session_sid="",
            is_active=False,
            session_last_seen=(
                timezone.now()
            ),
        )

        # -------------------------------------------------------------
        # Development/native:
        # refresh comes from JSON.
        #
        # Production web:
        # refresh comes from HttpOnly cookie.
        # -------------------------------------------------------------

        raw_refresh = str(
            request.data.get(
                "refresh"
            )
            or ""
        ).strip()

        if (
            not raw_refresh
            and getattr(
                settings,
                "JWT_COOKIE_AUTH_ENABLED",
                False,
            )
        ):
            raw_refresh = str(
                request.COOKIES.get(
                    refresh_cookie_name()
                )
                or ""
            ).strip()

        if raw_refresh:
            try:
                refresh = RefreshToken(
                    raw_refresh
                )

                same_user = (
                    str(
                        refresh.get(
                            "user_id"
                        )
                    )
                    ==
                    str(request.user.pk)
                )

                same_device = (
                    refresh.get(
                        "device_uid"
                    )
                    == device_uid
                )

                same_sid = (
                    not sid
                    or refresh.get("sid")
                    == sid
                )

                if (
                    same_user
                    and same_device
                    and same_sid
                ):
                    refresh.blacklist()

            except (
                TokenError,
                AttributeError,
            ):
                # Logout remains idempotent.
                pass

        response = Response(
            {
                "message": (
                    "Successfully logged out."
                )
            },
            status=status.HTTP_200_OK,
        )

        # Critical for production web:
        # delete both HttpOnly JWT cookies.
        clear_auth_cookies(
            response
        )

        return response

class RequesterRejectionAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, email):
        if not _is_admin_user(request.user):
            return Response({"error": "Admin only"}, status=status.HTTP_403_FORBIDDEN)

        # Case 1: reject and remove a new requester account.
        requester = Requester.objects.filter(
            user__email=email,
            approved=False,
        ).select_related("user").first()

        if requester:
            user = requester.user
            send_requester_access_rejected_email(user)

            notify_admins(
                event_type="requester.account.rejected",
                title="🚫 Requester rejected",
                body=f"{email} was rejected and removed.",
                payload=make_payload(type="requester.review", requester_email=email, status="rejected"),
                priority="high",
            )

            user.delete()
            return Response(
                {'success': 'Requester account rejected and deleted successfully.'},
                status=status.HTTP_200_OK,
            )

        # Case 2: reject requester access for an existing contributor/worker.
        user = User.objects.filter(
            email=email,
            requester_request_pending=True,
            is_requester=False,
        ).first()

        if user:
            user.requester_request_pending = False
            user.save(update_fields=["requester_request_pending"])

            send_requester_access_rejected_email(user)

            notify_admins(
                event_type="requester.account.rejected",
                title="🚫 Requester access rejected",
                body=f"{email} requester access request was rejected.",
                payload=make_payload(type="requester.review", requester_email=email, status="rejected"),
                priority="high",
            )

            return Response(
                {'success': 'Requester access request rejected successfully.'},
                status=status.HTTP_200_OK,
            )

        return Response(
            {'error': 'Pending requester request not found.'},
            status=status.HTTP_404_NOT_FOUND,
        )


class ApprovedRequestersListAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
            return Response({"error": "Admin only"}, status=status.HTTP_403_FORBIDDEN)

        approved_requesters = Requester.objects.filter(approved=True).select_related('user')
        data = [
            {

                'fullName': requester.user.fullName,
                'email': requester.user.email,
                'organization_name': requester.organization_name,
                'location': requester.location,
                'mobile_phone': requester.user.mobile_phone,
            }
            for requester in approved_requesters
        ]
        return Response(data, status=status.HTTP_200_OK)


class ApprovedRequestersListcountAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if not _is_admin_user(request.user):
            return Response({"error": "Admin only"}, status=status.HTTP_403_FORBIDDEN)

        new_accounts_count = Requester.objects.filter(approved=False).count()
        access_requests_count = User.objects.filter(
            requester_request_pending=True,
            is_requester=False,
        ).count()
        return JsonResponse({'not_approved_count': new_accounts_count + access_requests_count})

class ChangePasswordAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user

        old_password = request.data.get("old_password")
        new_password = request.data.get("password")
        confirm_password = request.data.get("confirm_password")

        if not old_password or not new_password or not confirm_password:
            return Response(
                {"error": "old_password, password, and confirm_password are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if new_password != confirm_password:
            return Response(
                {"error": "Passwords do not match."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not user.check_password(old_password):
            return Response(
                {"error": "Old password is incorrect."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            validate_password(new_password, user=user)
        except ValidationError as exc:
            return Response(
                {"error": exc.messages},
                status=status.HTTP_400_BAD_REQUEST,
            )

        user.set_password(new_password)
        user.save(update_fields=["password"])

        DeviceToken.objects.filter(user=user).update(
            session_active=False,
            session_sid="",
            is_active=False,
            session_last_seen=timezone.now(),
        )

        send_password_changed(user)

        return Response(
            {"message": "Password changed successfully. Please log in again."},
            status=status.HTTP_200_OK,
        )


class CustomPasswordResetTokenGenerator(PasswordResetTokenGenerator):
    def _make_hash_value(self, user, timestamp):
        return str(user.pk) + str(user.password) + str(timestamp)


def password_forgotten_reset(user):
    raw_token = secrets.token_urlsafe(48)

    # Invalidate old active links for this user
    PasswordResetToken.objects.filter(
        user=user,
        used_at__isnull=True
    ).update(used_at=timezone.now())

    rec = PasswordResetToken.objects.create(
        user=user,
        token_hash=PasswordResetToken.hash_token(raw_token),
    )

    reset_password_link = (
        f'{settings.FRONTEND_URL.rstrip("/")}/set-new-password/'
        f'{rec.selector}/{raw_token}'
    )

    send_forget_password_email(user, reset_password_link)

    return Response({'message': 'Email sent'}, status=status.HTTP_200_OK)

class ForgetPasswordView(APIView):
    RESET_RESPONSE = {
        "message": "If an account exists for this email, a password reset link has been sent."
    }

    def get(self, request, email):
        user = User.objects.filter(email=email).first()

        if user:
            password_forgotten_reset(user)

        return Response(self.RESET_RESPONSE, status=status.HTTP_200_OK)

class PasswordResetConfirmView(APIView):

    def _invalid_response(self):
        return Response(
            {'message': 'Invalid or expired link'},
            status=status.HTTP_400_BAD_REQUEST
        )

    def _get_valid_record(self, selector, token):
        rec = PasswordResetToken.objects.select_related("user").filter(
            selector=selector,
            used_at__isnull=True
        ).first()

        if not rec:
            return None

        expected_hash = PasswordResetToken.hash_token(token)

        if not constant_time_compare(rec.token_hash, expected_hash):
            return None

        if rec.is_expired():
            return None

        return rec

    def get(self, request, selector, token):
        rec = self._get_valid_record(selector, token)

        if not rec:
            return self._invalid_response()

        return Response(
            {'message': 'Valid reset link'},
            status=status.HTTP_200_OK
        )

    def post(self, request, selector, token):
        password = request.data.get('password')
        confirm_password = request.data.get('confirm_password')

        if not password or password != confirm_password:
            return Response(
                {'message': 'Passwords do not match'},
                status=status.HTTP_400_BAD_REQUEST
            )

        with transaction.atomic():
            rec = PasswordResetToken.objects.select_for_update().select_related("user").filter(
                selector=selector,
                used_at__isnull=True
            ).first()

            if not rec:
                return self._invalid_response()

            expected_hash = PasswordResetToken.hash_token(token)

            if not constant_time_compare(rec.token_hash, expected_hash):
                return self._invalid_response()

            if rec.is_expired():
                return self._invalid_response()

            user = rec.user

            try:
                validate_password(password, user=user)
            except ValidationError as exc:
                return Response(
                    {'message': exc.messages},
                    status=status.HTTP_400_BAD_REQUEST
                )

            user.set_password(password)
            user.save(update_fields=["password"])
            DeviceToken.objects.filter(user=user).update(
                session_active=False,
                session_sid="",
                is_active=False,
                session_last_seen=timezone.now(),
            )

            # Burn the token immediately
            rec.used_at = timezone.now()
            rec.save(update_fields=["used_at"])

            # Invalidate any other active reset links for the same user
            PasswordResetToken.objects.filter(
                user=user,
                used_at__isnull=True
            ).exclude(pk=rec.pk).update(used_at=timezone.now())

        send_password_changed(user)

        return Response(
            {'message': 'Password updated successfully'},
            status=status.HTTP_200_OK
        )