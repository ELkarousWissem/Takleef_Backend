import random

from django.utils.timezone import now
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

from mobicrowd.authentication.email_sending import send_verification_email, send_otp_email, \
    send_account_pending_approval
from mobicrowd.models.Users import User, Worker
from mobicrowd.serializers.usersSerializers import RequesterSerializer, WorkerSerializer, UserSerializer

from allauth.account.models import EmailAddress

from mobicrowd.views.apis.notify_helpers import notify_admins, make_payload


def email_address_exists(email):
    return EmailAddress.objects.filter(email=email).exists()

class RegisterRequesterAPIView(APIView):

    def post(self, request):

        if User.objects.filter(email=request.data.get('email')).exists():
            return Response({'message': 'User with this email already exists.'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            serializer = RequesterSerializer(data={
                'user': {
                    'email': request.data.get('email'),
                    'password': request.data.get('password'),
                    'fullName': request.data.get('fullName'),
                    'mobile_phone': request.data.get('mobile_phone'),
                    'is_requester': True,
                    'is_worker': True,
                    'is_active': False,
                    'role': 'Requester',
                },
                # Organization membership is now managed through the organization invitation/access flow.
                # Keep this field out of open requester registration to avoid trusting user-supplied organization names.
                'location': request.data.get('location'),

            })

            if serializer.is_valid():
                serializer.save()
                notify_admins(
                    event_type="requester.review",
                    title="🆕 New requester awaiting approval",
                    body=f"{request.data.get('fullName')} ({request.data.get('email')})",
                    payload=make_payload(type="requester.review",
                                         body_for_ui=f"{request.data.get('fullName')} ({request.data.get('email')})",
                                         requester_email=request.data.get('email')),

                    priority="high",
                )
                send_account_pending_approval(request.data.get('email'),request.data.get('fullName'))
                return Response({'message': 'Your account is pending approval from the admin. Once approved, you will receive a verification email'}, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RegisterWorkerAPIView(APIView):
    def post(self, request):

        if User.objects.filter(email=request.data.get('email')).exists():
            return Response({'message': 'User with this email already exists.'}, status=status.HTTP_400_BAD_REQUEST)

        else:
            serializer = WorkerSerializer(data={
                'user': {
                    'email': request.data.get('email'),
                    'password': request.data.get('password'),
                    'fullName': request.data.get('fullName'),
                    'mobile_phone': request.data.get('mobile_phone'),
                    'is_worker': True,
                    'is_active': False,
                    'role': 'Worker',

                },
                'location': request.data.get('location'),
                'device_specs': request.data.get('device_specs'),

            })

            if serializer.is_valid():
                # Save the competitor and get the user instance from it
                worker = serializer.save()
                user = worker.user

                # Now you can call your send_verification_email function with the user
                send_otp_email(user)

                return Response({'message': 'We have just sent you an OTP verification email. The OTP lasts only 10 minutes.'},
                                status=status.HTTP_201_CREATED)
                # 🚀 Debugging: Log the exact serializer errors
            print(f"❌ Serializer errors: {serializer.errors}")  # Debugging output

            return Response({'errors': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)


class VerifyOTPAPIView(APIView):
    def post(self, request):
        email = request.data.get('email')
        otp = request.data.get('otp')

        try:
            worker = Worker.objects.get(user__email=email, otp=otp)
            user = User.objects.get(email=email)

            # Check if OTP has expired (valid for 10 minutes)
            time_difference = now() - worker.otp_created_at
            if time_difference.total_seconds() > 600:  # 10 minutes
                return Response({'message': 'OTP has expired. Request a new one.'}, status=status.HTTP_400_BAD_REQUEST)

            # Activate user
            user.is_active = True
            user.save()
            worker.otp = None  # Clear OTP after successful verification
            worker.save()

            return Response({'message': 'Account verified successfully. You can now log in.'}, status=status.HTTP_200_OK)
        except Worker.DoesNotExist:
            return Response({'message': 'Invalid OTP or email.'}, status=status.HTTP_400_BAD_REQUEST)

class ResendOTPAPIView(APIView):
    def post(self, request):
        email = request.data.get('email')

        try:
            worker = Worker.objects.get(user__email=email)

            # Generate new OTP and send it via email
            send_otp_email(worker.user)

            return Response({'message': 'A new OTP has been sent to your email.'}, status=status.HTTP_200_OK)
        except Worker.DoesNotExist:
            return Response({'message': 'Email not registered.'}, status=status.HTTP_400_BAD_REQUEST)


class RegisterAdminAPIView(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            User.objects.create_superuser(**serializer.validated_data)
            return Response({'message': 'Admin registered successfully'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class RequestRequesterAccessAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user

        if getattr(user, "role", "") == "Admin":
            return Response(
                {"message": "Admin does not need requester approval."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if user.is_requester:
            return Response(
                {"message": "You already have requester access."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if getattr(user, "requester_request_pending", False):
            return Response(
                {"message": "Your requester request is already pending admin approval."},
                status=status.HTTP_400_BAD_REQUEST
            )

        user.requester_request_pending = True
        user.save(update_fields=["requester_request_pending"])

        notify_admins(
            event_type="requester.review",
            title="New requester access request",
            body=f"{user.fullName} ({user.email}) requested requester access",
            payload=make_payload(
                type="requester.review",
                body_for_ui=f"{user.fullName} ({user.email}) requested requester access",
                requester_email=user.email,
                requester_user_id=user.id
            ),
            priority="high",
        )

        return Response(
            {"message": "Your requester access request was sent to the admin."},
            status=status.HTTP_200_OK
        )