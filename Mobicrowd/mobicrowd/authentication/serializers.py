from django.contrib.auth.models import Group, User

from rest_framework import serializers
from django.conf import settings
from allauth.utils import email_address_exists

from mobicrowd.authentication.email_sending import send_verification_email
from mobicrowd.models.Users import Worker, Requester


class RegistrationSerializer(serializers.ModelSerializer):
    user_type = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['email', 'password', 'fullName', 'mobile_phone', 'user_type']
        extra_kwargs = {
            'password': {'write_only': True}
        }
    def save(self):
        user_type = self.validated_data.pop('user_type')
        is_worker = user_type == 'Worker'
        is_requester = user_type == 'Requester'
        is_superuser = user_type == 'Admin'

        email = self.validated_data['email']
        if email_address_exists(email):
            raise serializers.ValidationError({'email': 'Email already exists'})
        user = User(
            email=email,
            fullName=self.validated_data['fullName'],
            mobile_phone=self.validated_data['mobile_phone'],
            is_worker=is_worker,
            is_requester=is_requester,
            is_superuser=is_superuser,
        )
        password = self.validated_data['password']
        user.set_password(password)
        user.save()

        if settings.ACCOUNT_EMAIL_VERIFICATION == "mandatory":
            user.is_active = False
            if is_worker:
                user.role = 'Worker'
                user.save()
                worker = Worker.objects.create(user=user)
                worker.save()
                send_verification_email(user)
            elif is_requester:
                user.role = 'Requester'
                user.save()
                requester = Requester.objects.create(user=user)
                requester.save()
            elif is_superuser:
                user.is_active = True
                user.role = 'Admin'
                user.save()

        return user
