from rest_framework import serializers

from mobicrowd.models.Users import User, Worker, Requester


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = "__all__"

        extra_kwargs = {
            "profile_photo": {
                "required": False,
                "allow_null": True
            },
            "password": {
                "write_only": True
            }
        }

    def create(self, validated_data):
        user = User.objects.create_user(
            email=validated_data["email"],
            password=validated_data["password"],
            fullName=validated_data["fullName"],
            mobile_phone=validated_data["mobile_phone"],
            is_requester=validated_data.get("is_requester", False),
            is_worker=validated_data.get("is_worker", False),
            is_active=validated_data.get("is_active", True),
            role=validated_data["role"],
        )

        return user

class ProfilePhotoUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["profile_photo"]
        extra_kwargs = {
            "profile_photo": {
                "required": True,
                "allow_null": False
            }
        }

    def validate_profile_photo(self, uploaded_file):
        max_size = 5 * 1024 * 1024

        if uploaded_file.size > max_size:
            raise serializers.ValidationError(
                "The profile photo must not exceed 5 MB."
            )

        allowed_content_types = {
            "image/jpeg",
            "image/png",
            "image/webp",
        }

        content_type = getattr(uploaded_file, "content_type", "")

        if content_type not in allowed_content_types:
            raise serializers.ValidationError(
                "Only JPG, PNG and WebP images are supported."
            )

        return uploaded_file

class RequesterNamesSerializer(serializers.ModelSerializer):
    user = UserSerializer()

    class Meta:
        model = Requester
        fields = '__all__'
        extra_kwargs = {
                    'organization_name': {'required': False, 'allow_blank': True, 'allow_null': True}
                }
class RequesterSerializer(serializers.ModelSerializer):
    user = UserSerializer()

    class Meta:
        model = Requester
        fields = '__all__'

    def create(self, validated_data):
        user_data = validated_data.pop('user')

        user_data['is_requester'] = True
        user_data['is_worker'] = True
        user_data['role'] = 'Requester'

        user_serializer = UserSerializer(data=user_data)
        user_serializer.is_valid(raise_exception=True)
        user = user_serializer.save()

        requester = Requester.objects.create(user=user, **validated_data)

        Worker.objects.get_or_create(
            user=user,
            defaults={
                'location': validated_data.get('location', '') or '',
                'device_specs': {}
            }
        )

        return requester

class WorkerSerializer(serializers.ModelSerializer):
    user = UserSerializer()

    class Meta:
        model = Worker
        fields = '__all__'
    def create(self, validated_data):
        user_data = validated_data.pop('user')
        user_serializer = UserSerializer(data=user_data)
        if user_serializer.is_valid():
            user = user_serializer.save()
            worker = Worker.objects.create(user=user, **validated_data)
            return worker
        else:
            raise serializers.ValidationError(user_serializer.errors)