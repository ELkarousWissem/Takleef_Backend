# serializers.py
from rest_framework import serializers

from mobicrowd.models.device_cam_specs import Device
from mobicrowd.models.notifications import DeviceToken, Notification

from mobicrowd.emoji_codec import decode_for_api

class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Device
        fields = '__all__'


class DeviceTokenSerializer(serializers.ModelSerializer):
    # If you’d prefer to not return full tokens, uncomment and use get_token_masked
    # token = serializers.SerializerMethodField()

    class Meta:
        model = DeviceToken
        fields = ("id", "token", "platform", "timezone", "is_active", "created_at", "last_seen")
        read_only_fields = ("id", "is_active", "created_at", "last_seen")
        extra_kwargs = {
            "token": {"write_only": True},  # don’t echo the raw token by default
        }
class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = ["id", "event_type", "title", "body", "payload", "priority", "created_at", "read_at"]

    def to_representation(self, instance):
        data = super().to_representation(instance)
        data["title"] = decode_for_api(data.get("title"))
        data["body"]  = decode_for_api(data.get("body"))
        return data