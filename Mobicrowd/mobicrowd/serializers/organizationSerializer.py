from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import IntegrityError, transaction
from rest_framework import serializers

from mobicrowd.authentication.email_sending import (
    build_reset_password_link,
    send_forget_password_email,
    send_new_representative_welcome_email,
    send_old_representative_removed_email,
)
from mobicrowd.models.Users import (
    Organization,
    OrganizationLicenceKey,
    OrganizationMembership,
    Requester,
    User,
)


class OrganizationSerializer(serializers.ModelSerializer):
    representative_email = serializers.EmailField(write_only=True, required=False)
    representative_full_name = serializers.CharField(write_only=True, required=False, allow_blank=True, allow_null=True)
    representative_mobile_phone = serializers.CharField(write_only=True, required=False, allow_blank=True, allow_null=True)
    requester_approved = serializers.BooleanField(write_only=True, required=False, default=True)
    requester_location = serializers.CharField(write_only=True, required=False, allow_blank=True, allow_null=True)

    representative_user_email = serializers.EmailField(source='representative.user.email', read_only=True)
    representative_fullName = serializers.CharField(source='representative.user.fullName', read_only=True)

    available_requester_licences = serializers.IntegerField(read_only=True)
    available_contributor_licences = serializers.IntegerField(read_only=True)

    class Meta:
        model = Organization
        fields = [
            'id',
            'representative',
            'name',
            'phone_number',
            'postal_code',
            'email',
            'location',
            'industry_sector',
            'status',
            'activation_code',
            'licence_requester',
            'licence_contributor',
            "licence_duration",
            "licence_expiration_date",
            'description',
            'created_at',
            'activated_at',
            'available_requester_licences',
            'available_contributor_licences',
            'representative_email',
            'representative_full_name',
            'representative_mobile_phone',
            'requester_approved',
            'requester_location',
            'representative_user_email',
            'representative_fullName',
        ]
        read_only_fields = [
            'id',
            'representative',
            'status',
            'activation_code',
            'created_at',
            'activated_at',
            'available_requester_licences',
            'available_contributor_licences',
            'representative_user_email',
            'representative_fullName',
        ]

    def validate_name(self, value):
        """
        Reject organization names that already exist, ignoring letter case.

        The current instance is excluded during updates so changing only the
        capitalization of an organization's own name remains valid.
        """
        name = (value or "").strip()

        if not name:
            raise serializers.ValidationError(
                "Organization name is required."
            )

        queryset = Organization.objects.filter(name__iexact=name)

        if self.instance is not None:
            queryset = queryset.exclude(pk=self.instance.pk)

        if queryset.exists():
            raise serializers.ValidationError(
                "An organization with this name already exists."
            )

        return name

    @staticmethod
    def _raise_known_integrity_error(exc):
        """Translate the DB constraint error into a frontend-safe DRF error."""
        if "unique_organization_name_case_insensitive" in str(exc):
            raise serializers.ValidationError({
                "name": ["An organization with this name already exists."]
            }) from exc
        raise exc

    def validate(self, attrs):
        instance = getattr(self, 'instance', None)

        rep_email = attrs.get('representative_email')
        if rep_email is not None:
            rep_email = rep_email.strip().lower()
            attrs['representative_email'] = rep_email

        if instance is None:
            if not rep_email:
                raise serializers.ValidationError({
                    'representative_email': 'This field is required.'
                })

            user = User.objects.filter(email__iexact=rep_email).first()
            if user:
                if not user.is_requester:
                    raise serializers.ValidationError({
                        'representative_email': (
                            'This email already exists for a contributor/non-requester profile. '
                            'It cannot be used as an organization representative.'
                        )
                    })
                requester = Requester.objects.filter(user=user).first()
                if requester and Organization.objects.filter(representative=requester).exists():
                    raise serializers.ValidationError({
                        'representative_email': (
                            'This email is already assigned as representative of another organization.'
                        )
                    })
            else:
                if not attrs.get('representative_full_name'):
                    raise serializers.ValidationError({
                        'representative_full_name': 'This field is required when representative email does not exist.'
                    })
                if not attrs.get('representative_mobile_phone'):
                    raise serializers.ValidationError({
                        'representative_mobile_phone': 'This field is required when representative email does not exist.'
                    })
            return attrs

        if rep_email is None or rep_email == '':
            return attrs

        current_email = instance.representative.user.email.strip().lower()
        if rep_email == current_email:
            return attrs

        user = User.objects.filter(email__iexact=rep_email).first()
        if user:
            if not user.is_requester:
                raise serializers.ValidationError({
                    'representative_email': 'This email must belong to a requester account.'
                })
            if not user.is_active:
                raise serializers.ValidationError({
                    'representative_email': 'Representative must have an activated requester account.'
                })

            requester = Requester.objects.filter(user=user).first()
            if not requester:
                raise serializers.ValidationError({
                    'representative_email': 'Requester profile not found for this user.'
                })

            other_org = Organization.objects.filter(representative=requester).exclude(pk=instance.pk).first()
            if other_org:
                raise serializers.ValidationError({
                    'representative_email': 'This email is already assigned as representative of another organization.'
                })

        return attrs

    def create(self, validated_data):
        representative_email = validated_data.pop('representative_email')
        representative_full_name = validated_data.pop('representative_full_name', None)
        representative_mobile_phone = validated_data.pop('representative_mobile_phone', None)
        requester_approved = validated_data.pop('requester_approved', True)
        requester_location = validated_data.pop('requester_location', '') or ''

        existing_user = User.objects.filter(email__iexact=representative_email).first()
        is_new_user = existing_user is None

        try:
            org, generated_password = Organization.create_with_representative(
                org_name=validated_data['name'],
                representative_email=representative_email,
                representative_full_name=representative_full_name,
                representative_mobile_phone=representative_mobile_phone,
                licence_requester=validated_data.get('licence_requester', 0),
                licence_contributor=validated_data.get('licence_contributor', 0),
                requester_approved=requester_approved,
                requester_location=requester_location,
                organization_email=validated_data.get('email'),
            )
            self.context['generated_password'] = generated_password
        except DjangoValidationError as e:
            raise serializers.ValidationError(
                e.message_dict if hasattr(e, 'message_dict') else e.messages
            )
        except IntegrityError as e:
            self._raise_known_integrity_error(e)

        extra_fields = [
            'phone_number',
            'postal_code',
            'email',
            'location',
            'industry_sector',
            'licence_duration',
            'licence_expiration_date',
            'description',
        ]

        updated_fields = []
        for field in extra_fields:
            if field in validated_data:
                setattr(org, field, validated_data.get(field))
                updated_fields.append(field)

        if updated_fields:
            org.save(update_fields=updated_fields)

        if is_new_user:
            reset_password_link = build_reset_password_link(org.representative.user)
            send_forget_password_email(
                user=org.representative.user,
                reset_password_link=reset_password_link,
            )
            send_new_representative_welcome_email(
                user=org.representative.user,
                organization=org,
                temporary_password=None,
                show_account_activation=True,
            )
        else:
            send_new_representative_welcome_email(
                user=org.representative.user,
                organization=org,
                temporary_password=None,
                show_account_activation=False,
            )

        return org

    def update(self, instance, validated_data):
        representative_email = validated_data.pop('representative_email', None)
        representative_full_name = validated_data.pop('representative_full_name', None)
        representative_mobile_phone = validated_data.pop('representative_mobile_phone', None)
        requester_approved = validated_data.pop('requester_approved', True)
        requester_location = validated_data.pop('requester_location', None)

        old_representative = instance.representative
        old_representative_user = old_representative.user if old_representative and old_representative.user else None
        old_representative_email = (
            old_representative_user.email.strip().lower()
            if old_representative_user and old_representative_user.email
            else None
        )

        for field in [
            'name',
            'phone_number',
            'postal_code',
            'email',
            'location',
            'industry_sector',
            'licence_requester',
            'licence_contributor',
            'licence_duration',
            'licence_expiration_date',
            'description',
        ]:
            if field in validated_data:
                setattr(instance, field, validated_data[field])

        try:
            instance.save()
        except IntegrityError as e:
            self._raise_known_integrity_error(e)

        if representative_email:
            normalized_email = representative_email.strip().lower()
            try:
                instance.update_representative_by_email(
                    representative_email=normalized_email,
                    representative_full_name=representative_full_name,
                    representative_mobile_phone=representative_mobile_phone,
                    requester_approved=requester_approved,
                    requester_location=requester_location or instance.location or '',
                )
            except DjangoValidationError as e:
                raise serializers.ValidationError(e.message_dict if hasattr(e, 'message_dict') else e.messages)

            instance.refresh_from_db()

            if normalized_email != old_representative_email:
                send_new_representative_welcome_email(
                    user=instance.representative.user,
                    organization=instance,
                    temporary_password=None,
                    show_account_activation=False,
                )

                if old_representative_user:
                    still_represents_other_org = Organization.objects.filter(
                        representative=old_representative,
                    ).exclude(pk=instance.pk).exists()

                    became_public_requester = not still_represents_other_org
                    if became_public_requester:
                        old_representative_user.is_representative = False
                        old_representative_user.save(update_fields=['is_representative'])

                    send_old_representative_removed_email(
                        user=old_representative_user,
                        organization=instance,
                        became_public_requester=became_public_requester,
                    )

        return instance


class MyOrganizationSerializer(serializers.ModelSerializer):
    organisationName = serializers.CharField(source='name', read_only=True)
    uniqueCode = serializers.CharField(source='activation_code', read_only=True)
    myRole = serializers.SerializerMethodField()
    isRepresentative = serializers.SerializerMethodField()
    country = serializers.CharField(source='location', read_only=True)
    industrySector = serializers.CharField(source='industry_sector', read_only=True)

    class Meta:
        model = Organization
        fields = [
            'id',
            'organisationName',
            'uniqueCode',
            'myRole',
            'isRepresentative',
            'country',
            'industrySector',
            'description',
            'status',
        ]

    def get_myRole(self, obj):
        request = self.context.get('request')
        if not request or not request.user.is_authenticated:
            return None

        user = request.user
        if obj.representative and obj.representative.user_id == user.id:
            return 'Requester'

        membership = OrganizationMembership.objects.filter(
            organization=obj,
            user=user,
            status='active',
        ).first()

        if not membership:
            return None

        return 'Requester' if membership.is_requester else 'Contributor'

    def get_isRepresentative(self, obj):
        request = self.context.get('request')
        if not request or not request.user.is_authenticated:
            return False
        return bool(obj.representative and obj.representative.user_id == request.user.id)


class OrganizationDetailSerializer(serializers.ModelSerializer):
    organisationName = serializers.CharField(source='name', read_only=True)
    uniqueCode = serializers.CharField(source='activation_code', read_only=True)
    myRole = serializers.SerializerMethodField()
    isRepresentative = serializers.SerializerMethodField()
    country = serializers.CharField(source='location', read_only=True)
    industrySector = serializers.CharField(source='industry_sector', read_only=True)
    zipCode = serializers.CharField(source='postal_code', read_only=True)
    representative_user_email = serializers.EmailField(source='representative.user.email', read_only=True)

    class Meta:
        model = Organization
        fields = [
            'id',
            'organisationName',
            'uniqueCode',
            'myRole',
            'isRepresentative',
            'country',
            'industrySector',
            'zipCode',
            'representative_user_email',
        ]

    def get_isRepresentative(self, obj):
        request = self.context.get('request')
        if not request or not request.user.is_authenticated:
            return False
        return bool(obj.representative and obj.representative.user_id == request.user.id)

    def get_myRole(self, obj):
        request = self.context.get('request')
        if not request or not request.user.is_authenticated:
            return None

        user = request.user
        if obj.representative and obj.representative.user_id == user.id:
            return 'Requester'

        membership = OrganizationMembership.objects.filter(
            organization=obj,
            user=user,
            status='active',
        ).first()

        if not membership:
            return None

        return 'Requester' if membership.is_requester else 'Contributor'


class OrganizationMemberSerializer(serializers.ModelSerializer):
    name = serializers.CharField(source='user.fullName', read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)
    role = serializers.SerializerMethodField()
    joinedDate = serializers.DateTimeField(source='joined_at', read_only=True)

    class Meta:
        model = OrganizationMembership
        fields = [
            'id',
            'name',
            'email',
            'role',
            'status',
            'joinedDate',
        ]

    def get_role(self, obj):
        return 'Requester' if obj.is_requester else 'Contributor'


class OrganizationInviteSerializer(serializers.Serializer):
    emails = serializers.ListField(
        child=serializers.EmailField(),
        allow_empty=False,
    )
    role = serializers.ChoiceField(choices=['requester', 'contributor'])
    duration_hours = serializers.IntegerField(required=False, default=48, min_value=1)


class OrganizationMembershipCreateSerializer(serializers.Serializer):
    email = serializers.EmailField()
    role = serializers.ChoiceField(choices=['requester', 'contributor'])

    def validate(self, attrs):
        request = self.context['request']
        organization = self.context['organization']

        email = attrs['email'].strip().lower()
        role = attrs['role']
        attrs['email'] = email

        if organization.representative.user_id != request.user.id:
            raise serializers.ValidationError({
                'detail': 'Only the representative can create memberships.'
            })

        if organization.status != 'active':
            raise serializers.ValidationError({
                'detail': 'Memberships can only be created for an active organization.'
            })

        user = User.objects.filter(email__iexact=email).first()
        if not user:
            raise serializers.ValidationError({
                'email': 'No user found with this email.'
            })

        if not user.is_active:
            raise serializers.ValidationError({
                'email': 'User account must be active.'
            })

        if OrganizationMembership.objects.filter(organization=organization, user=user).exists():
            raise serializers.ValidationError({
                'email': 'This user already belongs to this organization.'
            })

        has_available_licence = OrganizationLicenceKey.objects.filter(
            organization=organization,
            role=role,
            status='idle',
        ).exists()

        if not has_available_licence:
            raise serializers.ValidationError({
                'role': f'No available {role} licence.'
            })

        attrs['user'] = user
        return attrs

    @transaction.atomic
    def create(self, validated_data):
        organization = self.context['organization']
        user = validated_data['user']
        role = validated_data['role']

        licence_key = (
            OrganizationLicenceKey.objects
            .select_for_update()
            .filter(
                organization=organization,
                role=role,
                status='idle',
            )
            .order_by('id')
            .first()
        )

        if not licence_key:
            raise serializers.ValidationError({
                'role': f'No available {role} licence.'
            })

        membership = OrganizationMembership(
            organization=organization,
            user=user,
            role=role,
            licence_key=licence_key,
            status='active',
        )
        membership.full_clean()
        membership.save()
        licence_key.activate_for_user(user)
        return membership
