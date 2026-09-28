import hmac
import random
import secrets
import string
import hashlib
import uuid
from datetime import timedelta
from django.contrib.auth.hashers import make_password
from django.db import models
import secrets
import string
from django.core.exceptions import ValidationError
from django.db import transaction
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin, Group
from django.utils.timezone import now
from django.db.models.functions import Lower
import hashlib
from datetime import timedelta

from django.conf import settings
from django.db import models
from django.utils import timezone

DEFAULT_INVITATION_DURATION_HOURS = 168


def generate_unique_code(prefix="ORG", length=12):
    alphabet = string.ascii_uppercase + string.digits
    return f"{prefix}-" + "".join(secrets.choice(alphabet) for _ in range(length))

def generate_unique_licence_key(role):
    prefix = "REQ" if role == "requester" else "CTR"

    while True:
        value = generate_unique_code(prefix=prefix, length=12)
        if not OrganizationLicenceKey.objects.filter(key=value).exists():
            return value

class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **otherfields):
        if not email:
            raise ValueError('Users must have an email address')

        user = self.model(
            email=self.normalize_email(email),
            **otherfields
        )

        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **otherfields):
        otherfields.setdefault("role", "Admin")
        otherfields.setdefault("is_active", True)
        otherfields.setdefault("is_superuser", True)

        if otherfields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self.create_user(
            email,
            password=password,
            **otherfields
        )


class User(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    fullName = models.CharField(max_length=255)
    mobile_phone = models.CharField(max_length=20)
    profile_photo = models.ImageField(
        upload_to="users-profile-photos/",
        max_length=600,
        blank=True,
        null=True
    )
    is_worker = models.BooleanField(default=False)
    is_requester = models.BooleanField(default=False)
    is_representative = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    requester_request_pending = models.BooleanField(default=False)
    role = models.CharField(max_length=20, choices=(('Worker', 'Worker'), ('Requester', 'Requester'), ('Admin', 'Admin')))
    objects = UserManager()
    username = None
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['fullName', 'mobile_phone']


    def __str__(self):
        return self.email

    def get_full_name(self):
        return f"{self.fullName}"

    # def has_perm(self, perm, obj=None):
    #     return True

    def has_module_perms(self, app_label):
        return True

    # It appears to be intended to indicate whether a user has administrative privileges, but it is not clear exactly what those privileges are or how they differ from superuser privileges.
    @property
    def is_staff(self):
        return self.is_admin

    # super admin that have full access to all parts of the system, including the ability to manage other users and modify their permissions.
    @property
    def is_admin(self):
        return self.is_superuser

    class Meta:
        verbose_name_plural = "Custom Users"


class PasswordResetToken(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="password_reset_tokens"
    )

    # Public random identifier used in URL instead of user_id
    selector = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        db_index=True,
        editable=False
    )

    token_hash = models.CharField(max_length=64, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    used_at = models.DateTimeField(null=True, blank=True)

    @staticmethod
    def hash_token(raw: str) -> str:
        return hmac.new(
            settings.SECRET_KEY.encode("utf-8"),
            raw.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def is_expired(self) -> bool:
        timeout_seconds = getattr(settings, "PASSWORD_RESET_TIMEOUT", 3600)
        return timezone.now() > (self.created_at + timedelta(seconds=timeout_seconds))

    def mark_used(self):
        self.used_at = timezone.now()
        self.save(update_fields=["used_at"])
class Token(models.Model):
    token = models.TextField()
    user = models.OneToOneField(User, on_delete=models.CASCADE)



class Worker(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True, related_name='worker_profile')
    device_specs = models.JSONField()  # Stores device specifications as JSON
    location = models.CharField(max_length=255)  # Location field
    otp = models.CharField(max_length=6, blank=True, null=True)  # Stores OTP
    otp_created_at = models.DateTimeField(null=True, blank=True)  # Timestamp of OTP generation

    def generate_otp(self):
        """Generate and save a new OTP."""
        self.otp = str(random.randint(100000, 999999))  # 6-digit OTP
        self.otp_created_at = now()
        self.save()

    def delete(self, *args, **kwargs):
        # Delete the related User instance explicitly
        self.user.delete()
        super().delete(*args, **kwargs)


class Requester(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True, related_name='requester_profile')
    approved = models.BooleanField(default=False)
    organization_name = models.CharField(max_length=255, blank=True, null=True)  # Added field
    location = models.CharField(max_length=255)  # Location field


    def delete(self, *args, **kwargs):
        self.user.delete()
        # Delete associated reviews
        super().delete(*args, **kwargs)

import secrets

from django.core.exceptions import ValidationError
from django.db import models, transaction
from django.db.models.functions import Lower
from django.utils import timezone


class Organization(models.Model):
    representative = models.OneToOneField(
        Requester,
        on_delete=models.CASCADE,
        related_name='represented_organization'
    )

    name = models.CharField(max_length=255)

    phone_number = models.CharField(
        max_length=30,
        blank=True,
        null=True
    )

    postal_code = models.CharField(
        max_length=20,
        blank=True,
        null=True
    )

    email = models.EmailField(unique=True)

    location = models.CharField(
        max_length=255,
        blank=True,
        null=True
    )

    industry_sector = models.CharField(
        max_length=255,
        blank=True,
        null=True
    )

    status = models.CharField(
        max_length=20,
        choices=(
            ('pending', 'Pending'),
            ('active', 'Active'),
            ('suspended', 'Suspended'),
        ),
        default='pending'
    )

    activation_code = models.CharField(
        max_length=50,
        unique=True,
        blank=True
    )

    licence_requester = models.PositiveIntegerField(default=0)
    licence_contributor = models.PositiveIntegerField(default=0)

    description = models.TextField(
        blank=True,
        null=True
    )

    created_at = models.DateTimeField(auto_now_add=True)
    activated_at = models.DateTimeField(null=True, blank=True)

    licence_starting_date = models.DateField(
        null=True,
        blank=True
    )

    renewable = models.BooleanField(
        default=False
    )
    LICENCE_DURATION_CHOICES = [
        ("monthly", "Monthly"),
        ("yearly", "Yearly"),
    ]

    licence_duration = models.CharField(
        max_length=20,
        choices=LICENCE_DURATION_CHOICES,
        default="monthly"
    )

    licence_expiration_date = models.DateField(
        null=True,
        blank=True
    )

    members = models.ManyToManyField(
        User,
        through='OrganizationMembership',
        related_name='organizations',
        blank=True
    )

    # ---------------------------------------------------------
    # DATABASE CONSTRAINTS
    # ---------------------------------------------------------

    class Meta:
        constraints = [
            models.UniqueConstraint(
                Lower("name"),
                name="unique_organization_name_case_insensitive",
            ),
        ]

    def __str__(self):
        return self.name

    @property
    def is_contract_not_started(self):
        return bool(
            self.licence_starting_date
            and timezone.localdate() < self.licence_starting_date
        )

    @property
    def is_contract_expired(self):
        return bool(
            self.licence_expiration_date
            and timezone.localdate() >= self.licence_expiration_date
        )

    @property
    def contract_status(self):
        if self.is_contract_not_started:
            return "not_started"
        if self.is_contract_expired:
            return "expired"
        return "active"

    @property
    def contract_block_message(self):
        if self.is_contract_not_started:
            return "The organization contract has not started yet."
        if self.is_contract_expired:
            return (
                "The organization contract has expired. "
                "Please renew the contract to continue."
            )
        return None

    @property
    def available_requester_licences(self):
        return self.licence_keys.filter(
            role='requester',
            status='idle'
        ).count()

    @property
    def available_contributor_licences(self):
        return self.licence_keys.filter(
            role='contributor',
            status='idle'
        ).count()

    def activate(self, code=None):
        if code is not None and code != self.activation_code:
            raise ValidationError("Invalid activation code.")

        self.status = 'active'
        self.activated_at = timezone.now()
        self.save(update_fields=['status', 'activated_at'])

    def save(self, *args, **kwargs):
        is_new = self.pk is None

        old_licence_requester = 0
        old_licence_contributor = 0

        if not is_new:
            old = Organization.objects.get(pk=self.pk)

            old_licence_requester = old.licence_requester
            old_licence_contributor = old.licence_contributor

            # ------------------------------------------------------
            # Validate licence reductions BEFORE saving
            # ------------------------------------------------------

            requester_difference = (
                self.licence_requester - old_licence_requester
            )

            contributor_difference = (
                self.licence_contributor - old_licence_contributor
            )

            if requester_difference < 0:
                number_to_remove = abs(requester_difference)

                idle_count = OrganizationLicenceKey.objects.filter(
                    organization=self,
                    role='requester',
                    status='idle'
                ).count()

                if idle_count < number_to_remove:
                    raise ValidationError(
                        "Cannot reduce requester licences below "
                        "the number of licences already assigned "
                        "or pending activation."
                    )

            if contributor_difference < 0:
                number_to_remove = abs(contributor_difference)

                idle_count = OrganizationLicenceKey.objects.filter(
                    organization=self,
                    role='contributor',
                    status='idle'
                ).count()

                if idle_count < number_to_remove:
                    raise ValidationError(
                        "Cannot reduce contributor licences below "
                        "the number of licences already assigned "
                        "or pending activation."
                    )

        if not self.activation_code:
            self.activation_code = generate_unique_code(
                prefix="ACT",
                length=10
            )

        # ----------------------------------------------------------
        # Save organization
        # ----------------------------------------------------------

        super().save(*args, **kwargs)

        # ----------------------------------------------------------
        # Synchronize REQUESTER licence keys
        # ----------------------------------------------------------

        requester_difference = (
            self.licence_requester - old_licence_requester
        )

        if requester_difference > 0:

            new_keys = [
                OrganizationLicenceKey(
                    organization=self,
                    role='requester',
                    key=generate_unique_licence_key('requester')
                )
                for _ in range(requester_difference)
            ]

            OrganizationLicenceKey.objects.bulk_create(new_keys)

        elif requester_difference < 0:

            number_to_remove = abs(requester_difference)

            idle_keys = OrganizationLicenceKey.objects.filter(
                organization=self,
                role='requester',
                status='idle'
            ).order_by('-id')[:number_to_remove]

            OrganizationLicenceKey.objects.filter(
                id__in=[key.id for key in idle_keys]
            ).delete()

        # ----------------------------------------------------------
        # Synchronize CONTRIBUTOR licence keys
        # ----------------------------------------------------------

        contributor_difference = (
            self.licence_contributor - old_licence_contributor
        )

        if contributor_difference > 0:

            new_keys = [
                OrganizationLicenceKey(
                    organization=self,
                    role='contributor',
                    key=generate_unique_licence_key('contributor')
                )
                for _ in range(contributor_difference)
            ]

            OrganizationLicenceKey.objects.bulk_create(new_keys)

        elif contributor_difference < 0:

            number_to_remove = abs(contributor_difference)

            idle_keys = OrganizationLicenceKey.objects.filter(
                organization=self,
                role='contributor',
                status='idle'
            ).order_by('-id')[:number_to_remove]

            OrganizationLicenceKey.objects.filter(
                id__in=[key.id for key in idle_keys]
            ).delete()

    @classmethod
    @transaction.atomic
    def create_with_representative(
        cls,
        *,
        org_name,
        representative_email,
        representative_full_name=None,
        representative_mobile_phone=None,
        licence_requester=0,
        licence_contributor=0,
        requester_approved=True,
        requester_location='',
        organization_email=None,
    ):
        user = User.objects.filter(
            email__iexact=representative_email
        ).first()

        generated_password = None

        if user:
            requester = Requester.objects.filter(
                user=user
            ).first()

            if not requester:
                raise ValidationError(
                    "This email already exists but is not "
                    "registered as requester."
                )

            user_fields_to_update = []

            if not user.is_worker:
                user.is_worker = True
                user_fields_to_update.append('is_worker')

            if not user.is_requester:
                user.is_requester = True
                user_fields_to_update.append('is_requester')

            if not user.is_representative:
                user.is_representative = True
                user_fields_to_update.append('is_representative')

            if not user.role:
                user.role = 'Requester'
                user_fields_to_update.append('role')

            if user_fields_to_update:
                user.save(
                    update_fields=user_fields_to_update
                )

            Worker.objects.get_or_create(
                user=user,
                defaults={
                    "device_specs": {},
                    "location": (
                        requester.location
                        or requester_location
                        or ""
                    )
                }
            )

            requester.organization_name = org_name

            if requester_location:
                requester.location = requester_location

            requester.save(
                update_fields=[
                    'organization_name',
                    'location'
                ]
            )

        else:
            if (
                not representative_full_name
                or not representative_mobile_phone
            ):
                raise ValidationError(
                    "fullName and mobile_phone are required "
                    "to create a new representative account."
                )

            generated_password = secrets.token_urlsafe(12)

            user = User.objects.create_user(
                email=representative_email,
                password=generated_password,
                fullName=representative_full_name,
                mobile_phone=representative_mobile_phone,
                is_worker=True,
                is_requester=True,
                is_representative=True,
                is_active=False,
                role='Requester'
            )

            requester = Requester.objects.create(
                user=user,
                approved=requester_approved,
                organization_name=org_name,
                location=requester_location or ''
            )

            Worker.objects.get_or_create(
                user=user,
                defaults={
                    "device_specs": {},
                    "location": requester_location or ""
                }
            )

        org = cls.objects.create(
            representative=requester,
            name=org_name,
            email=organization_email,
            licence_requester=licence_requester,
            licence_contributor=licence_contributor
        )

        return org, generated_password

    @transaction.atomic
    def update_representative_by_email(
        self,
        *,
        representative_email,
        representative_full_name=None,
        representative_mobile_phone=None,
        requester_approved=True,
        requester_location=""
    ):
        email = (
            representative_email or ""
        ).strip().lower()

        if not email:
            raise ValidationError(
                "Representative email is required."
            )

        old_requester = self.representative
        old_user = (
            old_requester.user
            if old_requester
            else None
        )

        current_email = (
            old_user.email.strip().lower()
            if old_user and old_user.email
            else None
        )

        if current_email == email:
            return self.representative, None

        user = User.objects.filter(
            email__iexact=email
        ).first()

        generated_password = None

        if user:
            requester = Requester.objects.filter(
                user=user
            ).first()

            if not requester:
                raise ValidationError(
                    "This email already exists for a "
                    "contributor/non-requester profile. "
                    "It cannot be used as an organization "
                    "representative. Please use another email."
                )

            other_org = (
                Organization.objects
                .filter(representative=requester)
                .exclude(pk=self.pk)
                .first()
            )

            if other_org:
                raise ValidationError(
                    "This email is already assigned as "
                    "representative of another organization."
                )

            changed_user_fields = []

            if not user.is_worker:
                user.is_worker = True
                changed_user_fields.append("is_worker")

            if not user.is_requester:
                user.is_requester = True
                changed_user_fields.append("is_requester")

            if not user.is_representative:
                user.is_representative = True
                changed_user_fields.append("is_representative")

            if not user.role:
                user.role = "Requester"
                changed_user_fields.append("role")

            if changed_user_fields:
                user.save(
                    update_fields=changed_user_fields
                )

            changed_requester_fields = []

            if requester.organization_name != self.name:
                requester.organization_name = self.name
                changed_requester_fields.append(
                    "organization_name"
                )

            if (
                requester_location
                and requester.location != requester_location
            ):
                requester.location = requester_location
                changed_requester_fields.append(
                    "location"
                )

            if changed_requester_fields:
                requester.save(
                    update_fields=changed_requester_fields
                )

            Worker.objects.get_or_create(
                user=user,
                defaults={
                    "device_specs": {},
                    "location": (
                        requester.location
                        or requester_location
                        or ""
                    )
                }
            )

        else:
            if (
                not representative_full_name
                or not representative_mobile_phone
            ):
                raise ValidationError(
                    "fullName and mobile_phone are required "
                    "to create a new representative account."
                )

            generated_password = secrets.token_urlsafe(12)

            user = User.objects.create_user(
                email=email,
                password=generated_password,
                fullName=representative_full_name,
                mobile_phone=representative_mobile_phone,
                is_worker=True,
                is_requester=True,
                is_representative=True,
                is_active=False,
                role='Requester'
            )

            requester = Requester.objects.create(
                user=user,
                approved=requester_approved,
                organization_name=self.name,
                location=requester_location or ''
            )

            Worker.objects.get_or_create(
                user=user,
                defaults={
                    "device_specs": {},
                    "location": requester_location or ""
                }
            )

        self.representative = requester
        self.save(
            update_fields=["representative"]
        )

        if old_user and old_user.pk != user.pk:
            still_representative = (
                Organization.objects
                .filter(representative=old_requester)
                .exists()
            )

            if (
                not still_representative
                and old_user.is_representative
            ):
                old_user.is_representative = False
                old_user.save(
                    update_fields=["is_representative"]
                )

                old_requester.organization_name = ''
                old_requester.save(
                    update_fields=["organization_name"]
                )

        return requester, generated_password


class OrganizationLicenceKey(models.Model):
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='licence_keys'
    )

    role = models.CharField(
        max_length=20,
        choices=(
            ('requester', 'Requester'),
            ('contributor', 'Contributor'),
        )
    )

    key = models.CharField(max_length=50, unique=True, blank=True)

    status = models.CharField(
        max_length=30,
        choices=(
            ('idle', 'Idle'),
            ('pending_activation', 'Pending Activation'),
            ('active', 'Active'),
        ),
        default='idle'
    )

    invited_email = models.EmailField(null=True, blank=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    activated_at = models.DateTimeField(null=True, blank=True)

    assigned_user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='assigned_organization_licence_keys'
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.organization.name} - {self.role} - {self.key}"

    def save(self, *args, **kwargs):
        if not self.key:
            prefix = "REQ" if self.role == 'requester' else "CTR"
            self.key = generate_unique_code(prefix=prefix, length=12)
        super().save(*args, **kwargs)

    def mark_as_sent(self, email, duration_hours=DEFAULT_INVITATION_DURATION_HOURS):
        self.status = 'pending_activation'
        self.invited_email = email
        self.sent_at = timezone.now()
        self.expires_at = timezone.now() + timedelta(hours=duration_hours)
        self.activated_at = None
        self.assigned_user = None
        self.save(update_fields=[
            'status', 'invited_email', 'sent_at',
            'expires_at', 'activated_at', 'assigned_user'
        ])

    def activate_for_user(self, user):
        self.status = 'active'
        self.assigned_user = user
        self.activated_at = timezone.now()
        self.expires_at = None
        self.save(update_fields=[
            'status', 'assigned_user', 'activated_at', 'expires_at'
        ])

    def reset_if_expired(self):
        if self.status == 'pending_activation' and self.expires_at and timezone.now() > self.expires_at:
            self.status = 'idle'
            self.invited_email = None
            self.sent_at = None
            self.expires_at = None
            self.assigned_user = None
            self.activated_at = None
            self.save(update_fields=[
                'status', 'invited_email', 'sent_at',
                'expires_at', 'assigned_user', 'activated_at'
            ])

    @classmethod
    def reset_all_expired_keys(cls):
        expired = cls.objects.filter(
            status='pending_activation',
            expires_at__isnull=False,
            expires_at__lt=timezone.now()
        )
        for obj in expired:
            obj.reset_if_expired()


class OrganizationInvitation(models.Model):
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='invitations'
    )

    licence_key = models.ForeignKey(
        OrganizationLicenceKey,
        on_delete=models.PROTECT,
        related_name='invitations'
    )

    email = models.EmailField()

    role = models.CharField(
        max_length=20,
        choices=(
            ('requester', 'Requester'),
            ('contributor', 'Contributor'),
        )
    )

    status = models.CharField(
        max_length=20,
        choices=(
            ('sent', 'Sent'),
            ('accepted', 'Accepted'),
            ('declined', 'Declined'),
            ('expired', 'Expired'),
            ('cancelled', 'Cancelled'),
        ),
        default='sent'
    )

    duration_hours = models.PositiveIntegerField(default=DEFAULT_INVITATION_DURATION_HOURS)
    sent_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(blank=True, null=True)
    responded_at = models.DateTimeField(null=True, blank=True)

    token = models.CharField(max_length=128, unique=True, null=True, blank=True)

    def __str__(self):
        return f"{self.email} - {self.organization.name} - {self.role}"

    def clean(self):
        if self.licence_key.organization_id != self.organization_id:
            raise ValidationError("This licence key does not belong to this organization.")

        if self.licence_key.role != self.role:
            raise ValidationError("Invitation role must match the licence key role.")

        if self.licence_key.status != 'idle':
            raise ValidationError("This licence key is not available for invitation.")

    def generate_token(self):
        if not self.token:
            self.token = secrets.token_urlsafe(32)

    def is_expired(self):
        return bool(self.expires_at and timezone.now() > self.expires_at)

    def save(self, *args, **kwargs):
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(hours=self.duration_hours)

        if not self.token:
            self.generate_token()

        super().save(*args, **kwargs)

    @transaction.atomic
    def send_invitation(self):
        self.full_clean()
        self.save()
        self.licence_key.mark_as_sent(self.email, self.duration_hours)

    @transaction.atomic
    def accept(self, user):
        if user.email.strip().lower() != self.email.strip().lower():
            raise ValidationError("This invitation can only be accepted by the invited email.")

        if self.status == 'sent' and self.expires_at and timezone.now() > self.expires_at:
            self.status = 'expired'
            self.responded_at = timezone.now()
            self.save(update_fields=['status', 'responded_at'])
            self.licence_key.reset_if_expired()
            raise ValidationError("This invitation has expired.")

        if self.status != 'sent':
            raise ValidationError("Only a sent invitation can be accepted.")

        membership, created = OrganizationMembership.objects.get_or_create(
            organization=self.organization,
            user=user,
            defaults={
                'role': self.role,
                'licence_key': self.licence_key,
                'status': 'active',
            }
        )

        if not created:
            raise ValidationError("This user already belongs to this organization.")

        self.licence_key.activate_for_user(user)

        self.status = 'accepted'
        self.responded_at = timezone.now()
        self.save(update_fields=['status', 'responded_at'])

        return membership

    @transaction.atomic
    def accept_without_login(self):
        if self.status == 'sent' and self.expires_at and timezone.now() > self.expires_at:
            self.status = 'expired'
            self.responded_at = timezone.now()
            self.save(update_fields=['status', 'responded_at'])
            self.licence_key.reset_if_expired()
            raise ValidationError("This invitation has expired.")

        if self.status != 'sent':
            raise ValidationError("Only a sent invitation can be accepted.")

        user = User.objects.filter(email__iexact=self.email).first()
        if not user:
            raise ValidationError("No user found for this invitation email.")

        if self.role == 'requester' and not user.is_active:
            raise ValidationError("Requester account must be activated before accepting the invitation.")

        membership, created = OrganizationMembership.objects.get_or_create(
            organization=self.organization,
            user=user,
            defaults={
                'role': self.role,
                'licence_key': self.licence_key,
                'status': 'active',
            }
        )

        if not created:
            raise ValidationError("This user already belongs to this organization.")

        self.licence_key.activate_for_user(user)

        self.status = 'accepted'
        self.responded_at = timezone.now()
        self.save(update_fields=['status', 'responded_at'])

        return membership

    @transaction.atomic
    def decline(self):
        if self.status != 'sent':
            raise ValidationError("Only a sent invitation can be declined.")

        self.status = 'declined'
        self.responded_at = timezone.now()
        self.save(update_fields=['status', 'responded_at'])

        self.licence_key.status = 'idle'
        self.licence_key.invited_email = None
        self.licence_key.sent_at = None
        self.licence_key.expires_at = None
        self.licence_key.assigned_user = None
        self.licence_key.activated_at = None
        self.licence_key.save(update_fields=[
            'status', 'invited_email', 'sent_at',
            'expires_at', 'assigned_user', 'activated_at'
        ])

    @transaction.atomic
    def decline_without_login(self):
        if self.status != 'sent':
            raise ValidationError("Only a sent invitation can be declined.")

        if self.expires_at and timezone.now() > self.expires_at:
            self.status = 'expired'
            self.responded_at = timezone.now()
            self.save(update_fields=['status', 'responded_at'])
            self.licence_key.reset_if_expired()
            raise ValidationError("This invitation has expired.")

        self.status = 'declined'
        self.responded_at = timezone.now()
        self.save(update_fields=['status', 'responded_at'])

        self.licence_key.status = 'idle'
        self.licence_key.invited_email = None
        self.licence_key.sent_at = None
        self.licence_key.expires_at = None
        self.licence_key.assigned_user = None
        self.licence_key.activated_at = None
        self.licence_key.save(update_fields=[
            'status', 'invited_email', 'sent_at',
            'expires_at', 'assigned_user', 'activated_at'
        ])

    @transaction.atomic
    def cancel(self):
        if self.status != 'sent':
            raise ValidationError("Only a sent invitation can be cancelled.")

        self.status = 'cancelled'
        self.responded_at = timezone.now()
        self.save(update_fields=['status', 'responded_at'])

        self.licence_key.status = 'idle'
        self.licence_key.invited_email = None
        self.licence_key.sent_at = None
        self.licence_key.expires_at = None
        self.licence_key.assigned_user = None
        self.licence_key.activated_at = None
        self.licence_key.save(update_fields=[
            'status', 'invited_email', 'sent_at',
            'expires_at', 'assigned_user', 'activated_at'
        ])


class OrganizationMembership(models.Model):
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        related_name='memberships'
    )

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='organization_memberships'
    )

    # org-specific role
    role = models.CharField(
        max_length=20,
        choices=(
            ('requester', 'Requester'),
            ('contributor', 'Contributor'),
        )
    )

    licence_key = models.OneToOneField(
        OrganizationLicenceKey,
        on_delete=models.PROTECT,
        related_name='membership'
    )

    status = models.CharField(
        max_length=20,
        choices=(
            ('active', 'Active'),
            ('suspended', 'Suspended'),
            ('removed', 'Removed'),
        ),
        default='active'
    )

    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('organization', 'user')

    def __str__(self):
        return f"{self.organization.name} - {self.user.email} - {self.role}"

    def clean(self):
        if self.licence_key.organization_id != self.organization_id:
            raise ValidationError("Licence key does not belong to this organization.")

        if self.licence_key.role != self.role:
            raise ValidationError("Membership role must match the licence key role.")

    @property
    def is_requester(self) -> bool:
            return self.role == 'requester'

    @property
    def is_contributor(self) -> bool:
        return self.role in ['contributor', 'requester']

    @property
    def is_contributor_only(self) -> bool:
            return self.role == 'contributor'