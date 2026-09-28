import os
import secrets
from email.mime.image import MIMEImage
from urllib.parse import urlencode

from django.contrib.staticfiles import finders
from django.core.mail import EmailMessage, EmailMultiAlternatives
from django.db import transaction
from django.template.loader import render_to_string

from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.utils.html import strip_tags
from django.utils import timezone

from django.conf import settings
from mobicrowd.models.Users import Token, PasswordResetToken
def send_organization_invitation_cancelled_email(
    *,
    invited_email,
    organization,
    role,
    cancelled_by_name,
):
    subject = f"Invitation to join {organization.name} retracted"

    context = {
        "organization_name": organization.name,
        "organization_email": organization.email,
        "organization_location": organization.location,
        "cancelled_by_name": cancelled_by_name,
        "role_label": "Requester" if role == "requester" else "Contributor",
        "frontend_url": _frontend_base_url(),
    }

    html_body = render_to_string("organization_invitation_cancelled_email", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[invited_email],
        reply_to=["takleefinc@gmail.com"],
    )
    msg.attach_alternative(html_body, "text/html")
    msg.mixed_subtype = "related"

    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()

def _resolve_logo_path() -> str | None:
    """
    Return an absolute path to the logo file for CID embedding.
    Priority:
      1) settings.EMAIL_LOGO_PATH (recommended)
      2) staticfiles: 'emails/logo-02.png'
      3) fallback to '<BASE_DIR>/templates/logo-02.png'
    """
    # 1) explicit setting
    p = getattr(settings, "EMAIL_LOGO_PATH", None)
    if p and os.path.isfile(p):
        return p

    p = finders.find("emails/takleef.png")
    if p and os.path.isfile(p):
        return p

    # 3) fallback side-by-side with your template
    p = os.path.join(getattr(settings, "BASE_DIR", ""), "templates", "takleef.png")
    if os.path.isfile(p):
        return p

    return None


def _frontend_base_url() -> str:
    """Return the configured frontend base URL without a trailing slash."""
    return (
        getattr(settings, "FRONTEND_URL", None)
        or getattr(settings, "FRONTEND_URL", None)
        or getattr(settings, "BASE_URL", "")
    ).rstrip("/")

def send_verification_email(user):
    token_generator = PasswordResetTokenGenerator()

    with transaction.atomic():
        token, created = Token.objects.get_or_create(user=user)
        if not created:
            token.delete()
            token = Token.objects.create(user=user)
        token.token = token_generator.make_token(user)
        token.save()

    verification_url = f"{settings.BASE_URL}/api/account_confirm_email/{user.pk}/{token.token}"
    subject = "Verify your account"

    context = {
        "full_name": user.fullName,
        "verification_url": verification_url,
    }

    html_body = render_to_string("email_template", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
        reply_to=["takleefinc@gmail.com"],  # keep same reply-to info
    )
    msg.attach_alternative(html_body, "text/html")

    # Attach inline logo for <img src="cid:logo">
    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")  # must match cid:logo in template
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)
    # else: silently skip (or log warning)

    msg.send()

def send_approve_join_request_email(user, event):
    subject = f"You're Approved for the Event: {event.title}"
    context = {
        "full_name": user.fullName,
        "event_title": event.title,
        "event_deadline": event.deadline.strftime("%d/%m/%Y"),
        "max_photos_per_worker": event.max_photos_per_worker,
        # If your template includes reward, add it here too:
        # "reward": event.photo_reward,
    }

    html_body = render_to_string("approve_request", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
        reply_to=["takleefinc@gmail.com"],  # or keep old value if you prefer
    )
    msg.attach_alternative(html_body, "text/html")

    # Attach inline logo for <img src="cid:logo">
    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")  # must match cid:logo in template
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()



def send_new_representative_welcome_email(
    user,
    organization,
    temporary_password=None,
    show_account_activation=True,
):
    """Send the representative welcome email.

    Backward compatible with the old call signature while allowing the
    organization update flow to avoid sending an account-activation link to
    an already-active representative.
    """
    print("sending email")

    account_activation_url = None
    if show_account_activation:
        token_generator = PasswordResetTokenGenerator()

        with transaction.atomic():
            token, created = Token.objects.get_or_create(user=user)
            if not created:
                token.delete()
                token = Token.objects.create(user=user)

            token.token = token_generator.make_token(user)
            token.save()

        account_activation_url = f"{settings.BASE_URL}/api/account_confirm_email/{user.pk}/{token.token}"

    organization_activation_url = (
        f"{settings.BASE_URL}/api/organizations/{organization.pk}/activate"
        f"?code={organization.activation_code}"
    )
    show_organization_activation = organization.status != "active"

    subject = "Welcome to Takleef — Your representative account is ready"

    context = {
        "full_name": user.fullName,
        "email": user.email,
        "temporary_password": temporary_password,
        "organization_name": organization.name,
        "organization_activation_key": organization.activation_code,
        "account_activation_url": account_activation_url,
        "organization_activation_url": organization_activation_url,
        "show_account_activation": show_account_activation,
        "show_organization_activation": show_organization_activation,
    }

    html_body = render_to_string("new_representative_organization_welcome", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
        reply_to=["takleefinc@gmail.com"],
    )
    msg.attach_alternative(html_body, "text/html")
    msg.mixed_subtype = "related"

    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()

def send_reject_join_request_email(user, event):
    subject = f"Update on Your Join Request for: {event.title}"
    context = {
        "full_name": user.fullName,
        "event_title": event.title,
    }

    html_body = render_to_string("reject_request", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
        reply_to=["takleefinc@gmail.com"],  # or keep old value if you prefer
    )
    msg.attach_alternative(html_body, "text/html")

    # Attach inline logo for <img src="cid:logo">
    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")  # must match cid:logo in template
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()
def send_otp_email(user):

    user.worker_profile.generate_otp()
    subject = "Takleef - Your OTP for Account Verification"
    context = {
        "full_name": user.fullName,
        "otp_code": user.worker_profile.otp,  # Inject OTP into the email template
        # "confirm_url": confirm_url,
    }
    html_body = render_to_string("otp_email_template", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
    )
    msg.attach_alternative(html_body, "text/html")

    # Attach the inline logo for <img src="cid:logo">
    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")  # must match cid:logo
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)
    # else: silently skip if not found (or log a warning)

    msg.send()


def send_account_pending_approval(email,fullname):


    # Render email template with OTP
    subject = "Takleef — Your Requester Account Is Pending Admin Approval"
    context = {
        "full_name": fullname,
    }
    html_body = render_to_string("pending_requester_account", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[email],
    )
    msg.attach_alternative(html_body, "text/html")

    # Attach the inline logo for <img src="cid:logo">
    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")  # must match cid:logo
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)
    # else: silently skip if not found (or log a warning)

    msg.send()

def send_password_changed(user):
    print("hello")


    # Render email template with OTP
    subject = "Takleef - Your Password Was Changed"
    context = {
        "full_name": user.fullName,
    }
    html_body = render_to_string("password-changed", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
    )
    msg.attach_alternative(html_body, "text/html")

    # Attach the inline logo for <img src="cid:logo">
    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")  # must match cid:logo
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)
    # else: silently skip if not found (or log a warning)

    msg.send()


def send_forget_password_email(user, reset_password_link):
    subject = "Takleef - Reset Password"
    context = {
        "full_name": user.fullName,
        "email": user.email,
        "reset_password_link": reset_password_link,
    }

    html_body = render_to_string("forget_password", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
    )
    msg.attach_alternative(html_body, "text/html")

    # Attach inline logo for <img src="cid:logo">
    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")  # must match cid:logo in template
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()


def build_reset_password_link(user):
    """Create a secure one-use password reset link using selector + raw token.

    This preserves the safer takleef-lava reset design and does not expose the
    user id in the reset URL.
    """
    raw_token = secrets.token_urlsafe(48)

    with transaction.atomic():
        PasswordResetToken.objects.filter(
            user=user,
            used_at__isnull=True,
        ).update(used_at=timezone.now())

        rec = PasswordResetToken.objects.create(
            user=user,
            token_hash=PasswordResetToken.hash_token(raw_token),
        )

    return f"{_frontend_base_url()}/set-new-password/{rec.selector}/{raw_token}"


def send_organization_invitation_email(
    *,
    invited_email,
    organization,
    role,
    invitation,
    invited_by_name,
    invitation_url,
    custom_message='',
):
    subject = f"Invitation to join {organization.name}"

    context = {
        "organization_name": organization.name,
        "organization_email": organization.email,
        "organization_location": organization.location,
        "invited_by_name": invited_by_name,
        "role_label": "Requester" if role == "requester" else "Contributor",
        "expires_at": invitation.expires_at,
        "invitation_url": invitation_url,
        "custom_message": custom_message,
        "frontend_url": _frontend_base_url(),
    }

    html_body = render_to_string("organization_invitation_email", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[invited_email],
        reply_to=["takleefinc@gmail.com"],
    )
    msg.attach_alternative(html_body, "text/html")
    msg.mixed_subtype = "related"

    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()


def send_old_representative_removed_email(user, organization, became_public_requester: bool):
    subject = f"Representative role updated for {organization.name}"

    context = {
        "full_name": user.fullName,
        "organization_name": organization.name,
        "became_public_requester": became_public_requester,
    }

    html_body = render_to_string("old_representative_removed", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
        reply_to=["takleefinc@gmail.com"],
    )
    msg.attach_alternative(html_body, "text/html")
    msg.mixed_subtype = "related"

    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()


def send_requester_access_approved_email(user):
    subject = "Takleef — Your requester access has been approved"

    context = {
        "full_name": user.fullName,
    }

    html_body = render_to_string("requester_access_approved", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
        reply_to=["takleefinc@gmail.com"],
    )
    msg.attach_alternative(html_body, "text/html")
    msg.mixed_subtype = "related"

    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()


def send_requester_access_rejected_email(user):
    subject = "Takleef — Your requester access request was not approved"

    context = {
        "full_name": user.fullName,
    }

    html_body = render_to_string("requester_access_rejected", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
        reply_to=["takleefinc@gmail.com"],
    )
    msg.attach_alternative(html_body, "text/html")
    msg.mixed_subtype = "related"

    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()


def send_login_takeover_otp_email(user, otp_code, device_label="", platform="", expires_minutes=10):
    subject = "Takleef - Verify New Login"

    context = {
        "full_name": user.fullName,
        "otp_code": otp_code,
        "device_label": device_label,
        "platform": platform,
        "expires_minutes": expires_minutes,
    }

    html_body = render_to_string("login_takeover_otp", context)
    text_body = strip_tags(html_body)

    msg = EmailMultiAlternatives(
        subject=subject,
        body=text_body,
        from_email=getattr(settings, "EMAIL_HOST_USER", None) or getattr(settings, "DEFAULT_FROM_EMAIL", None),
        to=[user.email],
    )
    msg.attach_alternative(html_body, "text/html")

    logo_path = _resolve_logo_path()
    if logo_path:
        with open(logo_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<logo>")
            img.add_header("Content-Disposition", "inline", filename=os.path.basename(logo_path))
            msg.attach(img)

    msg.send()