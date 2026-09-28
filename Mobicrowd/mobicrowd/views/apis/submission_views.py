import io
import zipfile
from pathlib import Path
import cv2
import requests
from botocore.exceptions import ClientError
from django.http import StreamingHttpResponse, HttpResponse
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.views import APIView
from ultralytics import YOLO

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from .notify_helpers import make_payload, notify_admins
from mobicrowd.views.apis.workshop_apis.textual_safety_check.helpers import check_text_safety
from ...models.Users import Worker, Requester, OrganizationMembership
from ...models.submisson import Submission, Photo, Event, EventWorker, SubmissionReport, \
    Video, PHOTO, VIDEO
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from ...qdrant_service import ensure_text_embeddings_indexes
from ...serializers.submissionSerializers import SubmissionSerializer, SubmissionSerializerForWorker, \
    WorkerEventInfoSerializer, SubmissionReportSerializer, SubmissionStatusSerializer
from ...tasks import logger
from ...video_tasks import blur_video_faces_sparse
from mobicrowd.views.apis.submission_flow.security import text_digest
from mobicrowd.views.apis.submission_flow.text_submission import (
    TextPipelineTechnicalError,
    validate_and_finalize_text_submission,
)

BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed
CACHE_DIR = BASE_DIR / "tmp" / "blurred-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
YOLO_MODEL_PATH = "mobicrowd/models/UNIQUE/yolov12n-face.pt"  # Update this path

def check_event_access(user, event):
    if getattr(user, "role", None) == "Admin":
        return True

    if event.organization_id is None:
        return True

    if OrganizationMembership.objects.filter(
        user=user,
        organization=event.organization,
        status='active',
    ).exists():
        return True

    requester = Requester.objects.filter(user=user).first()
    if requester and event.organization and event.organization.representative == requester:
        return True

    return False


def get_event_membership(user, event):
    if event.organization_id is None:
        return None

    return OrganizationMembership.objects.filter(
        user=user,
        organization=event.organization,
        status='active',
    ).first()


def can_access_submission_file(user, submission):
    event = submission.event

    if getattr(user, "role", None) == "Admin" or getattr(user, "is_superuser", False):
        return True

    if submission.worker and submission.worker.user_id == user.id:
        return True

    if event.organization_id is None:
        return bool(event.requester and event.requester.user_id == user.id)

    membership = event.organization_membership
    if (
        membership
        and membership.user_id == user.id
        and membership.role == "requester"
        and membership.status == "active"
    ):
        return True

    requester = Requester.objects.filter(user=user).first()
    return bool(
        requester
        and event.organization
        and event.organization.representative_id == requester.user_id
    )


def can_update_submission_status(user, submission, new_status):
    # Approval is pipeline-managed and can never be produced by this generic endpoint.
    if new_status == Submission.APPROVED:
        return False
    if new_status != Submission.REFUSED:
        return False
    if getattr(user, "role", None) == "Admin" or getattr(user, "is_superuser", False):
        return True
    return bool(submission.worker and submission.worker.user_id == user.id)


def load_yolo_model(path=YOLO_MODEL_PATH):
    return YOLO(path)

def encode_cv2_image_to_base64(cv2_img):
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_img}"
def blur_faces(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face, (171, 131), 60)
        image[y1:y2, x1:x2] = blurred_face
    return image
def blur_faces_pipeline_from_base64(base64_image_str, filename_prefix="blurred"):
    import numpy as np
    from datetime import datetime

    # Decode base64 image string
    base64_data = base64_image_str.split(',')[1] if ',' in base64_image_str else base64_image_str
    image_data = base64.b64decode(base64_data)
    nparr = np.frombuffer(image_data, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError("Failed to decode image from base64 string.")

    model = load_yolo_model()
    results = model.predict(image_bgr)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

    final_image = blur_faces(image_bgr, boxes) if len(boxes) > 0 else image_bgr
    base64_img = encode_cv2_image_to_base64(final_image)

    # Generate a fake path string (optional)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"{filename_prefix}_{timestamp}.jpg"
    output_path = str(CACHE_DIR / output_filename)

    return {
        "blurred": len(boxes) > 0,
        "base64": base64_img,
        "image_bytes": base64.b64decode(base64_img.split(",")[1]),
        "saved_path": output_path  # optional if you later want to log
    }


class SubmissionCountView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, event_id: int, worker_id: int):
        try:
            event = Event.objects.get(pk=event_id)
        except Event.DoesNotExist:
            return Response(
                {"error": "Event not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        if not check_event_access(request.user, event):
            return Response(
                {"error": "Not allowed"},
                status=status.HTTP_403_FORBIDDEN,
            )

        count_ = Submission.objects.filter(
            event_id=event_id,
            worker__user_id=worker_id,
        ).count()
        return Response(count_)

class AllSubmissionCountView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, event_id):
        try:
                    event = Event.objects.get(pk=event_id)
                    if not check_event_access(request.user, event):
                        return Response({"error": "Not allowed"}, status=403)
        except Event.DoesNotExist:
                    return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
        # Get the connected worker
        # Count submissions for the worker and event
        submission_count = Submission.objects.filter(event_id=event_id).count()
        # Serialize the count
        return Response(submission_count)

class AcceptedSubmissionCountView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, event_id: int, worker_id: int):
        try:
                    event = Event.objects.get(pk=event_id)
                    if not check_event_access(request.user, event):
                        return Response({"error": "Not allowed"}, status=403)
        except Event.DoesNotExist:
                    return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
        count_ = Submission.objects.filter(
            event_id=event_id,
            worker__user_id=worker_id,   # <-- key change
            status=Submission.APPROVED,
        ).count()
        return Response(count_)

class RefusedSubmissionCountView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, event_id: int, worker_id: int):
        try:
                    event = Event.objects.get(pk=event_id)
                    if not check_event_access(request.user, event):
                        return Response({"error": "Not allowed"}, status=403)
        except Event.DoesNotExist:
                    return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)

        count_ = Submission.objects.filter(
            event_id=event_id,
            worker__user_id=worker_id,   # <-- key change
            status=Submission.REFUSED,
        ).count()
        return Response(count_)
class ALLAcceptedSubmissionCountView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, event_id):
        try:
            event = Event.objects.get(pk=event_id)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)

        # admin => accès direct
        if getattr(request.user, "role", None) != "Admin":
            if not check_event_access(request.user, event):
                return Response({"error": "Not allowed"}, status=403)

        submission_count = Submission.objects.filter(
            event_id=event_id,
            status=Submission.APPROVED
        ).count()

        return Response(submission_count)
class ALLRejectedSubmissionCountView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, event_id):
        # Get the connected worker
        # Count submissions for the worker and event
        try:
                    event = Event.objects.get(pk=event_id)
                    if not check_event_access(request.user, event):
                        return Response({"error": "Not allowed"}, status=403)
        except Event.DoesNotExist:
                    return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
        submission_count = Submission.objects.filter(event_id=event_id,status=Submission.REFUSED).count()
        # Serialize the count
        return Response(submission_count)


def _normalize_submission_status(value, default=None):
    """
    Accept frontend naming variants and return the canonical Submission status.
    """
    if value is None or str(value).strip() == "":
        return default

    raw = str(value).strip().lower()
    refused = getattr(Submission, "REFUSED", "refused")
    approved = getattr(Submission, "APPROVED", "approved")
    pending = getattr(Submission, "PENDING", "pending")

    mapping = {
        "approved": approved,
        "approve": approved,
        "accepted": approved,
        "accept": approved,
        "refused": refused,
        "rejected": refused,
        "reject": refused,
        "declined": refused,
        "decline": refused,
        "failed": refused,
        "blocked": refused,
        "pending": pending,
        "in_processing": "in_processing",
        "processing": "in_processing",
    }
    return mapping.get(raw, default)


def _extract_submission_message(data, fallback=""):
    """
    One place for validation/rejection message extraction.
    The frontend may send message/error_message/validation_message/reason/detail.
    """
    for key in (
        "message",
        "error_message",
        "validation_message",
        "rejection_message",
        "reason",
        "detail",
    ):
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(fallback or "").strip()

class SubmissionCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        worker_id = request.data.get("worker_id")
        event_id = request.data.get("event_id")
        location = request.data.get("location")
        media_type = str(request.data.get("media_type") or "").strip().lower()
        media_path = request.data.get("media_path")
        metadata = request.data.get("metadata") if isinstance(request.data.get("metadata"), dict) else {}

        # Server-managed fields are intentionally ignored. Older clients may still
        # send them, but they no longer carry any authority.
        submission_text = request.data.get("text")
        if submission_text is None and media_type == "text":
            submission_text = request.data.get("worker_message")  # compatibility only
        if not submission_text and isinstance(metadata, dict):
            submission_text = ((metadata.get("text") or {}).get("content") or "")
        submission_text = str(submission_text or "").strip()

        try:
            worker = Worker.objects.get(pk=worker_id)
            event = Event.objects.get(pk=event_id)
        except Worker.DoesNotExist:
            return Response({"error": "Worker not found"}, status=status.HTTP_404_NOT_FOUND)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)

        if worker.user_id != request.user.id:
            return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)
        if not check_event_access(request.user, event):
            return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

        membership = get_event_membership(request.user, event)
        if event.organization_id is not None:
            if not membership:
                return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)
            if membership.role != "contributor":
                return Response(
                    {"error": "Only contributors can submit to this event"},
                    status=status.HTTP_403_FORBIDDEN,
                )

        # Frontend checks are not security checks. Enforce join approval and event time here.
        if not EventWorker.objects.filter(
            event=event,
            worker=worker,
            status=EventWorker.APPROVED,
        ).exists():
            return Response(
                {"code": "event_membership_not_approved", "error": "Contributor is not approved for this event."},
                status=status.HTTP_403_FORBIDDEN,
            )

        now = timezone.now()
        if event.startdate and now < event.startdate:
            return Response({"code": "event_not_started"}, status=status.HTTP_409_CONFLICT)
        if event.deadline and now > event.deadline:
            return Response({"code": "event_ended"}, status=status.HTTP_409_CONFLICT)

        allowed_media = event.media_types if isinstance(event.media_types, list) else []
        if media_type not in allowed_media:
            return Response(
                {"error": f"Media type '{media_type}' not allowed for this event."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # TEXT: create PENDING/VALIDATING and let the backend own the complete decision.
        if media_type == "text":
            if not submission_text:
                return Response(
                    {"error": "Text submission is required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Permanent Qdrant schema guard.
            #
            # text_embeddings is filtered by event_id during redundancy checks.
            # Qdrant Cloud requires an INTEGER payload index for that field.
            # The helper is idempotent and cached per Django process, so this
            # does not recreate the collection or delete existing vectors.
            try:
                ensure_text_embeddings_indexes()
            except Exception:
                logger.exception(
                    "Qdrant text schema initialization failed event_id=%s",
                    event.id,
                )
                return Response(
                    {
                        "ok": False,
                        "saved": False,
                        "code": "text_validation_unavailable",
                        "message": "Text validation is temporarily unavailable. Please retry.",
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            submission = Submission.objects.create(
                worker=worker,
                event=event,
                location=location,
                status=Submission.PENDING,
                flow_stage=Submission.FLOW_VALIDATING,
                flow_artifact_digest=text_digest(submission_text),
                text=submission_text,
                message=None,
            )
            submission_id = submission.id
            try:
                decision = validate_and_finalize_text_submission(submission_id)
            except TextPipelineTechnicalError as exc:
                # Technical failures are not contributor rejections and must not become paid/approved rows.
                submission.delete()
                logger.exception("Text validation technical failure submission_id=%s", submission_id)
                return Response(
                    {
                        "ok": False,
                        "saved": False,
                        "code": "text_validation_unavailable",
                        "message": "Text validation is temporarily unavailable. Please retry.",
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            except Exception:
                submission.delete()
                logger.exception("Unexpected text validation failure submission_id=%s", submission_id)
                return Response(
                    {
                        "ok": False,
                        "saved": False,
                        "code": "text_validation_unavailable",
                        "message": "Text validation is temporarily unavailable. Please retry.",
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            submission.refresh_from_db()
            return Response(
                {
                    "ok": submission.status == Submission.APPROVED,
                    "saved": True,
                    "submission_id": submission.id,
                    "id": submission.id,
                    "status": submission.status,
                    "flow_stage": submission.flow_stage,
                    "media_type": "text",
                    "text": submission.text,
                    "worker_message": submission.worker_message,
                    "submission_message": submission.message,
                    "message": submission.message,
                    "decision": decision.get("decision"),
                    "validation_stage": decision.get("stage"),
                },
                status=status.HTTP_201_CREATED,
            )

        # PHOTO / VIDEO: creation never approves. It only creates a workflow shell.
        # if not media_path:
        #     return Response(
        #         {"error": "media_path is required for photo/video submission creation."},
        #         status=status.HTTP_400_BAD_REQUEST,
        #     )

        duration = request.data.get("duration")
        if duration is None and isinstance(metadata, dict):
            duration = (metadata.get("media") or {}).get("durationSec")
        if media_type == "video":
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                return Response({"error": "Valid video duration is required."}, status=status.HTTP_400_BAD_REQUEST)
            if duration < 0:
                return Response({"error": "Video duration cannot be negative."}, status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            submission = Submission.objects.create(
                worker=worker,
                event=event,
                location=location,
                status=Submission.PENDING,
                flow_stage=Submission.FLOW_CREATED,
                message=None,
            )
            if media_type == "photo":
                photo = Photo.objects.create(
                    image=media_path,
                    original_metadata=metadata,
                    caption="",
                )
                submission.photo = photo
            else:
                video = Video.objects.create(
                    video=media_path,
                    metadata=metadata,
                    duration=duration,
                )
                submission.video = video
            submission.save(update_fields=["photo"] if media_type == "photo" else ["video"])

        return Response(
            {
                "submission_id": submission.id,
                "id": submission.id,
                "status": submission.status,
                "flow_stage": submission.flow_stage,
                "message": submission.message,
            },
            status=status.HTTP_201_CREATED,
        )

class SubmissionStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, submission_id, *args, **kwargs):
        try:
            submission = (
                Submission.objects.select_related("worker", "worker__user", "event")
                .get(id=submission_id)
            )
        except Submission.DoesNotExist:
            return Response({"error": "Submission not found"}, status=status.HTTP_404_NOT_FOUND)

        if not can_access_submission_file(request.user, submission):
            return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

        return Response(
            {
                "submission_id": submission.id,
                "status": submission.status,
                "flow_stage": submission.flow_stage,
                "message": submission.message,
                "proceed_or_not": getattr(submission, "proceed_or_not", None),
            }
        )


class InProcessingSubmissionsByEventView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, event_id, *args, **kwargs):
        try:
            event = Event.objects.get(pk=event_id)
            if not check_event_access(request.user, event):
                return Response({"error": "Not allowed"}, status=403)
            submissions = Submission.objects.filter(event=event, status='in_processing')
            serializer = SubmissionSerializer(submissions, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)


class CompressedImageProxyView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        image_url = request.query_params.get("url")
        if not image_url:
            return Response({'error': 'Missing image URL'}, status=400)

        try:
            response = requests.get(image_url, stream=True)
            image = Image.open(response.raw).convert('RGB')
            image.thumbnail((320, 320))

            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=50)
            buffer.seek(0)
            return HttpResponse(buffer, content_type='image/jpeg')

        except Exception as e:
            return Response({'error': str(e)}, status=500)

class InProcessingSubmissionsByWorkerView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, event_id,worker_id, *args, **kwargs):
        try:
            event = Event.objects.get(pk=event_id)
            if not check_event_access(request.user, event):
                return Response({"error": "Not allowed"}, status=403)
            worker = Worker.objects.get(pk=worker_id)
            submissions = Submission.objects.filter(event=event,worker=worker, status='in_processing')
            serializer = SubmissionSerializerForWorker(submissions, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
# View for Approved Submissions by Worker
class ApprovedSubmissionsByWorkerView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, event_id, worker_id, *args, **kwargs):
        try:
            event = Event.objects.get(pk=event_id)
            if not check_event_access(request.user, event):
                return Response({"error": "Not allowed"}, status=403)
            worker = Worker.objects.get(pk=worker_id)
            submissions = Submission.objects.filter(event=event, worker=worker, status='approved')
            serializer = SubmissionSerializer(submissions, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)
        except Worker.DoesNotExist:
            return Response({"error": "Worker not found"}, status=status.HTTP_404_NOT_FOUND)


# View for Approved Submissions by Event
class ApprovedSubmissionsByEventView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, event_id, *args, **kwargs):
        try:
            Event.objects.get(pk=event_id)
            if not check_event_access(request.user, Event.objects.get(pk=event_id)):
                            return Response({"error": "Not allowed"}, status=403)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)

        submissions = Submission.objects.filter(event_id=event_id, status='approved')
        serializer = SubmissionSerializer(submissions, many=True)

        return Response(
            {
                "submissions": serializer.data,
                "count": submissions.count()
            },
            status=status.HTTP_200_OK
        )
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed

def save_to_cache(image_bytes):
    cache_dir = BASE_DIR / "tmp" / "caption"
    cache_dir.mkdir(parents=True, exist_ok=True)

    timestamp = str(int(time.time() * 1000))
    file_path = cache_dir / f"{timestamp}.jpeg"

    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return str(file_path)


import boto3
class DownloadImageUrlView(APIView):
    permission_classes = [IsAuthenticated]  # Optional: if download requires login

    def get(self, request, photo_id):
        try:
            photo = Photo.objects.get(id=photo_id)
            submission = (
                Submission.objects
                .select_related(
                    "worker",
                    "worker__user",
                    "event",
                    "event__requester",
                    "event__requester__user",
                    "event__organization",
                    "event__organization__representative",
                    "event__organization__representative__user",
                    "event__organization_membership",
                    "event__organization_membership__user",
                )
                .filter(photo=photo)
                .first()
            )

            if not submission:
                return Response({"error": "Submission not found for this photo"}, status=404)

            if not can_access_submission_file(request.user, submission):
                return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

            key = f"multimedia/{str(photo.image)}"

            s3 = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME
            )

            url = s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': settings.AWS_STORAGE_BUCKET_NAME,
                    'Key' : key,
                    'ResponseContentDisposition': f'attachment; filename="{photo.image.name.split("/")[-1]}"'
                },
                ExpiresIn=60
            )
            print("key", photo.image)

            return Response({'url': url})

        except Photo.DoesNotExist:
            return Response({'error': 'Photo not found'}, status=404)
        except ClientError as e:
            return Response({'error': str(e)}, status=500)
class DownloadSubmissionReportUrlView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, report_id):
        try:
            report = SubmissionReport.objects.select_related(
                "submission",
                "submission__worker",
                "submission__worker__user",
                "submission__event",
            ).get(id=report_id)

            submission = getattr(report, "submission", None)
            if submission is None:
                return Response(
                    {"error": "Submission not found for this report"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            if not can_access_submission_file(request.user, submission):
                return Response(
                    {"error": "Not allowed"},
                    status=status.HTTP_403_FORBIDDEN,
                )

            if not report.file:
                return Response(
                    {"error": "No file attached to this report"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # report.file.name example from your table: "reports/w6/e1/....jpg" or ".mp4"
            stored_path = report.file.name  # same as str(report.file)

            # If you use django-storages with AWS_LOCATION="multimedia", use it automatically.
            base_prefix = getattr(settings, "AWS_LOCATION", "")
            base_prefix = (base_prefix or "").strip("/")

            if base_prefix:
                key = f"{base_prefix}/{stored_path.lstrip('/')}"
            else:
                key = stored_path.lstrip('/')

            filename = stored_path.split("/")[-1]

            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME,
            )

            url = s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": settings.AWS_STORAGE_BUCKET_NAME,
                    "Key": key,
                    "ResponseContentDisposition": f'attachment; filename="{filename}"',
                },
                ExpiresIn=60,
            )

            return Response({"url": url})

        except SubmissionReport.DoesNotExist:
            return Response({"error": "SubmissionReport not found"}, status=status.HTTP_404_NOT_FOUND)
        except ClientError as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DownloadEventZipView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, event_id):
        event = get_object_or_404(Event, pk=event_id)

        if not check_event_access(request.user, event):
            return Response({"error": "Not allowed"}, status=403)
        # --- fetch all APPROVED submissions for the event ---
        photo_qs = Photo.objects.filter(
            submission__event_id=event_id,
            submission__status=Submission.APPROVED,
        ).select_related("submission")

        if not photo_qs.exists():
            return StreamingHttpResponse(
                iter([b'No approved photos.']), status=404,
                content_type='text/plain'
            )

        # --- create an in-memory zip stream ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            s3 = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME
            )

            for photo in photo_qs:
                key = f"multimedia/{str(photo.image)}"
             # eg. multimedia/submissions/...
                filename = key.split('/')[-1]            # keep original name

                obj = s3.get_object(
                    Bucket=settings.AWS_STORAGE_BUCKET_NAME,
                    Key=key
                )
                zf.writestr(filename, obj['Body'].read())

        zip_buffer.seek(0)

        response = StreamingHttpResponse(
            streaming_content=iter(lambda: zip_buffer.read(8192), b''),
            content_type='application/zip'
        )
        response['Content-Disposition'] = (
            f'attachment; filename="event_{event_id}_photos.zip"'
        )
        return response

class WorkerEventInfoAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, event_id, *args, **kwargs):
        try:
            # Retrieve the event
            event = Event.objects.get(pk=event_id)
            if not check_event_access(request.user, event):
                return Response({"error": "Not allowed"}, status=403)

            # Get the workers who have joined the event with approved status
            event_workers = EventWorker.objects.filter(event=event, status=EventWorker.APPROVED)

            # Prepare data to return
            data = []
            for event_worker in event_workers:
                serializer = WorkerEventInfoSerializer(event_worker)
                data.append(serializer.data)

            return Response(data, status=status.HTTP_200_OK)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)

# views.py
class SubmissionStatusUpdateAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, pk):
        submission = get_object_or_404(
            Submission.objects.select_related("worker", "worker__user", "event"),
            pk=pk,
        )
        new_status = _normalize_submission_status(request.data.get("status"), default=None)
        if new_status is None:
            return Response({"error": "A valid status is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Critical invariant: generic APIs never approve submissions.
        if new_status == Submission.APPROVED:
            return Response(
                {
                    "code": "pipeline_managed_status",
                    "message": "Submission approval can only be produced by the validated server pipeline.",
                },
                status=status.HTTP_409_CONFLICT,
            )
        if not can_update_submission_status(request.user, submission, new_status):
            return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

        new_message = _extract_submission_message(
            request.data,
            fallback="Submission cancelled/refused.",
        )
        with transaction.atomic():
            submission = Submission.objects.select_for_update().get(pk=pk)
            if submission.status == Submission.APPROVED or submission.flow_stage == Submission.FLOW_FINALIZED:
                return Response(
                    {"code": "already_finalized", "message": "Finalized approved submissions cannot be changed here."},
                    status=status.HTTP_409_CONFLICT,
                )
            submission.status = Submission.REFUSED
            submission.flow_stage = Submission.FLOW_REFUSED
            submission.message = new_message or "Submission refused."
            submission.save(update_fields=["status", "flow_stage", "message"])

        return Response(
            {
                "id": submission.id,
                "submission_id": submission.id,
                "status": submission.status,
                "flow_stage": submission.flow_stage,
                "message": submission.message,
            },
            status=status.HTTP_200_OK,
        )

class SubmissionReportListCreate(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser, MultiPartParser, FormParser]  # ✅ important

    def post(self, request):
        # 1) Validate non-file fields using your serializer (keep your serializer unchanged)
        ser = SubmissionReportSerializer(data=request.data, context={"request": request})
        ser.is_valid(raise_exception=True)

        # 2) Read payload exactly as frontend sends it
        report_type = (request.data.get("type") or PHOTO).lower()
        filename = request.data.get("filename") or ("reported_video.mp4" if report_type == VIDEO else "reported_photo.jpg")
        file_b64 = (request.data.get("file_base64") or "").strip()

        if not file_b64:
            return Response({"file_base64": "This field is required."}, status=400)

        # 3) Create instance WITHOUT saving file/thumb yet
        instance: SubmissionReport = ser.save()

        # 4) Blur + save to DB fields
        if report_type == PHOTO:
            file_cf, thumb_cf = self._blur_photo_to_contentfiles(file_b64, filename)
        elif report_type == VIDEO:
            file_cf, thumb_cf = self._blur_video_to_contentfiles(file_b64, filename)
        else:
            return Response({"type": "type must be 'photo' or 'video'."}, status=400)

        instance.file.save(file_cf.name, file_cf, save=False)
        instance.thumbnail.save(thumb_cf.name, thumb_cf, save=False)
        instance.save(update_fields=["file", "thumbnail"])

        # 5) Return
        return Response(SubmissionReportSerializer(instance, context={"request": request}).data, status=status.HTTP_201_CREATED)

    # -------- helpers --------

    def _blur_photo_to_contentfiles(self, file_b64: str, filename: str):
        # your helper accepts raw base64 (no data: prefix needed)
        out = blur_faces_pipeline_from_base64(file_b64, filename_prefix="report_blurred")
        blurred_jpeg = out["image_bytes"]  # JPEG bytes

        base = os.path.splitext(os.path.basename(filename))[0] or "report"
        main_name = f"{base}_blurred.jpg"

        thumb_bytes = self._make_thumb_jpeg(blurred_jpeg, size=(300, 300), quality=60)
        thumb_name = f"thumb_{base}_blurred.jpg"

        return ContentFile(blurred_jpeg, name=main_name), ContentFile(thumb_bytes, name=thumb_name)

    def _make_thumb_jpeg(self, jpeg_bytes: bytes, size=(300, 300), quality=60) -> bytes:
        img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
        img = ImageOps.exif_transpose(img)
        img.thumbnail(size)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def _blur_video_to_contentfiles(self, file_b64: str, filename: str):
        # decode base64 video
        raw = base64.b64decode(file_b64)

        # reuse your existing video blurring pipeline (writes temp files)
        import tempfile, uuid, subprocess, shutil

        tmp_dir = tempfile.mkdtemp(prefix=f"report_vid_{uuid.uuid4().hex[:8]}_")
        try:
            src_path = os.path.join(tmp_dir, "input.mp4")
            with open(src_path, "wb") as f:
                f.write(raw)

            out_path = os.path.join(tmp_dir, "blurred_intermediate.mp4")
            blur_video_faces_sparse(
                in_path=src_path,
                out_path=out_path,
                stride=1,
                margin=0.25,
                conf=0.25,
                imgsz=640,
                blur_ksize=31,
                writer_fourcc="mp4v",
            )

            # full + preview (thumbnail)
            full_path = os.path.join(tmp_dir, "blurred_full.mp4")
            subprocess.run(
                ["ffmpeg","-y","-i",out_path,"-c:v","libx264","-preset","fast","-crf","20","-movflags","+faststart",full_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )

            preview_path = os.path.join(tmp_dir, "preview.mp4")
            subprocess.run(
                ["ffmpeg","-y","-i",out_path,"-vf","scale=854:480","-b:v","800k","-c:v","libx264","-preset","fast","-movflags","+faststart",preview_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )

            with open(full_path, "rb") as f:
                full_bytes = f.read()
            with open(preview_path, "rb") as f:
                preview_bytes = f.read()

            base = os.path.splitext(os.path.basename(filename))[0] or "report"
            main_name = f"{base}_blurred.mp4"
            thumb_name = f"preview_{base}_blurred.mp4"

            return ContentFile(full_bytes, name=main_name), ContentFile(preview_bytes, name=thumb_name)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

class SubmissionReportListAll(APIView):
    """
    GET /api/reports/all/ -> list every report for administrators.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if (
            getattr(request.user, "role", None) != "Admin"
            and not getattr(request.user, "is_superuser", False)
        ):
            return Response(
                {"error": "Admin only"},
                status=status.HTTP_403_FORBIDDEN,
            )

        qs = SubmissionReport.objects.all()
        ser = SubmissionReportSerializer(qs, many=True)
        return Response(ser.data)

class WorkerEventEarningsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, event_id, worker_id):
        """
        Returns total earnings of a worker for a given event.
        """
        try:
            event = Event.objects.get(pk=event_id)
            if not check_event_access(request.user, event):
                return Response({"error": "Not allowed"}, status=403)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=status.HTTP_404_NOT_FOUND)

        try:
            worker = Worker.objects.get(pk=worker_id)
        except Worker.DoesNotExist:
            return Response({"error": "Worker not found"}, status=status.HTTP_404_NOT_FOUND)

        # Filter only approved submissions for that worker and event
        approved_subs = Submission.objects.filter(
            worker=worker,
            event=event,
            status=Submission.APPROVED
        )

        # Count types
        photo_count = approved_subs.filter(photo__isnull=False).count()
        video_count = approved_subs.filter(video__isnull=False).count()
        text_count = approved_subs.filter(
            photo__isnull=True,
            video__isnull=True,
            text__isnull=False,
        ).exclude(text='').count()

        # Compute totals
        photo_total = round(photo_count * (event.photo_reward or 0), 3)
        video_total = round(video_count * (event.video_reward or 0), 3)
        text_total = round(text_count * (getattr(event, 'text_reward', 0) or 0), 3)
        total = round(photo_total + video_total + text_total, 3)

        data = {
            "event_id": event.id,
            "worker_id": worker.id,
            "photo_submissions": photo_count,
            "video_submissions": video_count,
            "text_submissions": text_count,
            "photo_reward_total": photo_total,
            "video_reward_total": video_total,
            "text_reward_total": text_total,
            "total_earned": total
        }

        return Response(data, status=status.HTTP_200_OK)

class WorkerTotalEarningsView(APIView):
    permission_classes = [IsAuthenticated]
    """
    Returns all earnings of a worker across all events.
    """
    def get(self, request, worker_id):
        try:
            worker = Worker.objects.get(pk=worker_id)
        except Worker.DoesNotExist:
            return Response({"error": "Worker not found"}, status=status.HTTP_404_NOT_FOUND)

        # Fetch all approved submissions by this worker
        approved_subs = Submission.objects.filter(
            worker=worker,
            status=Submission.APPROVED
        ).select_related("event")

        # Prepare aggregation
        event_earnings = {}
        for sub in approved_subs:
            ev = sub.event
            if ev.id not in event_earnings:
                event_earnings[ev.id] = {
                    "event_id": ev.id,
                    "event_title": ev.title,
                    "photo_submissions": 0,
                    "video_submissions": 0,
                    "text_submissions": 0,
                    "photo_reward_total": 0.0,
                    "video_reward_total": 0.0,
                    "text_reward_total": 0.0,
                    "total_earned": 0.0,
                }

            # classify submission type
            if sub.photo_id:
                event_earnings[ev.id]["photo_submissions"] += 1
                event_earnings[ev.id]["photo_reward_total"] += float(ev.photo_reward or 0)
            if sub.video_id:
                event_earnings[ev.id]["video_submissions"] += 1
                event_earnings[ev.id]["video_reward_total"] += float(ev.video_reward or 0)
            if not sub.photo_id and not sub.video_id and getattr(sub, "text", None):
                event_earnings[ev.id]["text_submissions"] += 1
                event_earnings[ev.id]["text_reward_total"] += float(getattr(ev, "text_reward", 0) or 0)

        # compute total sums
        for evdata in event_earnings.values():
            evdata["photo_reward_total"] = round(evdata["photo_reward_total"], 3)
            evdata["video_reward_total"] = round(evdata["video_reward_total"], 3)
            evdata["text_reward_total"] = round(evdata["text_reward_total"], 3)
            evdata["total_earned"] = round(
                evdata["photo_reward_total"] + evdata["video_reward_total"] + evdata["text_reward_total"], 3
            )

        total_all_events = round(
            sum(e["total_earned"] for e in event_earnings.values()), 3
        )

        return Response({
            "worker_id": worker.id,
            "total_earned_all_events": total_all_events,
            "events": list(event_earnings.values())
        }, status=status.HTTP_200_OK)


import base64
import os
import subprocess
import tempfile
import time
import uuid
from io import BytesIO

from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404
from PIL import Image, ImageOps
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

import logging

logger = logging.getLogger(__name__)

def _safe_preview(text, limit=80):
    clean = (text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."
def _text_safety_log(message, level="info"):
    """
    Print + logger so you can see logs in runserver and production logs.
    """
    print(message, flush=True)

    if level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)
def reject_if_text_unsafe(text, field_name, label=None, user=None):
    """
    Returns DRF Response with HTTP 200 if text is unsafe.
    Returns None if text is empty or safe.

    This is intentional:
    unsafe text is a handled validation result, not a technical API failure.
    """

    label = label or field_name
    clean_text = (text or "").strip()
    user_id = getattr(user, "id", None)

    if not clean_text:
        _text_safety_log(
            f"[textual-safety] SKIP field={field_name} user_id={user_id} reason=empty"
        )
        return None

    _text_safety_log(
        "[textual-safety] START "
        f"field={field_name} label={label} user_id={user_id} "
        f"length={len(clean_text)} preview='{_safe_preview(clean_text)}'"
    )

    try:
        safety = check_text_safety(
            text=clean_text,
            use_llm=True,
            output_language="auto",
        )
    except Exception as e:
        _text_safety_log(
            f"[textual-safety] ERROR field={field_name} user_id={user_id} error={str(e)}",
            level="error",
        )

        return Response(
            {
                "status": "error",
                "ok": False,
                "saved": False,
                "code": "text_safety_failed",
                "field": field_name,
                "message": f"{label} could not be checked. Please try again.",
                "reason": str(e),
            },
            status=status.HTTP_200_OK,
        )

    _text_safety_log(
        "[textual-safety] RESULT "
        f"field={field_name} user_id={user_id} "
        f"ok={safety.get('ok')} "
        f"safe={safety.get('safe')} "
        f"is_safe_for_work={safety.get('is_safe_for_work')} "
        f"method={safety.get('method')} "
        f"score={safety.get('score')} "
        f"categories={safety.get('categories')} "
        f"matched_terms={safety.get('matched_terms')}"
    )

    if not safety.get("ok"):
        _text_safety_log(
            f"[textual-safety] BLOCK field={field_name} user_id={user_id} reason=check_not_ok",
            level="warning",
        )

        return Response(
            {
                "status": "error",
                "ok": False,
                "saved": False,
                "code": "text_safety_failed",
                "field": field_name,
                "message": f"{label} could not be checked. Please try again.",
                "reason": safety.get("error") or safety.get("message"),
                "safety": safety,
            },
            status=status.HTTP_200_OK,
        )

    if not safety.get("is_safe_for_work", True):
        _text_safety_log(
            "[textual-safety] BLOCK "
            f"field={field_name} user_id={user_id} "
            f"reason={safety.get('reason')} "
            f"categories={safety.get('categories')}",
            level="warning",
        )

        return Response(
            {
                "status": "error",
                "ok": True,
                "saved": False,
                "code": "unsafe_text",
                "field": field_name,
                "message": f"{label} contains unsafe content. Please rewrite it.",
                "reason": safety.get("reason"),
                "categories": safety.get("categories", []),
                "matched_terms": safety.get("matched_terms", []),
                "safe": safety.get("safe"),
                "score": safety.get("score"),
                "safety": safety,
            },
            status=status.HTTP_200_OK,
        )

    _text_safety_log(
        f"[textual-safety] PASS field={field_name} user_id={user_id}"
    )

    return None
class SubmissionReportCreateAPIView(APIView):
    """
    Create or update a report for one declined submission.

    CREATE payload:
    {
      "submission_id": 123,
      "type": "photo" | "video",
      "comment": "...",
      "submission_message": "...",   # optional
      "filename": "something.jpg",   # optional
      "file_base64": "AAA...AAA"     # required on first create
    }

    UPDATE payload:
    {
      "submission_id": 123,
      "comment": "...",
      "submission_message": "..."    # optional
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        user = request.user

        _text_safety_log(
            f"[report-api] START user_id={getattr(user, 'id', None)}"
        )

        # 1) Resolve worker
        try:
            worker = Worker.objects.get(user=user)
        except Worker.DoesNotExist:
            _text_safety_log(
                f"[report-api] FORBIDDEN user_id={getattr(user, 'id', None)} reason=not_worker",
                level="warning",
            )
            return Response(
                {
                    "status": "error",
                    "message": "Only workers can submit reports.",
                    "detail": "Only workers can submit reports.",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        data = request.data

        # 2) submission_id is mandatory
        submission_id = data.get("submission_id")
        if not submission_id:
            return Response(
                {
                    "status": "error",
                    "message": "submission_id is required.",
                    "detail": "submission_id is required.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        submission = get_object_or_404(Submission, pk=submission_id)

        submission_worker_id = getattr(submission, "worker_id", None)
        if submission_worker_id is not None and submission_worker_id != worker.pk:
            _text_safety_log(
                f"[report-api] FORBIDDEN user_id={getattr(user, 'id', None)} "
                f"submission_id={submission_id} reason=other_worker_submission",
                level="warning",
            )
            return Response(
                {
                    "status": "error",
                    "message": "You cannot report another worker's submission.",
                    "detail": "You cannot report another worker's submission.",
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        # 3) Validate text fields
        comment = (data.get("comment") or "").strip()
        if not comment:
            return Response(
                {
                    "status": "error",
                    "message": "comment is required.",
                    "detail": "comment is required.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        submission_message = (data.get("submission_message") or "").strip() or None

        _text_safety_log(
            f"[report-api] TEXT_CHECK_START user_id={getattr(user, 'id', None)} "
            f"submission_id={submission_id} "
            f"comment_len={len(comment)} "
            f"submission_message_len={len(submission_message or '')}"
        )

        safety_response = reject_if_text_unsafe(
            comment,
            field_name="comment",
            label="Report comment",
            user=user,
        )

        if safety_response is not None:
            _text_safety_log(
                f"[report-api] BLOCKED_COMMENT submission_id={submission_id}",
                level="warning",
            )
            return safety_response

        safety_response = reject_if_text_unsafe(
            submission_message,
            field_name="submission_message",
            label="Submission message",
            user=user,
        )

        if safety_response is not None:
            _text_safety_log(
                f"[report-api] BLOCKED_SUBMISSION_MESSAGE submission_id={submission_id}",
                level="warning",
            )
            return safety_response

        _text_safety_log(
            f"[report-api] TEXT_CHECK_PASS submission_id={submission_id}"
        )

        # 4) Check if a report already exists
        existing = SubmissionReport.objects.filter(
            worker=worker,
            submission=submission,
        ).first()

        # ------------------------------------------------------------
        # UPDATE existing report
        # ------------------------------------------------------------
        if existing:
            existing.comment = comment
            existing.submission_message = submission_message
            existing.save(update_fields=["comment", "submission_message"])

            _text_safety_log(
                f"[report-api] UPDATED report_id={existing.id} submission_id={submission_id}"
            )

            return Response(
                {
                    "status": "success",
                    "message": "Report updated successfully.",
                    "id": existing.id,
                    "type": existing.type,
                    "comment": existing.comment,
                    "submission_message": existing.submission_message,
                    "file_url": self._abs_url(request, existing.file),
                    "thumbnail_url": self._abs_url(request, existing.thumbnail),
                    "created_at": existing.created_at.isoformat() if existing.created_at else None,
                    "updated": True,
                },
                status=status.HTTP_200_OK,
            )

        # ------------------------------------------------------------
        # CREATE new report
        # ------------------------------------------------------------
        raw_type = (data.get("type") or "").lower()
        if raw_type not in (PHOTO, VIDEO):
            return Response(
                {
                    "status": "error",
                    "message": "type must be 'photo' or 'video'.",
                    "detail": "type must be 'photo' or 'video'.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        file_b64 = data.get("file_base64")
        if not file_b64:
            return Response(
                {
                    "status": "error",
                    "message": "file_base64 is required for first report creation.",
                    "detail": "file_base64 is required for first report creation.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if isinstance(file_b64, str) and file_b64.startswith("data:"):
            try:
                file_b64 = file_b64.split(",", 1)[1]
            except IndexError:
                return Response(
                    {
                        "status": "error",
                        "message": "Invalid data URL for file_base64.",
                        "detail": "Invalid data URL for file_base64.",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        orig_name = (data.get("filename") or "").strip()
        base, _ext = os.path.splitext(orig_name)
        if not base:
            ts = int(time.time() * 1000)
            base = f"reported_{raw_type}_{ts}"

        event = getattr(submission, "event", None)

        report = SubmissionReport(
            worker=worker,
            submission=submission,
            event=event,
            type=raw_type,
            comment=comment,
            submission_message=submission_message,
        )

        if raw_type == PHOTO:
            try:
                _text_safety_log(
                    f"[report-api] PHOTO_BLUR_START submission_id={submission_id}"
                )

                blur_out = blur_faces_pipeline_from_base64(
                    file_b64,
                    filename_prefix="report_blurred",
                )
                blurred_bytes = blur_out["image_bytes"]

                _text_safety_log(
                    f"[report-api] PHOTO_BLUR_DONE submission_id={submission_id}"
                )

            except Exception as e:
                _text_safety_log(
                    f"[report-api] PHOTO_BLUR_ERROR submission_id={submission_id} error={str(e)}",
                    level="error",
                )
                return Response(
                    {
                        "status": "error",
                        "message": "Photo blurring failed.",
                        "detail": "photo blurring failed",
                        "error": str(e),
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            safe_name = f"{base}.jpg"
            report.file.save(safe_name, ContentFile(blurred_bytes), save=False)

            thumb_bytes, thumb_ext = self._build_photo_thumb(blurred_bytes)
            if thumb_bytes:
                report.thumbnail.save(
                    f"thumb_{base}{thumb_ext}",
                    ContentFile(thumb_bytes),
                    save=False,
                )

        else:
            try:
                video_bytes = base64.b64decode(file_b64, validate=False)
            except Exception:
                return Response(
                    {
                        "status": "error",
                        "message": "Invalid base64 payload.",
                        "detail": "Invalid base64 payload.",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            try:
                _text_safety_log(
                    f"[report-api] VIDEO_BLUR_START submission_id={submission_id}"
                )

                full_bytes, preview_bytes = self._blur_video_and_make_preview(video_bytes)

                _text_safety_log(
                    f"[report-api] VIDEO_BLUR_DONE submission_id={submission_id}"
                )

            except Exception as e:
                _text_safety_log(
                    f"[report-api] VIDEO_BLUR_ERROR submission_id={submission_id} error={str(e)}",
                    level="error",
                )
                return Response(
                    {
                        "status": "error",
                        "message": "Video blurring failed.",
                        "detail": "video blurring failed",
                        "error": str(e),
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            safe_name = f"{base}.mp4"
            report.file.save(safe_name, ContentFile(full_bytes), save=False)

            if preview_bytes:
                report.thumbnail.save(
                    f"thumb_{base}.mp4",
                    ContentFile(preview_bytes),
                    save=False,
                )

        report.save()

        _text_safety_log(
            f"[report-api] CREATED report_id={report.id} "
            f"submission_id={submission_id} type={raw_type}"
        )

        # Notify admins only on true creation
        if event is not None:
            worker_name = (
                    (user.get_full_name() or "").strip()
                    or user.email
                    or f"Worker #{user.id}"
            )

            title_admin = "New report submitted"
            body_admin = f"{worker_name} reported a {raw_type} in “{event.title}”"

            payload_admin = make_payload(
                type="submission.report.created",
                event_id=event.id,
                event_name=event.title,
                report_id=report.id,
                report_type=raw_type,
                worker_id=worker.user_id,
                submission_message=submission_message,
                status="created",
                body_for_ui=body_admin,
            )

            notify_admins(
                event_type="submission.report.created",
                title=title_admin,
                body=body_admin,
                payload=payload_admin,
                priority="high",
            )

            _text_safety_log(
                f"[report-api] ADMIN_NOTIFIED report_id={report.id} event_id={event.id}"
            )

        return Response(
            {
                "status": "success",
                "message": "Report sent successfully.",
                "id": report.id,
                "type": report.type,
                "comment": report.comment,
                "submission_message": report.submission_message,
                "file_url": self._abs_url(request, report.file),
                "thumbnail_url": self._abs_url(request, report.thumbnail),
                "created_at": report.created_at.isoformat() if report.created_at else None,
                "updated": False,
            },
            status=status.HTTP_201_CREATED,
        )

    def _abs_url(self, request, f):
        if not f:
            return None
        try:
            return request.build_absolute_uri(f.url)
        except Exception:
            return None

    def _build_photo_thumb(self, file_bytes):
        try:
            img = Image.open(BytesIO(file_bytes))
            img = img.convert("RGB")
            img = ImageOps.exif_transpose(img)
            img.thumbnail((300, 300))

            out = BytesIO()
            img.save(out, format="JPEG", quality=60)
            return out.getvalue(), ".jpg"
        except Exception:
            return None, None

    def _blur_video_and_make_preview(self, file_bytes: bytes) -> tuple[bytes, bytes]:
        with tempfile.TemporaryDirectory(prefix=f"report_vid_{uuid.uuid4().hex[:8]}_") as tmp_dir:
            src_path = os.path.join(tmp_dir, f"input_{uuid.uuid4().hex}.mp4")
            with open(src_path, "wb") as f:
                f.write(file_bytes)

            out_path = os.path.join(tmp_dir, f"blurred_{uuid.uuid4().hex}.mp4")
            blur_video_faces_sparse(
                in_path=src_path,
                out_path=out_path,
                stride=1,
                margin=0.25,
                conf=0.25,
                imgsz=640,
                blur_ksize=31,
                writer_fourcc="mp4v",
            )

            full_blurred_path = os.path.join(tmp_dir, f"blurred_full_{uuid.uuid4().hex}.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", out_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "20",
                    "-movflags", "+faststart",
                    full_blurred_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            preview_path = os.path.join(tmp_dir, f"preview_{uuid.uuid4().hex}.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", out_path,
                    "-vf", "scale=854:480",
                    "-b:v", "800k",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-movflags", "+faststart",
                    preview_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            with open(full_blurred_path, "rb") as f:
                full_bytes = f.read()
            with open(preview_path, "rb") as f:
                preview_bytes = f.read()

            return full_bytes, preview_bytes
class AddSubmissionMessageAPIView(APIView):
    permission_classes = [IsAuthenticated]
    """
    Set or update the message for an APPROVED submission.

    Expected JSON body:
    {
        "submission_id": 123,
        "worker_message": "Some text..."
    }
    """

    def post(self, request, *args, **kwargs):
        user = request.user
        submission_id = request.data.get("submission_id")
        worker_message = request.data.get("worker_message", "")
        worker_message = worker_message.strip()

        _text_safety_log(
            f"[worker-message-api] START user_id={getattr(user, 'id', None)} "
            f"submission_id={submission_id} message_len={len(worker_message)}"
        )

        if not submission_id:
            return Response(
                {
                    "status": "error",
                    "message": "submission_id is required.",
                    "detail": "submission_id is required.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            submission = Submission.objects.get(id=submission_id)
        except Submission.DoesNotExist:
            _text_safety_log(
                f"[worker-message-api] NOT_FOUND submission_id={submission_id}",
                level="warning",
            )
            return Response(
                {
                    "status": "error",
                    "message": "Submission not found.",
                    "detail": "Submission not found.",
                },
                status=status.HTTP_404_NOT_FOUND,
            )

        if not (
            (submission.worker and submission.worker.user_id == user.id)
            or getattr(user, "role", None) == "Admin"
            or getattr(user, "is_superuser", False)
        ):
            return Response({"error": "Not allowed"}, status=status.HTTP_403_FORBIDDEN)

        if submission.status != Submission.APPROVED:
            _text_safety_log(
                f"[worker-message-api] BLOCK_NOT_APPROVED submission_id={submission_id} status={submission.status}",
                level="warning",
            )
            return Response(
                {
                    "status": "error",
                    "message": "Message can only be set for approved submissions.",
                    "detail": "Message can only be set for approved submissions.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        safety_response = reject_if_text_unsafe(
            worker_message,
            field_name="worker_message",
            label="Submission note",
            user=user,
        )

        if safety_response is not None:
            _text_safety_log(
                f"[worker-message-api] BLOCKED_BY_TEXT_SAFETY submission_id={submission_id}",
                level="warning",
            )
            return safety_response

        submission.worker_message = worker_message or None
        submission.save(update_fields=["worker_message"])

        _text_safety_log(
            f"[worker-message-api] SAVED submission_id={submission.id} "
            f"worker_message_present={bool(submission.worker_message)}"
        )

        return Response(
            {
                "status": "success",
                "message": "Note saved successfully.",
                "id": submission.id,
                "submission_status": submission.status,
                "worker_message": submission.worker_message,
            },
            status=status.HTTP_200_OK,
        )


class MarkSubmissionMessageReadAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        submission_id = request.data.get('submission_id')

        if not submission_id:
            return Response({"detail": "submission_id is required."},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            submission = Submission.objects.select_related(
                "worker",
                "worker__user",
                "event",
            ).get(id=submission_id)
        except Submission.DoesNotExist:
            return Response(
                {"detail": "Submission not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if not can_access_submission_file(request.user, submission):
            return Response(
                {"error": "Not allowed"},
                status=status.HTTP_403_FORBIDDEN,
            )

        if not submission.worker_message:
            return Response({"detail": "No message to mark as read."},
                            status=status.HTTP_400_BAD_REQUEST)

        submission.worker_message_read = True
        submission.save(update_fields=['worker_message_read'])

        return Response({
            "id": submission.id,
            "worker_message_read": submission.worker_message_read,
        })
