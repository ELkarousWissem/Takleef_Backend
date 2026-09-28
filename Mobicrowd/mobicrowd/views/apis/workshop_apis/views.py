from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from django.core.files.base import ContentFile
from django.db import transaction
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from django.utils import timezone

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from mobicrowd.models.FileStorage import CustomS3Boto3Storage
from mobicrowd.models.submisson import (
    Event,
    Submission,
    Workshop,
    WorkshopActionResult,
    WorkshopMedia,
)

from .serializers import (
    WorkshopActionResultSerializer,
    WorkshopListSerializer,
    WorkshopMediaBulkAddSerializer,
    WorkshopMediaSerializer,
    WorkshopSerializer,
    is_valid_workshop_event,
)


DATA_URL_RE = re.compile(
    r'^data:(?P<mime>[-\w.]+/[-\w.+]+);base64,(?P<data>.*)$',
    re.IGNORECASE | re.DOTALL,
)

BASE64_OUTPUT_HINTS = (
    'annotated_image_base64',
    'enhanced_image_base64',
    'segmented_image_base64',
    'mask_base64',
    'image_base64',
    'output_image_base64',
    'result_image_base64',
    'removed_background_base64',
    'background_removed_base64',
    'annotated_video_base64',
    'processed_video_base64',
    'output_video_base64',
    'result_video_base64',
    'video_base64',
    'media_base64',
    'output_base64',
    'result_base64',
    'base64',
)

INPUT_BASE64_SKIP_HINTS = (
    'input',
    'source',
    'original',
    'raw_request',
    'request',
)


def _safe_json(value: Any, default: Any) -> Any:
    if value in (None, ''):
        return default

    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return default


def _normalize_uuid(value: Any) -> uuid.UUID:
    if value in (None, '', 'null'):
        return uuid.uuid4()

    try:
        return uuid.UUID(str(value))
    except Exception:
        return uuid.uuid4()


def _clean_filename(value: str, fallback: str) -> str:
    candidate = Path(str(value or '').strip()).name
    candidate = re.sub(r'[^A-Za-z0-9._-]+', '_', candidate).strip('._-')
    return candidate or fallback


def _extension_for_mime(mime_type: str, media_kind: str) -> str:
    guessed = mimetypes.guess_extension(mime_type or '')

    if guessed:
        return '.jpg' if guessed == '.jpe' else guessed

    if (mime_type or '').startswith('video/') or media_kind == 'video':
        return '.mp4'

    return '.jpg'


def _decode_base64_candidate(value: str, default_mime: str) -> Optional[Tuple[bytes, str]]:
    if not isinstance(value, str):
        return None

    text = value.strip()

    if len(text) < 80:
        return None

    mime_type = default_mime
    payload = text

    match = DATA_URL_RE.match(text)
    if match:
        mime_type = match.group('mime') or default_mime
        payload = match.group('data')
    else:
        compact = re.sub(r'\s+', '', text)
        if not re.fullmatch(r'[A-Za-z0-9+/=_-]+', compact):
            return None
        payload = compact

    payload = re.sub(r'\s+', '', payload)

    if not payload:
        return None

    missing_padding = len(payload) % 4
    if missing_padding:
        payload += '=' * (4 - missing_padding)

    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None

    if len(raw) < 32:
        return None

    return raw, mime_type


def _find_first_base64_media(
    value: Any,
    *,
    media_kind: str,
    path: str = '',
) -> Optional[Tuple[str, bytes, str]]:
    default_mime = 'video/mp4' if media_kind == 'video' else 'image/jpeg'

    if isinstance(value, dict):
        for key, nested in value.items():
            key_lower = str(key).lower()
            next_path = f'{path}.{key}' if path else str(key)

            if any(skip in key_lower for skip in INPUT_BASE64_SKIP_HINTS):
                continue

            if any(key_lower == hint or key_lower.endswith(hint) or hint in key_lower for hint in BASE64_OUTPUT_HINTS):
                decoded = _decode_base64_candidate(nested, default_mime)
                if decoded:
                    raw, mime_type = decoded
                    return next_path, raw, mime_type

        for key, nested in value.items():
            key_lower = str(key).lower()
            next_path = f'{path}.{key}' if path else str(key)

            if any(skip in key_lower for skip in INPUT_BASE64_SKIP_HINTS):
                continue

            found = _find_first_base64_media(
                nested,
                media_kind=media_kind,
                path=next_path,
            )

            if found:
                return found

    if isinstance(value, list):
        for index, nested in enumerate(value):
            found = _find_first_base64_media(
                nested,
                media_kind=media_kind,
                path=f'{path}[{index}]',
            )

            if found:
                return found

    if isinstance(value, str):
        lower = value.strip().lower()
        if lower.startswith('data:image/') or lower.startswith('data:video/'):
            decoded = _decode_base64_candidate(value, default_mime)
            if decoded:
                raw, mime_type = decoded
                return path or 'value', raw, mime_type

    return None


def _action_output_storage_folder(
    *,
    workshop: Workshop,
    media: Optional[WorkshopMedia],
    run_uid: str,
) -> str:
    media_part = f'media_{media.id}' if media else 'media_none'
    run_part = _clean_filename(run_uid or uuid.uuid4().hex, 'run')
    return f'workshop_{workshop.id}/{run_part}/{media_part}'


def _storage_url(storage: CustomS3Boto3Storage, key: str, request=None) -> str:
    try:
        url = storage.url(key)
    except Exception:
        url = key

    if request and isinstance(url, str) and url.startswith('/'):
        return request.build_absolute_uri(url)

    return url


def _save_action_output_media_to_s3(
    *,
    workshop: Workshop,
    media: Optional[WorkshopMedia],
    run_uid: str,
    action_key: str,
    media_kind: str,
    media_name: str,
    result_payload: Any,
    request=None,
) -> Optional[Dict[str, Any]]:
    """
    Extracts only the generated output image/video from the action response,
    uploads it to S3, and returns metadata for dedicated DB columns.

    The full payload is not stored in DB.
    """
    found = _find_first_base64_media(result_payload, media_kind=media_kind)

    if not found:
        return None

    source_field, raw_bytes, mime_type = found

    if media_kind == 'video' or (mime_type or '').startswith('video/'):
        storage_folder = 'workshop-action-results/videos'
        output_kind = 'video'
    else:
        storage_folder = 'workshop-action-results/images'
        output_kind = 'photo'

    extension = _extension_for_mime(mime_type, output_kind)
    base_name = _clean_filename(media_name or f'{action_key}_{media.id if media else "media"}', 'media')
    stem = Path(base_name).stem or 'media'
    file_name = f'{action_key}_{stem}_{uuid.uuid4().hex[:12]}{extension}'

    storage = CustomS3Boto3Storage(
        folder_name=storage_folder,
        dynamic_folder_name=_action_output_storage_folder(
            workshop=workshop,
            media=media,
            run_uid=run_uid,
        ),
    )

    saved_key = storage.save(
        file_name,
        ContentFile(raw_bytes, name=file_name),
    )

    return {
        'kind': output_kind,
        'storage_key': saved_key,
        'url': _storage_url(storage, saved_key, request=request),
        'mime_type': mime_type,
        'size_bytes': len(raw_bytes),
        'source_field': source_field,
        'file_name': file_name,
    }


def _extract_label_colors(result_payload: Any) -> List[Dict[str, Any]]:
    """Return only the annotation color legend block."""
    if not isinstance(result_payload, dict):
        return []

    candidates = [
        result_payload.get('label_colors'),
        (result_payload.get('result') or {}).get('label_colors') if isinstance(result_payload.get('result'), dict) else None,
        (result_payload.get('data') or {}).get('label_colors') if isinstance(result_payload.get('data'), dict) else None,
    ]

    for value in candidates:
        if isinstance(value, list):
            return _safe_json(value, [])

    return []


def _compact_result_metadata(result_payload: Any) -> Dict[str, Any]:
    """
    Store only small action metadata. Do not store detections arrays, raw output,
    base64, or full model text unless it is a short user-facing summary/story.
    """
    if not isinstance(result_payload, dict):
        return {}

    compact: Dict[str, Any] = {}

    for key in (
        'status',
        'ok',
        'decision',
        'safe',
        'color_strategy',
        'raw_model_item_count',
        'score',
        'confidence',
        'reason',
        'message',

        # Recaptured-check API fields.
        'decision_threshold',
        'recapture_score',
        'recapture_result',
        'file_type',
        'filename',

        # OCR/text extraction fields.
        'text',
        'extracted_text',
        'ocr_text',
        'detected_text',
        'language',
        'lines_count',
        'words_count',
    ):
        value = result_payload.get(key)
        if value not in (None, ''):
            compact[key] = value

    detections = result_payload.get('detections')
    if isinstance(detections, list):
        compact['detections_count'] = len(detections)

    for list_key in ('lines', 'blocks', 'words', 'ocr_lines', 'text_lines'):
        value = result_payload.get(list_key)
        if isinstance(value, list):
            compact[f'{list_key}_count'] = len(value)
            compact[list_key] = _safe_json(value[:200], [])

    label_colors = _extract_label_colors(result_payload)
    if label_colors:
        compact['labels_count'] = len(label_colors)
        compact['labels'] = [
            {
                'label': item.get('label'),
                'instance_count': item.get('instance_count'),
                'color_hex': item.get('color_hex'),
            }
            for item in label_colors
            if isinstance(item, dict)
        ]

    story = (
        result_payload.get('story') or
        result_payload.get('summary') or
        result_payload.get('caption') or
        result_payload.get('description')
    )
    if isinstance(story, str) and story.strip():
        compact['text'] = story.strip()[:4000]

    extracted_text = (
        result_payload.get('extracted_text') or
        result_payload.get('ocr_text') or
        result_payload.get('detected_text')
    )
    if isinstance(extracted_text, str) and extracted_text.strip():
        compact['extracted_text'] = extracted_text.strip()[:4000]

    return _safe_json(compact, {})


def _resolve_event_for_action_result(
    *,
    workshop: Workshop,
    media: Optional[WorkshopMedia],
    request_data: Dict[str, Any],
) -> Optional[Event]:
    if media and media.event_id:
        return media.event

    event_id = request_data.get('event') or request_data.get('event_id')

    if event_id in [None, '', 'null']:
        return None

    return get_object_or_404(
        Event,
        id=event_id,
        requester_id=workshop.requester_id,
        workshops=workshop,
    )


class WorkshopBaseMixin:
    def get_requester(self, request):
        return getattr(request.user, 'requester_profile', None)

    def get_workshop(self, request, pk):
        requester = self.get_requester(request)

        if requester is None:
            return None

        return get_object_or_404(
            Workshop.objects.prefetch_related('events'),
            pk=pk,
            requester=requester,
        )


class WorkshopListCreateAPIView(WorkshopBaseMixin, APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        requester = self.get_requester(request)

        if requester is None:
            return Response(
                {'detail': 'Only requesters can view workshops.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        workshops = (
            Workshop.objects
            .filter(requester=requester)
            .prefetch_related('events')
            .annotate(
                events_count=Count('events', distinct=True),
                media_count=Count('media_items', distinct=True),
                photo_count=Count(
                    'media_items',
                    filter=Q(media_items__media_type='photo'),
                    distinct=True,
                ),
                video_count=Count(
                    'media_items',
                    filter=Q(media_items__media_type='video'),
                    distinct=True,
                ),
            )
            .order_by('-created_at')
        )

        serializer = WorkshopListSerializer(
            workshops,
            many=True,
            context={'request': request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        requester = self.get_requester(request)

        if requester is None:
            return Response(
                {'detail': 'Only requesters can create workshops.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        serializer = WorkshopSerializer(
            data=request.data,
            context={'request': request},
        )

        if serializer.is_valid():
            workshop = serializer.save()
            return Response(
                WorkshopSerializer(
                    workshop,
                    context={'request': request},
                ).data,
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class WorkshopDetailAPIView(WorkshopBaseMixin, APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can view workshops.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        serializer = WorkshopSerializer(
            workshop,
            context={'request': request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, pk):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can update workshops.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        serializer = WorkshopSerializer(
            workshop,
            data=request.data,
            context={'request': request},
        )

        if serializer.is_valid():
            workshop = serializer.save()
            return Response(
                WorkshopSerializer(
                    workshop,
                    context={'request': request},
                ).data,
                status=status.HTTP_200_OK,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def patch(self, request, pk):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can update workshops.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        serializer = WorkshopSerializer(
            workshop,
            data=request.data,
            partial=True,
            context={'request': request},
        )

        if serializer.is_valid():
            workshop = serializer.save()
            return Response(
                WorkshopSerializer(
                    workshop,
                    context={'request': request},
                ).data,
                status=status.HTTP_200_OK,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can delete workshops.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        workshop.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class WorkshopMediaListAddAPIView(WorkshopBaseMixin, APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can view workshop media.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        media_items = (
            WorkshopMedia.objects
            .filter(workshop_id=workshop.id)
            .select_related('event', 'submission', 'photo', 'video')
            .defer('action_payload')
            .order_by('-added_at')
        )

        serializer = WorkshopMediaSerializer(
            media_items,
            many=True,
            context={'request': request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)

    @transaction.atomic
    def post(self, request, pk):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can add workshop media.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        serializer = WorkshopMediaBulkAddSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        created_items = []
        errors = []

        for item in serializer.validated_data['items']:
            submission_id = item['submission_id']
            media_type = item['media_type']

            submission = (
                Submission.objects
                .select_related('event', 'photo', 'video')
                .filter(id=submission_id)
                .first()
            )

            if submission is None:
                errors.append({
                    'submission_id': submission_id,
                    'detail': 'Submission not found.',
                })
                continue

            event = submission.event

            if event.requester_id != workshop.requester_id:
                errors.append({
                    'submission_id': submission_id,
                    'detail': 'This submission does not belong to this requester.',
                })
                continue

            if not workshop.events.filter(id=event.id).exists():
                errors.append({
                    'submission_id': submission_id,
                    'detail': 'This submission event is not selected in the workshop.',
                })
                continue

            if not is_valid_workshop_event(event):
                errors.append({
                    'submission_id': submission_id,
                    'detail': 'This event is not valid for workshop media because it contains text media type.',
                })
                continue

            event_media_types = set(event.media_types or [])

            if media_type not in event_media_types:
                errors.append({
                    'submission_id': submission_id,
                    'media_type': media_type,
                    'detail': f'This event does not allow {media_type}.',
                })
                continue

            if media_type == 'photo' and not submission.photo_id:
                errors.append({
                    'submission_id': submission_id,
                    'detail': 'This submission has no photo.',
                })
                continue

            if media_type == 'video' and not submission.video_id:
                errors.append({
                    'submission_id': submission_id,
                    'detail': 'This submission has no video.',
                })
                continue

            media_item, _created = WorkshopMedia.objects.get_or_create(
                workshop=workshop,
                submission=submission,
                media_type=media_type,
                defaults={
                    'event': event,
                    'photo': submission.photo if media_type == 'photo' else None,
                    'video': submission.video if media_type == 'video' else None,
                    'original_event_id': event.id,
                    'original_submission_id': submission.id,
                    'original_photo_id': submission.photo_id if media_type == 'photo' else None,
                    'original_video_id': submission.video_id if media_type == 'video' else None,
                },
            )

            created_items.append(media_item)

        if errors:
            return Response(
                {
                    'created': WorkshopMediaSerializer(
                        created_items,
                        many=True,
                        context={'request': request},
                    ).data,
                    'errors': errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            WorkshopMediaSerializer(
                created_items,
                many=True,
                context={'request': request},
            ).data,
            status=status.HTTP_201_CREATED,
        )


class WorkshopMediaDetailAPIView(WorkshopBaseMixin, APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, pk, media_id):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can update workshop media.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        media_item = get_object_or_404(
            WorkshopMedia.objects.defer('action_payload'),
            id=media_id,
            workshop_id=workshop.id,
        )

        serializer = WorkshopMediaSerializer(
            media_item,
            data=request.data,
            partial=True,
            context={'request': request},
        )

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, media_id):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can delete workshop media.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        media_item = get_object_or_404(
            WorkshopMedia.objects.defer('action_payload'),
            id=media_id,
            workshop_id=workshop.id,
        )

        media_item.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class WorkshopActionResultListCreateAPIView(WorkshopBaseMixin, APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can view workshop action results.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        results = (
            WorkshopActionResult.objects
            .filter(workshop_id=workshop.id)
            .select_related(
                'workshop',
                'event',
                'workshop_media',
                'workshop_media__event',
                'workshop_media__photo',
                'workshop_media__video',
            )
            .order_by('-completed_at', '-created_at')
        )

        event_id = request.query_params.get('event_id') or request.query_params.get('event')
        if event_id:
            results = results.filter(event_id=event_id)

        action_key = request.query_params.get('action_key')
        if action_key:
            results = results.filter(action_key=action_key)

        run_uid = request.query_params.get('run_uid')
        if run_uid:
            results = results.filter(run_uid=run_uid)

        serializer = WorkshopActionResultSerializer(
            results,
            many=True,
            context={'request': request},
        )

        return Response(serializer.data, status=status.HTTP_200_OK)

    @transaction.atomic
    def post(self, request, pk):
        workshop = self.get_workshop(request, pk)

        if workshop is None:
            return Response(
                {'detail': 'Only requesters can create workshop action results.'},
                status=status.HTTP_403_FORBIDDEN,
            )

        media = None
        media_id = request.data.get('workshop_media')

        if media_id not in [None, '', 'null']:
            media = get_object_or_404(
                WorkshopMedia.objects
                .select_related('event', 'photo', 'video')
                .defer('action_payload'),
                id=media_id,
                workshop_id=workshop.id,
            )

        event = _resolve_event_for_action_result(
            workshop=workshop,
            media=media,
            request_data=request.data,
        )

        run_uid = str(_normalize_uuid(request.data.get('run_uid')))
        media_kind = request.data.get('media_kind', 'photo')
        action_key = request.data.get('action_key')
        media_name = request.data.get('media_name', '')

        request_payload = _safe_json(request.data.get('request_payload'), {})
        incoming_result_payload = _safe_json(request.data.get('result_payload'), {})

        output_media = _save_action_output_media_to_s3(
            workshop=workshop,
            media=media,
            run_uid=run_uid,
            action_key=action_key or 'action',
            media_kind=media_kind,
            media_name=media_name,
            result_payload=incoming_result_payload,
            request=request,
        )

        label_colors = _extract_label_colors(incoming_result_payload)
        result_payload = _compact_result_metadata(incoming_result_payload)

        payload = {
            'run_uid': run_uid,
            'event': event.id if event else None,
            'workshop_media': media.id if media else None,
            'action_key': action_key,
            'media_kind': media_kind,
            'media_name': media_name,
            'status': request.data.get('status', 'success'),
            'output_media_url': output_media.get('url') if output_media else '',
            'output_media_storage_key': output_media.get('storage_key') if output_media else '',
            'output_media_kind': output_media.get('kind') if output_media else '',
            'output_media_mime_type': output_media.get('mime_type') if output_media else '',
            'output_media_size_bytes': output_media.get('size_bytes') if output_media else 0,
            'label_colors': label_colors,
            'request_payload': request_payload,
            'result_payload': result_payload,
            'error_message': request.data.get('error_message') or '',
            'started_at': request.data.get('started_at'),
            'completed_at': request.data.get('completed_at'),
            'duration_ms': request.data.get('duration_ms') or 0,
        }

        serializer = WorkshopActionResultSerializer(
            data=payload,
            context={'request': request},
        )

        if serializer.is_valid():
            result = serializer.save(
                workshop=workshop,
                requester=workshop.requester,
                workshop_media=media,
                event=event,
            )

            if media:
                WorkshopMedia.objects.filter(id=media.id).update(
                    action_status=WorkshopMedia.PROCESSED,
                    updated_at=timezone.now(),
                )

            return Response(
                WorkshopActionResultSerializer(
                    result,
                    context={'request': request},
                ).data,
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)