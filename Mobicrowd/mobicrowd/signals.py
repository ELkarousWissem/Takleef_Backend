from django.db.models import Q
from django.db.models.signals import pre_delete
from django.dispatch import receiver

from mobicrowd.models.notifications import Notification
from mobicrowd.models.submisson import Event


@receiver(pre_delete, sender=Event)
def delete_event_notifications(sender, instance: Event, **kwargs):
    """
    Remove every persisted notification associated with an event
    before the event itself is deleted.

    This covers old/read/unread notifications and works regardless
    of where the Event deletion originated.
    """

    event_id = instance.pk

    if not event_id:
        return

    Notification.objects.filter(
        Q(payload__event_id=event_id) |
        Q(payload__event_id=str(event_id))
    ).delete()