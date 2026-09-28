# mobicrowd/management/commands/schedule_event_reminders.py
from django.core.management import BaseCommand
from django.utils import timezone

from mobicrowd.models.submisson import Event
from mobicrowd.tasks import schedule_event_reminders

class Command(BaseCommand):
    help = "Schedules 80% and 30-min reminders for upcoming events."

    def handle(self, *args, **opts):
        now = timezone.now()
        qs = Event.objects.filter(deadline__gt=now)
        for e in qs.only("id","startdate","deadline"):
            schedule_event_reminders.delay(event_id=e.id)
        self.stdout.write(self.style.SUCCESS(f"Queued reminders for {qs.count()} events"))
