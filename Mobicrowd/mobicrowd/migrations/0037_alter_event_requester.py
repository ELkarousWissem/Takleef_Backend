# Adapted from faten migration 0034_alter_event_requester.py.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mobicrowd", "0036_event_organization"),
    ]

    operations = [
        migrations.AlterField(
            model_name="event",
            name="requester",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="events",
                to="mobicrowd.requester",
            ),
        ),
    ]
