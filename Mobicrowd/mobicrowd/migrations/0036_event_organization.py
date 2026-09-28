# Adapted from faten migration 0033_event_organization.py.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mobicrowd", "0035_organizationinvitation_token"),
    ]

    operations = [
        migrations.AddField(
            model_name="event",
            name="organization",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="events",
                to="mobicrowd.organization",
            ),
        ),
    ]
