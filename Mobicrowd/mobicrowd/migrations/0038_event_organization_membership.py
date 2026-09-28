# Adapted from faten migration 0035_event_organization_membership.py.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mobicrowd", "0037_alter_event_requester"),
    ]

    operations = [
        migrations.AddField(
            model_name="event",
            name="organization_membership",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="organization_events",
                to="mobicrowd.organizationmembership",
            ),
        ),
    ]
