# Adapted from faten migration 0036_alter_requester_organization_name.py.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mobicrowd", "0038_event_organization_membership"),
    ]

    operations = [
        migrations.AlterField(
            model_name="requester",
            name="organization_name",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
