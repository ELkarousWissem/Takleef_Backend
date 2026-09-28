# Adapted from faten migration 0037_user_requester_request_pending.py.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mobicrowd", "0039_alter_requester_organization_name"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="requester_request_pending",
            field=models.BooleanField(default=False),
        ),
    ]
