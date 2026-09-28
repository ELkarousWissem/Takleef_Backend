# Adapted from faten migration 0032_organizationinvitation_token.py.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mobicrowd", "0034_organization_description"),
    ]

    operations = [
        migrations.AddField(
            model_name="organizationinvitation",
            name="token",
            field=models.CharField(blank=True, max_length=128, null=True, unique=True),
        ),
    ]
