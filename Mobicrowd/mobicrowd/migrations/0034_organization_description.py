# Adapted from faten migration 0031_organization_description.py.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mobicrowd", "0033_remove_organization_website"),
    ]

    operations = [
        migrations.AddField(
            model_name="organization",
            name="description",
            field=models.TextField(blank=True, null=True),
        ),
    ]
