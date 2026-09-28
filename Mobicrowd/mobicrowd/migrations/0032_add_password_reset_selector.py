# Generated manually to safely add selector to existing PasswordResetToken rows.

import uuid
from django.db import migrations, models


def fill_selectors(apps, schema_editor):
    PasswordResetToken = apps.get_model("mobicrowd", "PasswordResetToken")

    for rec in PasswordResetToken.objects.filter(selector__isnull=True).only("id"):
        rec.selector = uuid.uuid4()
        rec.save(update_fields=["selector"])


class Migration(migrations.Migration):

    dependencies = [
        ("mobicrowd", "0031_photo_extracted_text"),
    ]

    operations = [
        migrations.AddField(
            model_name="passwordresettoken",
            name="selector",
            field=models.UUIDField(
                null=True,
                editable=False,
                db_index=True,
            ),
        ),

        migrations.RunPython(fill_selectors, migrations.RunPython.noop),

        migrations.AlterField(
            model_name="passwordresettoken",
            name="selector",
            field=models.UUIDField(
                default=uuid.uuid4,
                unique=True,
                db_index=True,
                editable=False,
            ),
        ),
    ]