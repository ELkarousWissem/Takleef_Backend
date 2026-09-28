# Adapted merge migration.
# The current takleef-lava branch already has migrations 0028-0032.
# faten removed Organization.website through a rename/remove pair; on this branch
# the equivalent safe operation is to remove the original website field directly.

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("mobicrowd", "0032_add_password_reset_selector"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="organization",
            name="website",
        ),
    ]
