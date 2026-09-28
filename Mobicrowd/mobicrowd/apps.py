from django.apps import AppConfig


class MobicrowdConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'mobicrowd'

    def ready(self):
        import mobicrowd.signals
