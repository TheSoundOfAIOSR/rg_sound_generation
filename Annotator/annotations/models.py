from django.db import models
from django.contrib.auth import get_user_model


class Annotation(models.Model):
    # Which audio file does this object annotate
    audio_file = models.CharField(max_length=100, null=False, blank=False)
    user = models.ForeignKey(
        get_user_model(),
        on_delete=models.CASCADE,
    )
    bright_vs_dark = models.IntegerField()
    full_vs_hollow = models.IntegerField()
    smooth_vs_rough = models.IntegerField()
    warm_vs_metallic = models.IntegerField()
    clear_vs_muddy = models.IntegerField()
    thin_vs_thick = models.IntegerField()
    pure_vs_noisy = models.IntegerField()
    rich_vs_sparse = models.IntegerField()
    soft_vs_hard = models.IntegerField()
    description = models.CharField(max_length=200)

    def __str__(self):
        return f'{self.pk}_{self.user.username}_{self.audio_file}'
