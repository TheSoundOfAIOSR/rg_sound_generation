import os
import json
import random

from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render
from annotations.models import Annotation


all_audio_files = [x for x in os.listdir('static') if x.lower().endswith('.ogg')]


def get_audio_spectrogram_file():
    audio_file = random.choice(all_audio_files)
    base_name = os.path.splitext(audio_file)[0]
    spectrogram_file = f'{base_name}.png'
    return audio_file, spectrogram_file


class StatisticsView(LoginRequiredMixin, TemplateView):
    template_name = 'annotation_stats.html'

    def get_context_data(self, **kwargs):
        context = super(StatisticsView, self).get_context_data(**kwargs)
        current_count = int(Annotation.objects.count())
        context['total_count'] = current_count
        context['level1'] = max(5000 - current_count, 0)
        context['level2'] = max(10000 - current_count, 0)
        return context


class CreateAnnotationView(LoginRequiredMixin, TemplateView):
    template_name = 'annotation_create.html'
    qualities = [
        ('Bright', 'Dark', 'bright_vs_dark'),
        ('Full', 'Hollow', 'full_vs_hollow'),
        ('Smooth', 'Rough', 'smooth_vs_rough'),
        ('Warm', 'Metallic', 'warm_vs_metallic'),
        ('Clear', 'Muddy', 'clear_vs_muddy'),
        ('Thin', 'Thick', 'thin_vs_thick'),
        ('Pure', 'Noisy', 'pure_vs_noisy'),
        ('Rich', 'Sparse', 'rich_vs_sparse'),
        ('Soft', 'Hard', 'soft_vs_hard')
    ]

    def get_context_data(self, **kwargs):
        context = super(CreateAnnotationView, self).get_context_data(**kwargs)
        audio_file_name, spectrogram_file_name = get_audio_spectrogram_file()
        context.update({
            'qualities': self.qualities,
            'audio_file_name': audio_file_name,
            'spectrogram_file_name': spectrogram_file_name
        })
        return context

    def post(self, request):
        result = json.loads(json.dumps(request.POST))
        was_played = result.get('was_played')
        was_played = False if was_played is None else True
        annotation_object = Annotation()
        for _, _, q in self.qualities:
            vars(annotation_object)[q] = result[q]
        annotation_object.audio_file = result.get('audio_file_name')
        annotation_object.user = self.request.user
        annotation_object.description = result.get('description')
        annotation_object.was_played = was_played
        annotation_object.save()
        # How many have you annotated?
        user_annotations = Annotation.objects.filter(user=self.request.user)
        return render(request, 'annotation_submit.html', {'count': len(user_annotations)})
