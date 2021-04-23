from django.urls import path
from annotations.views import CreateAnnotationView, StatisticsView


urlpatterns = [
    path('create/', CreateAnnotationView.as_view(), name='annotation_create'),
    path('stats/', StatisticsView.as_view(), name='annotation_stats')
]
