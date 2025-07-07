from django.urls import path
from .views import (
    MatchingPipelineListCreateView, MatchingPipelineDetailView, 
    update_participant_status
)

urlpatterns = [
    path('', MatchingPipelineListCreateView.as_view(), name='pipeline-list-create'),
    path('<int:pk>/', MatchingPipelineDetailView.as_view(), name='pipeline-detail'),
    path('<int:pipeline_id>/status/', update_participant_status, name='update-participant-status'),
]
