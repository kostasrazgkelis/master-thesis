from django.urls import path
from .views import (
    MatchingPipelineListCreateView, 
    MatchingPipelineDetailView,
    accept_pipeline,
    get_user_pipelines,
    trigger_pipeline_processing,
    test_celery,
    get_task_status
)

urlpatterns = [
    path('', MatchingPipelineListCreateView.as_view(), name='pipeline-list-create'),
    path('me/', get_user_pipelines, name='pipeline-user-list'),
    path('<uuid:pk>/', MatchingPipelineDetailView.as_view(), name='pipeline-detail'),
    path('<uuid:pipeline_id>/me/', accept_pipeline, name='pipeline-accept'),
    
    path('<uuid:pipeline_id>/trigger/', trigger_pipeline_processing, name='pipeline-trigger'),
    path('test-celery/', test_celery, name='test-celery'),
    path('task-status/<str:task_id>/', get_task_status, name='task-status'),
]
