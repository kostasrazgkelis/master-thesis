from django.urls import path
from .views import (
    MatchingPipelineListCreateView, 
    MatchingPipelineDetailView,
    accept_pipeline,
    get_user_pipelines,
    get_task_status,
    test_spark
)


urlpatterns = [
    path('', MatchingPipelineListCreateView.as_view(), name='pipeline-list-create'),
    path('me/', get_user_pipelines, name='pipeline-user-list'),
    path('<uuid:pk>/', MatchingPipelineDetailView.as_view(), name='pipeline-detail'),
    path('<uuid:pipeline_id>/me/', accept_pipeline, name='pipeline-accept'),

    path('task-status/<str:task_id>/', get_task_status, name='task-status'),

    path("test-spark/", test_spark, name='test-spark'),
]

