from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import (
    MatchingPipelineListCreateView,
    MatchingPipelineDetailView,
    MatchedDataViewSet,
    accept_pipeline,
    get_user_pipelines,
    get_task_status,
)

router = DefaultRouter()
matched_data_create = MatchedDataViewSet.as_view({"post": "create"})

urlpatterns = [
    path("", MatchingPipelineListCreateView.as_view(), name="pipeline-list-create"),
    path("me/", get_user_pipelines, name="pipeline-user-list"),
    path("<uuid:pk>/", MatchingPipelineDetailView.as_view(), name="pipeline-detail"),
    path("<uuid:pipeline_id>/me/", accept_pipeline, name="pipeline-accept"),
    # Correct nested route for matched-data create
    path(
        "<uuid:pipeline_id>/me/matched-data/",
        matched_data_create,
        name="matched-data-create",
    ),
    path("task-status/<str:task_id>/", get_task_status, name="task-status"),
]

urlpatterns += router.urls
