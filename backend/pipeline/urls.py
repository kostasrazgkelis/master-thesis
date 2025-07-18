from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import (
    MatchedDataViewSet,
    MatchingPipelineDetailView,
    MatchingPipelineListCreateView,
    accept_pipeline,
    get_task_status,
    get_user_pipelines,
)

router = DefaultRouter()


urlpatterns = [
    path("", MatchingPipelineListCreateView.as_view(), name="pipeline-list-create"),
    path("me/", get_user_pipelines, name="pipeline-user-list"),
    path("<uuid:pk>/", MatchingPipelineDetailView.as_view(), name="pipeline-detail"),
    path("<uuid:pipeline_id>/me/", accept_pipeline, name="pipeline-accept"),
    # Correct nested route for matched-data create
    path(
        "<uuid:pipeline_id>/me/matched-data/",
        MatchedDataViewSet.as_view({"get": "list", "post": "create"}),
        name="matched-data-list-create",
    ),
    path(
        "<uuid:pipeline_id>/me/matched-data/<uuid:pk>/",
        MatchedDataViewSet.as_view(
            {
                "get": "retrieve",
                "put": "update",
                "patch": "partial_update",
                "delete": "destroy",
            }
        ),
        name="matched-data-detail",
    ),
    path("task-status/<str:task_id>/", get_task_status, name="task-status"),
]

urlpatterns += router.urls
