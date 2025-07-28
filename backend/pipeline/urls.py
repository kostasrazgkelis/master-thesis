from django.urls import path
from rest_framework.routers import DefaultRouter

from .views import (
    MatchedDataViewSet,
    MatchingPipelineDetailView,
    MatchingPipelineListCreateView,
    accept_pipeline,
    get_user_pipelines,
)

router = DefaultRouter()


urlpatterns = [
    path("", MatchingPipelineListCreateView.as_view(), name="pipeline-list-create"),
    path("me/", get_user_pipelines, name="pipeline-user-list"),
    path("<uuid:pk>/", MatchingPipelineDetailView.as_view(), name="pipeline-detail"),
    path("<uuid:pipeline_id>/me/", accept_pipeline, name="pipeline-accept"),
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
]

urlpatterns += router.urls
