import logging
from typing import override
from webbrowser import get
from rest_framework import generics, permissions, status, viewsets, serializers
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django.db import models
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.openapi import OpenApiTypes
from rest_framework.exceptions import NotFound
import shutil

from .serializers import (
    MatchedDataSerializer,
    MatchingPipelineSerializer,
    CreatePipelineSerializer,
    AcceptPipelineSerializer,
)
from .models import MatchedData, MatchingPipeline, PipelineParty
from .tasks import get_matched_data, multi_party_matching_pipeline


logger = logging.getLogger(__name__)
User = get_user_model()


class MatchingPipelineListCreateView(generics.ListCreateAPIView):
    """
    API view to list all pipelines or create a new pipeline.
    """

    serializer_class = MatchingPipelineSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Return pipelines where user is either creator or party
        user = self.request.user
        return MatchingPipeline.objects.filter(
            models.Q(created_by=user) | models.Q(parties__user=user)
        ).distinct()

    def get_serializer_class(self):
        if self.request.method == "POST":
            return CreatePipelineSerializer
        return MatchingPipelineSerializer

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)

    @extend_schema(
        summary="List pipelines or create new pipeline",
        description="Get all pipelines where user is involved or create a new matching pipeline",
        responses={200: MatchingPipelineSerializer(many=True)},
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    @extend_schema(
        summary="Create new pipeline",
        description="Create a new matching pipeline and invite users",
        request=CreatePipelineSerializer,
        responses={201: MatchingPipelineSerializer()},
    )
    def post(self, request, *args, **kwargs):
        return super().post(request, *args, **kwargs)


class MatchingPipelineDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    API view to retrieve, update or delete a specific pipeline.
    """

    serializer_class = MatchingPipelineSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # Only allow access to pipelines where user is creator or party
        user = self.request.user
        return MatchingPipeline.objects.filter(
            models.Q(created_by=user) | models.Q(parties__user=user)
        ).distinct()


@extend_schema(
    summary="Accept pipeline and upload file",
    description="Accept participation in a pipeline and upload data file. Supported formats: CSV, JSON, XLSX, Parquet, TXT (max 50MB)",
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "format": "binary",
                    "description": "Data file to upload for the pipeline",
                }
            },
            "required": ["file"],
        }
    },
    responses={
        200: MatchingPipelineSerializer(),
        400: {
            "type": "object",
            "properties": {
                "error": {"type": "string"},
                "file": {"type": "array", "items": {"type": "string"}},
            },
            "example": {"file": ["File size cannot exceed 50MB"]},
        },
        404: {
            "type": "object",
            "properties": {"error": {"type": "string"}},
            "example": {"error": "Pipeline or party participation not found"},
        },
    },
    parameters=[
        OpenApiParameter(
            name="pipeline_id",
            type=OpenApiTypes.UUID,
            location=OpenApiParameter.PATH,
            description="UUID of the pipeline to accept",
        )
    ],
)
@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def accept_pipeline(request, pipeline_id):
    """
    API view to accept pipeline and upload file - matches your PUT pipeline/{id}/me/ endpoint.
    Handles multipart/form-data for file upload.
    """

    try:
        pipeline = MatchingPipeline.objects.get(id=pipeline_id)
        party = PipelineParty.objects.get(pipeline=pipeline, user=request.user)
    except MatchingPipeline.DoesNotExist:
        return Response(
            {"error": "Pipeline not found"}, status=status.HTTP_404_NOT_FOUND
        )
    except PipelineParty.DoesNotExist:
        return Response(
            {"error": "You are not a party in this pipeline"},
            status=status.HTTP_404_NOT_FOUND,
        )

    # Check if party already accepted
    if party.accepted:
        return Response(
            {"error": f"User {party.user.username} has already accepted this pipeline"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    # Validate and process file upload
    serializer = AcceptPipelineSerializer(instance=party, data=request.data)
    if serializer.is_valid():
        # This will call party.accept_and_upload(file) which:
        # 1. Saves the uploaded file to the party.file field
        # 2. Sets party.accepted = True
        # 3. Updates pipeline.parties_accepted count
        # 4. Updates pipeline status if all parties accepted
        serializer.save()

        # Refresh pipeline from database to get updated data
        pipeline.refresh_from_db()

        # Check if all parties have accepted and trigger processing
        if pipeline.all_parties_accepted and pipeline.status == "READY":
            task = multi_party_matching_pipeline.delay(str(pipeline.id))
            pipeline.celery_task_id = task.id if task else None
            pipeline.save()
            pipeline.refresh_from_db()

        pipeline_serializer = MatchingPipelineSerializer(pipeline)
        return Response(pipeline_serializer.data)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@extend_schema(
    summary="Get user's pipelines",
    description="Get all pipelines where user is a party - matches your GET /pipelines/me/ endpoint",
    responses={200: MatchingPipelineSerializer(many=True)},
)
@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def get_user_pipelines(request):
    """
    API view to get pipelines for current user - matches your GET /pipelines/me/ endpoint.
    """
    user = request.user
    pipelines = MatchingPipeline.objects.filter(parties__user=user).distinct()

    serializer = MatchingPipelineSerializer(pipelines, many=True)
    return Response({"pipelines": serializer.data})


@extend_schema(
    summary="Get Celery task status",
    description="Get the status and result of any Celery task by task ID",
    parameters=[
        OpenApiParameter(
            name="task_id",
            description="Celery task ID",
            required=True,
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
        )
    ],
    responses={
        200: {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "status": {"type": "string"},
                "result": {"type": "object"},
                "progress": {"type": "object"},
                "error": {"type": "string"},
            },
        }
    },
)
@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def get_task_status(request, task_id):
    """
    Get the status and result of any Celery task by task ID.
    Useful for checking the status of test_celery_connection tasks.
    """
    try:
        from celery.result import AsyncResult

        task_result = AsyncResult(task_id)

        response_data = {
            "task_id": task_id,
            "status": task_result.status,
        }

        if task_result.state == "PENDING":
            response_data["message"] = "Task is waiting to be processed"
            response_data["result"] = None
        elif task_result.state == "PROGRESS":
            response_data["message"] = "Task is in progress"
            response_data["progress"] = task_result.info
            response_data["result"] = None
        elif task_result.state == "SUCCESS":
            response_data["message"] = "Task completed successfully"
            response_data["result"] = task_result.result
            # If it's our test task, highlight the UUID
            if isinstance(task_result.result, dict) and "test_id" in task_result.result:
                response_data["test_uuid"] = task_result.result["test_id"]
        elif task_result.state == "FAILURE":
            response_data["message"] = "Task failed"
            response_data["error"] = str(task_result.info)
            response_data["result"] = None
        else:
            response_data["message"] = f"Task status: {task_result.state}"
            response_data["result"] = (
                task_result.result if task_result.ready() else None
            )

        return Response(response_data)

    except Exception as e:
        return Response(
            {"error": f"Failed to get task status: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class MatchedDataViewSet(viewsets.ModelViewSet):
    queryset = MatchedData.objects.all()
    serializer_class = MatchedDataSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        pipeline_id = self.kwargs.get("pipeline_id")
        right_parties = set(self.request.data.get("right_parties"))
        active_user = self.request.user

        if active_user.id in right_parties:
            raise serializers.ValidationError("You cannot match with yourself.")

        try:
            pipeline = MatchingPipeline.objects.get(id=pipeline_id)
        except MatchingPipeline.DoesNotExist:
            raise NotFound("Pipeline not found.")

        if not right_parties.issubset(pipeline.get_all_parties_id()):
            raise serializers.ValidationError(
                "One or more right_parties are not part of the pipeline."
            )

        existing_matches = MatchedData.objects.filter(
            pipeline=pipeline, left_party=active_user
        )

        submitted_set = right_parties
        for match in existing_matches:
            existing_set = set(match.right_parties.values_list("id", flat=True))
            if submitted_set == existing_set:
                raise serializers.ValidationError(
                    "Matched data already exists for this pipeline and parties."
                )

        matched_data = serializer.save(
            left_party=active_user, right_parties=right_parties, pipeline=pipeline
        )

        get_matched_data.delay(
            pipeline_id=matched_data.uuid,
        )

    def list(self, request, *args, **kwargs):
        """Override list method to add custom logic"""
        queryset = self.filter_queryset(
            self.get_queryset().filter(left_party=request.user)
        )

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, *args, **kwargs):
        """
        GET /pipelines/<pipeline_id>/me/matched-data/<matched_data_id>/
        Retrieve a specific matched data record
        """
        try:
            instance = self.get_object()

            serializer = self.get_serializer(instance)
            return Response(serializer.data)

        except MatchedData.DoesNotExist:
            raise NotFound("Matched data not found.")
        except Exception as e:
            logger.error(f"Error retrieving matched data: {str(e)}")
            return Response(
                {"error": "Failed to retrieve matched data"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def destroy(self, request, *args, **kwargs):
        """
        GET /pipelines/<pipeline_id>/me/matched-data/<matched_data_id>/
        Retrieve a specific matched data record
        """
        try:
            instance = self.get_object()
            shutil.rmtree(instance.folder_path, ignore_errors=True)
            instance.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except MatchedData.DoesNotExist:
            raise NotFound("Matched data not found.")
        except Exception as e:
            logger.error(f"Error deleting matched data: {str(e)}")
            return Response(
                {"error": "Failed to delete matched data"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
