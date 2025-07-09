from rest_framework import generics, permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django.db import models
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.openapi import OpenApiTypes
from .serializers import (
    MatchingPipelineSerializer, CreatePipelineSerializer, AcceptPipelineSerializer
)
from .models import MatchingPipeline, PipelineParty

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
            models.Q(created_by=user) | 
            models.Q(parties__user=user)
        ).distinct()
    
    def get_serializer_class(self):
        if self.request.method == 'POST':
            return CreatePipelineSerializer
        return MatchingPipelineSerializer
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @extend_schema(
        summary="List pipelines or create new pipeline",
        description="Get all pipelines where user is involved or create a new matching pipeline",
        responses={200: MatchingPipelineSerializer(many=True)}
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)
    
    @extend_schema(
        summary="Create new pipeline",
        description="Create a new matching pipeline and invite users",
        request=CreatePipelineSerializer,
        responses={201: MatchingPipelineSerializer()}
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
            models.Q(created_by=user) | 
            models.Q(parties__user=user)
        ).distinct()


@extend_schema(
    summary="Accept pipeline and upload file",
    description="Accept participation in a pipeline and upload data file. Supported formats: CSV, JSON, XLSX, Parquet, TXT (max 50MB)",
    request={
        'multipart/form-data': {
            'type': 'object',
            'properties': {
                'file': {
                    'type': 'string',
                    'format': 'binary',
                    'description': 'Data file to upload for the pipeline'
                }
            },
            'required': ['file']
        }
    },
    responses={
        200: MatchingPipelineSerializer(),
        400: {
            'type': 'object',
            'properties': {
                'error': {'type': 'string'},
                'file': {'type': 'array', 'items': {'type': 'string'}}
            },
            'example': {
                'file': ['File size cannot exceed 50MB']
            }
        },
        404: {
            'type': 'object',
            'properties': {
                'error': {'type': 'string'}
            },
            'example': {
                'error': 'Pipeline or party participation not found'
            }
        }
    },
    parameters=[
        OpenApiParameter(
            name='pipeline_id',
            type=OpenApiTypes.UUID,
            location=OpenApiParameter.PATH,
            description='UUID of the pipeline to accept'
        )
    ]
)
@api_view(['PUT'])
@permission_classes([permissions.IsAuthenticated])
def accept_pipeline(request, pipeline_id):
    """
    API view to accept pipeline and upload file - matches your PUT pipeline/{id}/me/ endpoint.
    Handles multipart/form-data for file upload.
    """
    
    try:
        pipeline = MatchingPipeline.objects.get(id=pipeline_id)
        party = PipelineParty.objects.get(
            pipeline=pipeline,
            user=request.user
        )
    except MatchingPipeline.DoesNotExist:
        return Response(
            {'error': 'Pipeline not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except PipelineParty.DoesNotExist:
        return Response(
            {'error': 'You are not a party in this pipeline'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Check if party already accepted
    if party.accepted:
        return Response(
            {'error': f'User {party.user.username} has already accepted this pipeline'}, 
            status=status.HTTP_400_BAD_REQUEST
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
        
        # Return updated pipeline data
        pipeline_serializer = MatchingPipelineSerializer(pipeline)
        return Response(pipeline_serializer.data)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@extend_schema(
    summary="Get user's pipelines",
    description="Get all pipelines where user is a party - matches your GET /pipelines/me/ endpoint",
    responses={200: MatchingPipelineSerializer(many=True)}
)
@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def get_user_pipelines(request):
    """
    API view to get pipelines for current user - matches your GET /pipelines/me/ endpoint.
    """
    user = request.user
    pipelines = MatchingPipeline.objects.filter(
        parties__user=user
    ).distinct()
    
    serializer = MatchingPipelineSerializer(pipelines, many=True)
    return Response({'pipelines': serializer.data})


@extend_schema(
    summary="Trigger pipeline processing manually",
    description="""
    Manually trigger pipeline processing. Pipeline must be in READY state and all parties must have accepted.
    
    **Query Parameters:**
    - `force=true` - Force trigger even if there's an existing task running
    
    **Normal Flow:**
    1. Upload files for all parties → Status becomes READY → Auto-triggers processing
    2. If already READY → Can manually trigger with this endpoint
    """,
    parameters=[
        OpenApiParameter(
            name='pipeline_id',
            description='Pipeline UUID',
            required=True,
            type=OpenApiTypes.UUID,
            location=OpenApiParameter.PATH
        ),
        OpenApiParameter(
            name='force',
            description='Force trigger even if task already exists (true/false)',
            required=False,
            type=OpenApiTypes.BOOL,
            location=OpenApiParameter.QUERY
        )
    ],
    responses={
        200: {
            'type': 'object',
            'properties': {
                'message': {'type': 'string'},
                'task_id': {'type': 'string'},
                'pipeline_id': {'type': 'string'},
                'status': {'type': 'string'},
                'force_triggered': {'type': 'boolean'}
            }
        },
        400: {
            'type': 'object',
            'properties': {
                'error': {'type': 'string'}
            },
            'examples': {
                'not_ready': {'error': 'Pipeline must be in READY state'},
                'parties_not_accepted': {'error': 'Not all parties have accepted (2/3 accepted)'}
            }
        }
    }
)
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def trigger_pipeline_processing(request, pipeline_id):
    """
    Manually trigger pipeline processing (mainly for testing).
    In normal operation, this is triggered automatically when status changes to READY.
    """
    try:
        pipeline = MatchingPipeline.objects.get(id=pipeline_id)
    except MatchingPipeline.DoesNotExist:
        return Response(
            {'error': 'Pipeline not found'},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Check if user has access to this pipeline
    user = request.user
    if not (pipeline.parties.filter(user=user).exists()):
        return Response(
            {'error': 'Access denied'},
            status=status.HTTP_403_FORBIDDEN
        )
    
    # Check if pipeline is in the right state
    if pipeline.status != 'READY':
        return Response(
            {'error': f'Pipeline must be in READY state to trigger processing, current state: {pipeline.status}'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Check if all parties have accepted
    if not pipeline.all_parties_accepted:
        return Response(
            {'error': f'Cannot trigger processing - not all parties have accepted and uploaded files ({pipeline.parties_accepted}/{pipeline.total_parties} accepted)'},
            status=status.HTTP_400_BAD_REQUEST
        )
        
    # Trigger processing
    try:
        pipeline.trigger_processing()
        
        # Refresh pipeline data
        pipeline.refresh_from_db()
        
        return Response({
            'message': 'Pipeline processing triggered successfully',
            'task_id': pipeline.celery_task_id,
            'pipeline_id': str(pipeline.id),
            'status': pipeline.status
            })
    except Exception as e:
        return Response(
            {'error': f'Failed to trigger processing: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Test Celery connection",
    description="Test that Celery is working by running a simple task with UUID tracking",
    responses={
        200: {
            'type': 'object',
            'properties': {
                'message': {'type': 'string'},
                'task_id': {'type': 'string'},
                'status': {'type': 'string'},
                'check_result_url': {'type': 'string'}
            }
        }
    }
)
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def test_celery(request):
    """
    Test Celery connection with a simple task that includes UUID tracking.
    """
    try:
        from .tasks import test_celery_connection
        
        task = test_celery_connection.delay()
        
        return Response({
            'message': 'Celery test task started successfully',
            'task_id': task.id,
            'status': 'Task queued and will include unique UUID',
            'check_result_url': f'/api/pipelines/task-status/{task.id}/',
            'info': 'The task will generate a unique UUID and process for 5 seconds'
        })
    except Exception as e:
        return Response(
            {'error': f'Failed to start Celery task: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@extend_schema(
    summary="Get Celery task status",
    description="Get the status and result of any Celery task by task ID",
    parameters=[
        OpenApiParameter(
            name='task_id',
            description='Celery task ID',
            required=True,
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH
        )
    ],
    responses={
        200: {
            'type': 'object',
            'properties': {
                'task_id': {'type': 'string'},
                'status': {'type': 'string'},
                'result': {'type': 'object'},
                'progress': {'type': 'object'},
                'error': {'type': 'string'}
            }
        }
    }
)
@api_view(['GET'])
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
            'task_id': task_id,
            'status': task_result.status,
        }
        
        if task_result.state == 'PENDING':
            response_data['message'] = 'Task is waiting to be processed'
            response_data['result'] = None
        elif task_result.state == 'PROGRESS':
            response_data['message'] = 'Task is in progress'
            response_data['progress'] = task_result.info
            response_data['result'] = None
        elif task_result.state == 'SUCCESS':
            response_data['message'] = 'Task completed successfully'
            response_data['result'] = task_result.result
            # If it's our test task, highlight the UUID
            if isinstance(task_result.result, dict) and 'test_id' in task_result.result:
                response_data['test_uuid'] = task_result.result['test_id']
        elif task_result.state == 'FAILURE':
            response_data['message'] = 'Task failed'
            response_data['error'] = str(task_result.info)
            response_data['result'] = None
        else:
            response_data['message'] = f'Task status: {task_result.state}'
            response_data['result'] = task_result.result if task_result.ready() else None
        
        return Response(response_data)
        
    except Exception as e:
        return Response(
            {'error': f'Failed to get task status: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
