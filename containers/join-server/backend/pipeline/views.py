from rest_framework import generics, permissions, status, serializers
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django.db import models
from drf_spectacular.utils import extend_schema
from .serializers import (
    MatchingPipelineSerializer, PipelineParticipantSerializer, CreatePipelineSerializer
)
from .models import MatchingPipeline, PipelineParticipant

User = get_user_model()


class MatchingPipelineListCreateView(generics.ListCreateAPIView):
    """
    API view to list all pipelines or create a new pipeline.
    """
    serializer_class = MatchingPipelineSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        # Return pipelines where user is either creator or participant
        user = self.request.user
        return MatchingPipeline.objects.filter(
            models.Q(created_by=user) | 
            models.Q(participants__user=user)
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
        # Only allow access to pipelines where user is creator or participant
        user = self.request.user
        return MatchingPipeline.objects.filter(
            models.Q(created_by=user) | 
            models.Q(participants__user=user)
        ).distinct()


@extend_schema(
    summary="Update participant status",
    description="Accept or decline participation in a pipeline",
    request=serializers.Serializer,
    responses={200: PipelineParticipantSerializer()}
)
@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def update_participant_status(request, pipeline_id):
    """
    API view to update participant status (accept/decline).
    """
    try:
        pipeline = MatchingPipeline.objects.get(id=pipeline_id)
        participant = PipelineParticipant.objects.get(
            pipeline=pipeline,
            user=request.user
        )
    except (MatchingPipeline.DoesNotExist, PipelineParticipant.DoesNotExist):
        return Response(
            {'error': 'Pipeline or participation not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
    
    new_status = request.data.get('status')
    if new_status not in ['ACCEPTED', 'DECLINED']:
        return Response(
            {'error': 'Status must be ACCEPTED or DECLINED'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    participant.status = new_status
    participant.save()
    
    # Check if all users have accepted and update pipeline status
    if pipeline.all_users_accepted() and pipeline.status == 'PENDING':
        pipeline.status = 'IN_PROGRESS'
        pipeline.save()
    
    serializer = PipelineParticipantSerializer(participant)
    return Response(serializer.data)
