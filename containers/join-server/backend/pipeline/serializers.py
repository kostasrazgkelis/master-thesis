from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import MatchingPipeline, PipelineParticipant

User = get_user_model()


class UserListSerializer(serializers.ModelSerializer):
    """
    Simplified serializer for user list view.
    """
    
    class Meta:
        model = User
        fields = [
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'is_active',
            'date_joined'
        ]


class PipelineParticipantSerializer(serializers.ModelSerializer):
    """
    Serializer for PipelineParticipant model.
    """
    user = UserListSerializer(read_only=True)
    
    class Meta:
        model = PipelineParticipant
        fields = [
            'id',
            'user',
            'status',
            'joined_at',
            'status_updated_at'
        ]


class MatchingPipelineSerializer(serializers.ModelSerializer):
    """
    Serializer for MatchingPipeline model.
    """
    participants = PipelineParticipantSerializer(many=True, read_only=True)
    created_by = UserListSerializer(read_only=True)
    all_users_accepted = serializers.BooleanField(read_only=True)
    participants_status = serializers.SerializerMethodField()
    
    class Meta:
        model = MatchingPipeline
        fields = [
            'id',
            'name',
            'status',
            'created_at',
            'updated_at',
            'created_by',
            'participants',
            'all_users_accepted',
            'participants_status'
        ]
        read_only_fields = ['created_at', 'updated_at', 'created_by']
    
    def get_participants_status(self, obj):
        """
        Get the participants status using the model method.
        """
        return obj.get_participants_status()


class CreatePipelineSerializer(serializers.ModelSerializer):
    """
    Serializer for creating a new pipeline.
    """
    participant_user_ids = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True,
        help_text="List of user IDs to invite to the pipeline"
    )
    
    class Meta:
        model = MatchingPipeline
        fields = ['name', 'participant_user_ids']
    
    def create(self, validated_data):
        participant_user_ids = validated_data.pop('participant_user_ids')
        pipeline = MatchingPipeline.objects.create(**validated_data)
        
        # Add participants
        for user_id in participant_user_ids:
            try:
                user = User.objects.get(id=user_id)
                PipelineParticipant.objects.create(
                    pipeline=pipeline,
                    user=user,
                    status='INVITED'
                )
            except User.DoesNotExist:
                continue
        
        return pipeline
