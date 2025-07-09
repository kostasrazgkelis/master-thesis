from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import MatchingPipeline, PipelineParty

User = get_user_model()


class UserListSerializer(serializers.ModelSerializer):
    """
    Simplified serializer for user list view.
    """
    uuid = serializers.IntegerField(source='id', read_only=True)
    
    class Meta:
        model = User
        fields = ['uuid']


class PipelinePartySerializer(serializers.ModelSerializer):
    """
    Serializer for PipelineParty model.
    """
    user_uuid = serializers.IntegerField(source='user.id', read_only=True)
    file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = PipelineParty
        fields = [
            'user_uuid',
            'file_url',
            'accepted'
        ]
    
    def get_file_url(self, obj):
        """Get file URL if file exists"""
        return obj.file.url if obj.file else None


# Keep the old name for backward compatibility in views
PipelineParticipantSerializer = PipelinePartySerializer


class MatchingPipelineSerializer(serializers.ModelSerializer):
    """
    Serializer for MatchingPipeline model.
    """
    parties = PipelinePartySerializer(many=True, read_only=True)
    created_by = UserListSerializer(read_only=True)
    all_parties_accepted = serializers.BooleanField(read_only=True)
    total_parties = serializers.IntegerField(read_only=True)
    task_status = serializers.SerializerMethodField()
    
    class Meta:
        model = MatchingPipeline
        fields = [
            'id',
            'name',
            'status',
            'match_columns',
            'parties_accepted',
            'total_parties',
            'created_at',
            'updated_at',
            'created_by',
            'parties',
            'all_parties_accepted',
            'task_status'
        ]

    
    def get_task_status(self, obj):
        """Get Celery task status information"""
        return obj.get_task_status()


class CreatePipelineSerializer(serializers.ModelSerializer):
    """
    Serializer for creating a new pipeline.
    """
    parties = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True,
        help_text="List of user IDs to invite to the pipeline",
        required=True,
        allow_empty=False
    )
    
    class Meta:
        model = MatchingPipeline
        fields = ['name', 'match_columns', 'parties']
    
    def validate_match_columns(self, value):
        """Validate match_columns field"""
        if not isinstance(value, list):
            raise serializers.ValidationError("match_columns must be a list")
        
        if not value:  # Empty list
            raise serializers.ValidationError("match_columns cannot be empty")
        
        # Check if all items are strings
        for item in value:
            if not isinstance(item, str):
                raise serializers.ValidationError("All items in match_columns must be strings")
        
        return value
    
    def validate_parties(self, value):
        """Validate parties field"""
        if not isinstance(value, list):
            raise serializers.ValidationError("parties must be a list")
        
        if not value:  # Empty list
            raise serializers.ValidationError("parties cannot be empty")
        
        # Check if all items are integers (user IDs)
        for item in value:
            if not isinstance(item, int):
                raise serializers.ValidationError("All items in parties must be integers (user IDs)")
        
        # Check for duplicates
        if len(value) != len(set(value)):
            raise serializers.ValidationError("parties cannot contain duplicate user IDs")
        
        # Validate that all users exist
        existing_users = User.objects.filter(id__in=value).values_list('id', flat=True)
        missing_users = set(value) - set(existing_users)
        if missing_users:
            raise serializers.ValidationError(f"Users with IDs {list(missing_users)} do not exist")
        
        return value
    
    def create(self, validated_data):
        parties_user_ids = validated_data.pop('parties')
        pipeline = MatchingPipeline.objects.create(**validated_data)
        
        # Get the creator from the context (passed from the view)
        creator = self.context['request'].user
        
        # Add the creator as a party first (if not already in the list)
        if creator.id not in parties_user_ids:
            parties_user_ids.insert(0, creator.id)  # Add creator at the beginning
        
        # Add all parties to the pipeline
        for user_id in parties_user_ids:
            user = User.objects.get(id=user_id)
            PipelineParty.objects.create(
                pipeline=pipeline,
                user=user,
                accepted=False  # All parties need to explicitly accept and upload file
            )
        
        return pipeline
    
    def to_representation(self, instance):
        """Return full pipeline data using MatchingPipelineSerializer"""
        return MatchingPipelineSerializer(instance, context=self.context).data


class AcceptPipelineSerializer(serializers.Serializer):
    """
    Serializer for accepting a pipeline and uploading a file.
    """
    file = serializers.FileField(
        help_text="Data file to upload for the pipeline (CSV, JSON, XLSX, Parquet, TXT - max 50MB)",
        required=True,
        allow_empty_file=False,
        style={'base_template': 'file_upload.html'}  # This helps with Swagger UI
    )
    
    class Meta:
        # This helps drf-spectacular understand it's a file upload
        swagger_schema_fields = {
            'type': 'object',
            'properties': {
                'file': {
                    'type': 'string',
                    'format': 'binary',
                    'description': 'Data file to upload'
                }
            }
        }
    
    def validate_file(self, value):
        """Validate uploaded file"""
        # Check file size (limit to 50MB)
        if value.size > 50 * 1024 * 1024:
            raise serializers.ValidationError("File size cannot exceed 50MB")
        
        # Check file extension (allow common data file formats)
        allowed_extensions = ['.csv', '.json', '.xlsx', '.parquet', '.txt']
        file_extension = value.name.lower().split('.')[-1] if '.' in value.name else ''
        if f'.{file_extension}' not in allowed_extensions:
            raise serializers.ValidationError(
                f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        return value
    
    def update(self, instance, validated_data):
        """Accept the pipeline and upload file"""
        file = validated_data.get('file')
        instance.accept_and_upload(file)
        return instance
