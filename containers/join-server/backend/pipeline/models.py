import uuid
from django.db import models
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.utils import timezone

User = get_user_model()


def upload_to_pipeline_files(instance, filename):
    """Generate upload path for pipeline files"""
    return f'pipelines/{instance.pipeline.id}/participants/{instance.user.id}/{filename}'


class MatchingPipeline(models.Model):
    """
    Model to manage data matching pipelines between multiple users.
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending - Waiting for parties to upload data'),
        ('READY', 'Ready - All parties uploaded, waiting to start'),
        ('RUNNING', 'Running - Pipeline is executing'),
        ('COMPLETED', 'Completed - Pipeline finished successfully'),
        ('FAILED', 'Failed - Pipeline execution failed'),
        ('CANCELLED', 'Cancelled - Pipeline was cancelled'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, help_text="Name/description of the matching pipeline")
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_pipelines')
    match_columns = models.JSONField(
        default=list, 
        help_text="List of column names to match on, e.g., ['col0', 'col1', 'col2']"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    parties_accepted = models.IntegerField(default=0, help_text="Number of parties that have accepted")
    execution_started_at = models.DateTimeField(null=True, blank=True, help_text="When pipeline execution started")
    execution_completed_at = models.DateTimeField(null=True, blank=True, help_text="When pipeline execution completed")
    error_message = models.TextField(null=True, blank=True, help_text="Error message if pipeline failed")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Pipeline: {self.name} ({self.status})"
    
    @property
    def total_parties(self):
        """Get total number of parties in this pipeline"""
        return self.parties.count()
    
    @property
    def all_parties_accepted(self):
        """Check if all parties have accepted and uploaded files"""
        return self.parties_accepted == self.total_parties and self.total_parties >= 2
    
    @property
    def is_ready_to_start(self):
        """Check if pipeline is ready to start (all parties uploaded data)"""
        return self.all_parties_accepted and self.status == 'READY'
    
    @property
    def can_transition_to_ready(self):
        """Check if pipeline can transition from PENDING to READY"""
        return self.all_parties_accepted and self.status == 'PENDING'
    
    def get_parties_status(self):
        """
        Returns a list of dictionaries with party information.
        """
        return [
            {
                'user_id': party.user.id,
                'file': party.file.url if party.file else None,
                'accepted': party.accepted,
            }
            for party in self.parties.all()
        ]
    
    def update_status(self):
        """Update pipeline status based on parties acceptance"""
        if self.status == 'PENDING' and self.can_transition_to_ready:
            self.status = 'READY'
            print(f"Pipeline {self.name} is now READY - all {self.total_parties} parties have uploaded data")
        elif self.status not in ['RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED']:
            # Only update if not in a final or running state
            if self.parties_accepted == 0:
                self.status = 'PENDING'
            elif self.parties_accepted > 0 and not self.all_parties_accepted:
                self.status = 'PENDING'
        
        self.save()
    
    def start_execution(self):
        """Start pipeline execution - transition from READY to RUNNING"""
        if self.status != 'READY':
            raise ValidationError(f"Cannot start pipeline - current status is {self.status}, must be READY")
        
        if not self.all_parties_accepted:
            raise ValidationError("Cannot start pipeline - not all parties have uploaded data")
        
        self.status = 'RUNNING'
        self.execution_started_at = timezone.now()
        self.save()
        print(f"Pipeline {self.name} execution started - status changed to RUNNING")
        
        # Here you would trigger the actual pipeline execution
        # For now, we'll just mark it as started
        return True
    
    def mark_completed(self):
        """Mark pipeline as completed"""
        if self.status != 'RUNNING':
            raise ValidationError(f"Cannot complete pipeline - current status is {self.status}, must be RUNNING")
        
        self.status = 'COMPLETED'
        self.execution_completed_at = timezone.now()
        self.save()
        print(f"Pipeline {self.name} completed successfully")
    
    def mark_failed(self, error_message=None):
        """Mark pipeline as failed"""
        if self.status not in ['RUNNING', 'READY']:
            raise ValidationError(f"Cannot fail pipeline - current status is {self.status}")
        
        self.status = 'FAILED'
        self.execution_completed_at = timezone.now()
        if error_message:
            self.error_message = error_message
        self.save()
        print(f"Pipeline {self.name} failed" + (f": {error_message}" if error_message else ""))
    
    def cancel(self):
        """Cancel the pipeline"""
        if self.status in ['COMPLETED', 'FAILED']:
            raise ValidationError(f"Cannot cancel pipeline - current status is {self.status}")
        
        self.status = 'CANCELLED'
        self.save()
        print(f"Pipeline {self.name} was cancelled")
    
    def clean(self):
        """Validate pipeline data"""
        if not isinstance(self.match_columns, list):
            raise ValidationError("match_columns must be a list")
        
        if self.parties_accepted > self.total_parties:
            raise ValidationError("parties_accepted cannot exceed total parties")
    
    class Meta:
        verbose_name = 'Matching Pipeline'
        verbose_name_plural = 'Matching Pipelines'


class PipelineParty(models.Model):
    """
    Model to track parties (users) participation in matching pipelines.
    Each party can upload a file and accept/decline participation.
    """
    pipeline = models.ForeignKey(
        MatchingPipeline, 
        on_delete=models.CASCADE, 
        related_name='parties'
    )
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='pipeline_parties'
    )
    file = models.FileField(
        upload_to=upload_to_pipeline_files,
        null=True,
        blank=True,
        help_text="Data file uploaded by this party"
    )
    accepted = models.BooleanField(
        default=False,
        help_text="Whether this party has accepted the pipeline"
    )
    joined_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.pipeline.name} (Accepted: {self.accepted})"
    
    def accept_and_upload(self, file):
        """Accept the pipeline and upload file"""
        self.file = file
        self.accepted = True
        self.save()
        
        # Update pipeline parties_accepted count
        self.pipeline.parties_accepted = self.pipeline.parties.filter(accepted=True).count()
        self.pipeline.update_status()
    
    def decline(self):
        """Decline participation in the pipeline"""
        self.accepted = False
        self.file = None
        self.save()
        
        # Update pipeline parties_accepted count
        self.pipeline.parties_accepted = self.pipeline.parties.filter(accepted=True).count()
        self.pipeline.update_status()
    
    class Meta:
        unique_together = ['pipeline', 'user']
        verbose_name = 'Pipeline Party'
        verbose_name_plural = 'Pipeline Parties'
