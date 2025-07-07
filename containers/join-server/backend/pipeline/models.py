from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class MatchingPipeline(models.Model):
    """
    Model to manage data matching pipelines between multiple users.
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('IN_PROGRESS', 'In Progress'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
        ('CANCELLED', 'Cancelled'),
    ]
    
    name = models.CharField(max_length=200, help_text="Name/description of the matching pipeline")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_pipelines')
    
    def __str__(self):
        return f"Pipeline: {self.name} ({self.status})"
    
    def all_users_accepted(self):
        """
        Check if all participating users have accepted the pipeline.
        Returns True if all users have 'ACCEPTED' status.
        """
        total_participants = self.participants.count()
        accepted_participants = self.participants.filter(status='ACCEPTED').count()
        
        return total_participants > 0 and total_participants == accepted_participants
    
    def get_participants_status(self):
        """
        Returns a list of dictionaries with user and status information.
        """
        return [
            {
                'user': participant.user,
                'status': participant.status,
                'joined_at': participant.joined_at
            }
            for participant in self.participants.all()
        ]
    
    class Meta:
        verbose_name = 'Matching Pipeline'
        verbose_name_plural = 'Matching Pipelines'


class PipelineParticipant(models.Model):
    """
    Model to track user participation and status in matching pipelines.
    """
    PARTICIPANT_STATUS_CHOICES = [
        ('INVITED', 'Invited'),
        ('ACCEPTED', 'Accepted'),
        ('DECLINED', 'Declined'),
        ('PENDING', 'Pending Response'),
    ]
    
    pipeline = models.ForeignKey(MatchingPipeline, on_delete=models.CASCADE, related_name='participants')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='pipeline_participations')
    status = models.CharField(max_length=20, choices=PARTICIPANT_STATUS_CHOICES, default='INVITED')
    joined_at = models.DateTimeField(auto_now_add=True)
    status_updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.pipeline.name} ({self.status})"
    
    class Meta:
        unique_together = ['pipeline', 'user']
        verbose_name = 'Pipeline Participant'
        verbose_name_plural = 'Pipeline Participants'
