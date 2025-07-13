import uuid
from django.db import models
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.utils import timezone
import logging

User = get_user_model()
logger = logging.getLogger(__name__)

STATUS_CHOICES = [
    ("PENDING", "Pending - Waiting for parties to upload data"),
    ("READY", "Ready - All parties uploaded, waiting to start"),
    ("RUNNING", "Running - Pipeline is executing"),
    ("COMPLETED", "Completed - Pipeline finished successfully"),
    ("FAILED", "Failed - Pipeline execution failed"),
    ("CANCELLED", "Cancelled - Pipeline was cancelled"),
]


def upload_to_pipeline_files(instance, filename):
    """Generate upload path for pipeline files"""
    return (
        f"pipelines/{instance.pipeline.id}/participants/{instance.user.id}/{filename}"
    )


class MatchingPipeline(models.Model):
    """
    Model to manage data matching pipelines between multiple users.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(
        max_length=200, help_text="Name/description of the matching pipeline"
    )
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="created_pipelines"
    )
    match_columns = models.JSONField(
        default=list,
        help_text="List of column names to match on, e.g., ['col0', 'col1', 'col2']",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="PENDING")
    parties_accepted = models.IntegerField(
        default=0, help_text="Number of parties that have accepted"
    )
    execution_started_at = models.DateTimeField(
        null=True, blank=True, help_text="When pipeline execution started"
    )
    execution_completed_at = models.DateTimeField(
        null=True, blank=True, help_text="When pipeline execution completed"
    )
    error_message = models.TextField(
        null=True, blank=True, help_text="Error message if pipeline failed"
    )
    result_data = models.JSONField(
        null=True, blank=True, help_text="JSON result data from pipeline execution"
    )
    celery_task_id = models.CharField(
        max_length=255, null=True, blank=True, help_text="Celery task ID for tracking"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    output_struct = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Directory path containing the Parquet files",
    )

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
        return self.status == "READY"

    @property
    def can_transition_to_ready(self):
        """Check if pipeline can transition from PENDING to READY"""
        return self.all_parties_accepted and self.status == "PENDING"

    def get_parties_status(self):
        """
        Returns a list of dictionaries with party information.
        """
        return [
            {
                "user_id": party.user.id,
                "file": party.file.url if party.file else None,
                "accepted": party.accepted,
            }
            for party in self.parties.all()
        ]

    def get_all_parties_id(self):
        return (party.user.id for party in self.parties.all())

    def update_status(self):
        """Update pipeline status based on parties acceptance"""

        if self.can_transition_to_ready:
            self.status = "READY"
            self.save()

            self.trigger_processing()
            return

    def trigger_processing(self):

        try:
            # Import here to avoid circular imports
            from .tasks import multi_party_matching_pipeline

            # Trigger the Celery task
            task = multi_party_matching_pipeline.delay(str(self.id))
            self.celery_task_id = task.id
            self.save()
        except Exception as e:
            error_msg = f"Failed to trigger Celery task: {str(e)}"
            logger.error(error_msg)
            self.mark_failed(error_msg)

    def mark_completed(self, result_data=None):
        """Mark pipeline as completed"""
        self.status = "COMPLETED"
        self.execution_completed_at = timezone.now()
        if result_data:
            self.result_data = result_data
        self.save()
        logger.info(f"Pipeline {self.name} completed successfully")

    def mark_failed(self, error_message=None):
        """Mark pipeline as failed"""
        self.status = "FAILED"
        self.execution_completed_at = timezone.now()
        if error_message:
            self.error_message = error_message
        self.save()
        logger.warning(
            f"Pipeline {self.name} failed"
            + (f": {error_message}" if error_message else "")
        )

    def mark_cancel(self):
        """Cancel the pipeline"""
        if self.status in ["COMPLETED", "FAILED"]:
            raise ValidationError(
                f"Cannot cancel pipeline - current status is {self.status}"
            )

        self.status = "CANCELLED"
        self.save()
        logger.info(f"Pipeline {self.name} was cancelled")

    def get_task_status(self):
        """Get the status of the associated Celery task"""
        if not self.celery_task_id:
            return None

        try:
            from celery.result import AsyncResult

            task_result = AsyncResult(self.celery_task_id)
            return {
                "task_id": self.celery_task_id,
                "status": task_result.status,
                "result": task_result.result if task_result.ready() else None,
                "traceback": task_result.traceback if task_result.failed() else None,
            }
        except Exception as e:
            logger.error(f"Error getting task status: {str(e)}")
            return None

    def clean(self):
        """Validate pipeline data"""
        if not isinstance(self.match_columns, list):
            raise ValidationError("match_columns must be a list")

        if self.parties_accepted > self.total_parties:
            raise ValidationError("parties_accepted cannot exceed total parties")

    class Meta:
        verbose_name = "Matching Pipeline"
        verbose_name_plural = "Matching Pipelines"


class MatchedData(models.Model):
    uuid = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    left_party = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="left_party_matches",
        help_text="User ID of the right party in the match",
    )
    right_parties = models.ManyToManyField(
        User,
        related_name="right_party_matches",
        help_text="Users on the right party in the match",
    )
    pipeline = models.ForeignKey(
        MatchingPipeline,
        on_delete=models.CASCADE,
        related_name="matched_data",
        blank=True,
        null=False,
    )
    folder_path = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Path to the folder containing matched data",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="PENDING")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.uuid, self.left_party, self.right_parties


class PipelineParty(models.Model):
    """
    Model to track parties (users) participation in matching pipelines.
    Each party can upload a file and accept/decline participation.
    """

    pipeline = models.ForeignKey(
        MatchingPipeline, on_delete=models.CASCADE, related_name="parties"
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="pipeline_parties"
    )
    file = models.FileField(
        upload_to=upload_to_pipeline_files,
        null=True,
        blank=True,
        help_text="Data file uploaded by this party",
    )
    accepted = models.BooleanField(
        default=False, help_text="Whether this party has accepted the pipeline"
    )
    joined_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return (
            f"{self.user.username} - {self.pipeline.name} (Accepted: {self.accepted})"
        )

    def accept_and_upload(self, file):
        """Accept the pipeline and upload file"""
        self.file = file
        self.accepted = True
        self.save()

        # Update pipeline parties_accepted count
        self.pipeline.parties_accepted = self.pipeline.parties.filter(
            accepted=True
        ).count()
        self.pipeline.update_status()

    def decline(self):
        """Decline participation in the pipeline"""
        self.accepted = False
        self.file = None
        self.save()

        # Update pipeline parties_accepted count
        self.pipeline.parties_accepted = self.pipeline.parties.filter(
            accepted=True
        ).count()
        self.pipeline.update_status()

    class Meta:
        unique_together = ["pipeline", "user"]
        verbose_name = "Pipeline Party"
        verbose_name_plural = "Pipeline Parties"
