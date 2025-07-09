import time
from django.db import transaction

from celery import shared_task
from django.core.files.storage import default_storage
from django.utils import timezone
from pipeline.models import MatchingPipeline
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def multi_party_matching_pipeline(self, pipeline_id):

    #TODO this will be updated for the actual mathcing alogirthm with spark
    pipeline = None

    try:
        with transaction.atomic():
            pipeline = (
                MatchingPipeline.objects
                .select_for_update()
                .get(id=pipeline_id)
            )

            pipeline.status = 'RUNNING'
            pipeline.execution_started_at = timezone.now()
            pipeline.save()

        # Sleep / process (outside the transaction)
        for _ in range(3):
            logger.warning("sleeping ...")
            time.sleep(2)

        # Refresh to get the latest state before marking completed
        pipeline.refresh_from_db()

        pipeline.mark_completed("done")
        return {"status": "completed"}

    except MatchingPipeline.DoesNotExist:
        msg = f"Pipeline {pipeline_id} not found"
        logger.error(msg)
        return {"error": msg}

    except Exception as e:
        msg = f"Unexpected error: {str(e)}"
        logger.error(msg, exc_info=True)

        try:
            if pipeline:
                pipeline.mark_failed(msg)
        except Exception:
            logger.error(f"Failed to mark pipeline {pipeline_id} as failed")

        return {"error": msg}
