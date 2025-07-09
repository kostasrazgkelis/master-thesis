import time
import pandas as pd
import os
import uuid
from celery import shared_task
from django.core.files.storage import default_storage
from django.utils import timezone
from pipeline.models import MatchingPipeline
import logging

logger = logging.getLogger(__name__)


@shared_task
def test_celery_connection():
    """
    Simple test task to verify Celery is working correctly.
    """
    # Generate a unique UUID for this test execution
    test_id = str(uuid.uuid4())
    
    logger.info(f"🚀 Test Celery task started! Test ID: {test_id}")
    
    time.sleep(5)  # Simulate some processing delay
    
    result = {
        'test_id': test_id,
        'message': 'Celery is working with auto-reload! 🔥',
        'timestamp': timezone.now().isoformat(),
        'status': 'success',
        'worker_info': {
            'task_name': 'test_celery_connection',
            'execution_time': '5 seconds'
        }
    }
    
    logger.info(f"✅ Test Celery task completed: {result}")
    return result

@shared_task(bind=True)
def process_pipeline(self, pipeline_id):
    """
    Process a matching pipeline by reading all uploaded files with pandas 
    and counting the number of rows per file.
    
    This is a simple initial implementation that demonstrates the Celery integration.
    In a real implementation, this would perform the actual data matching logic.
    
    Args:
        pipeline_id (str): UUID of the pipeline to process
        
    Returns:
        dict: Result containing row counts and processing status
    """
    try:
        logger.info(f"Starting pipeline processing for pipeline {pipeline_id}")
        
        # Get the pipeline instance
        try:
            pipeline = MatchingPipeline.objects.get(id=pipeline_id)
        except MatchingPipeline.DoesNotExist:
            error_msg = f"Pipeline {pipeline_id} not found"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Verify pipeline is in READY state
        if pipeline.status != 'READY':
            error_msg = f"Pipeline {pipeline_id} is not in READY state (current: {pipeline.status})"
            logger.error(error_msg)
            pipeline.mark_failed(error_msg)
            return {"error": error_msg}
        
        # Mark pipeline as RUNNING
        pipeline.status = 'RUNNING'
        pipeline.execution_started_at = timezone.now()
        pipeline.save()
        logger.warning(f"Pipeline {pipeline_id} marked as RUNNING")
        
        # Process files from all parties
        results = {}
        total_rows = 0
        
        for party in pipeline.parties.filter(accepted=True):
            if not party.file:
                error_msg = f"Party {party.user.username} has no file uploaded"
                logger.error(error_msg)
                pipeline.mark_failed(error_msg)
                return {"error": error_msg}
            
            try:
                # Get the file path
                file_path = party.file.path
                
                # Read the file with pandas
                logger.info(f"Reading file for party {party.user.username}: {file_path}")
                
                # Try to read as CSV first, then other formats
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    # Try with different encoding
                    df = pd.read_csv(file_path, encoding='latin-1')
                except Exception as e:
                    # If CSV fails, try Excel
                    try:
                        df = pd.read_excel(file_path)
                    except Exception as excel_error:
                        error_msg = f"Failed to read file for party {party.user.username}: {str(e)}"
                        logger.error(error_msg)
                        pipeline.mark_failed(error_msg)
                        return {"error": error_msg}
                
                # Count rows and store result
                row_count = len(df)
                total_rows += row_count
                
                results[party.user.username] = {
                    'rows': row_count,
                    'columns': list(df.columns),
                    'file_name': os.path.basename(party.file.name)
                }
                
                logger.info(f"Party {party.user.username}: {row_count} rows, {len(df.columns)} columns")
                
            except Exception as e:
                error_msg = f"Error processing file for party {party.user.username}: {str(e)}"
                logger.error(error_msg)
                pipeline.mark_failed(error_msg)
                return {"error": error_msg}
        
        # Prepare final result
        final_result = {
            'status': 'success',
            'total_parties': len(results),
            'total_rows': total_rows,
            'party_results': results,
            'processed_at': timezone.now().isoformat(),
            'pipeline_id': str(pipeline_id)
        }
        
        # Mark pipeline as completed and store results
        pipeline.mark_completed(final_result)
        
        logger.info(f"Pipeline {pipeline_id} completed successfully")
        logger.info(f"Final result: {final_result}")
        
        return final_result
        
    except Exception as e:
        error_msg = f"Unexpected error in pipeline processing: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        try:
            pipeline = MatchingPipeline.objects.get(id=pipeline_id)
            pipeline.mark_failed(error_msg)
        except:
            logger.error(f"Failed to mark pipeline {pipeline_id} as failed")
        
        return {"error": error_msg}


@shared_task
def cleanup_failed_pipelines():
    """
    Periodic task to clean up pipelines that have been stuck in RUNNING state
    for too long (e.g., due to worker crashes).
    """
    from datetime import timedelta
    
    # Find pipelines that have been running for more than 1 hour
    cutoff_time = timezone.now() - timedelta(hours=1)
    stuck_pipelines = MatchingPipeline.objects.filter(
        status='RUNNING',
        execution_started_at__lt=cutoff_time
    )
    
    count = 0
    for pipeline in stuck_pipelines:
        pipeline.mark_failed("Pipeline execution timed out")
        count += 1
        logger.warning(f"Marked stuck pipeline {pipeline.id} as failed")
    
    logger.info(f"Cleaned up {count} stuck pipelines")
    return {"cleaned_up": count}
