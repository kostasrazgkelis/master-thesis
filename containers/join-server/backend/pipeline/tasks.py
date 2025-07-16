import os
import logging

from django.db import transaction
from django.conf import settings
from django.utils import timezone

from celery import shared_task

from pipeline.models import MatchedData, MatchingPipeline

from itertools import combinations
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    expr,
    explode,
    lit,
    col,
    concat_ws,
    struct,
    collect_list,
    size,
    monotonically_increasing_id,
)

logger = logging.getLogger(__name__)
COLUMNS = ["_c1", "_c2", "_c3", "_c4", "_c5"]


@shared_task(bind=True, autoretry_for=(), retry_kwargs={"max_retries": 0})
def multi_party_matching_pipeline(self, pipeline_id):

    # TODO this will be updated for the actual mathcing alogirthm with spark
    pipeline = None

    try:

        with transaction.atomic():
            pipeline = MatchingPipeline.objects.select_for_update().get(id=pipeline_id)

            pipeline.status = "RUNNING"
            pipeline.execution_started_at = timezone.now()
            pipeline.save()

        if len(pipeline.match_columns) != len(COLUMNS):
            raise ValueError("Mismatch between pipeline.match_columns and COLUMNS")

        spark = (
            SparkSession.builder.appName(f"MatchedData-{pipeline_id}")
            .master("local[*]")
            # Tungsten engine configurations
            .config("spark.sql.tungsten.enabled", "true")
            .config("spark.sql.codegen.wholeStage", "true")
            .config("spark.sql.codegen.factoryMode", "CODEGEN_ONLY")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
            # Memory management
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memoryFraction", "0.8")
            .config("spark.sql.shuffle.partitions", "200")
            # Tungsten off-heap memory
            .config("spark.sql.columnVector.offheap.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryo.unsafe", "true")
            # Vectorized execution
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.parquet.enableVectorizedReader", "true")
            .config("spark.sql.orc.enableVectorizedReader", "true")
            .getOrCreate()
        )

        parties = pipeline.get_parties_status()

        # Load Data and process each party's dataset
        processed_dfs = []
        for party in parties:
            user_id, file = party["user_id"], party["file"]

            df_tmp = spark.read.csv(
                "." + os.path.join(settings.MEDIA_ROOT, file),
                header=True,
                inferSchema=True,
            )
            df_tmp = df_tmp.select(*pipeline.match_columns)
            df_tmp = df_tmp.select(*pipeline.match_columns).toDF(*COLUMNS)
            df_tmp = df_tmp.withColumn("id", monotonically_increasing_id())
            df_tmp = df_tmp.withColumn("origin", lit(user_id))

            df_tmp = df_tmp.select("origin", "id", *COLUMNS)
            df_tmp.cache()

            df_tmp.write.mode("overwrite").parquet(
                path=os.path.join(
                    settings.MEDIA_ROOT,
                    "pipelines",
                    str(pipeline_id),
                    "participants",
                    f"party_{user_id}",
                )
            )
            processed_dfs.append(df_tmp)

        # Union all datasets from different parties
        multiparty_datasets = processed_dfs[0]
        for df in processed_dfs[1:]:
            multiparty_datasets = multiparty_datasets.union(df)

        # Generate blocking keys by combining columns
        column_combinations = list(combinations(COLUMNS, 3))

        blocking_passes = []
        for pass_id, combo in enumerate(column_combinations):
            df_with_block_key = multiparty_datasets.withColumn(
                "block_key", concat_ws("_", *[col(c) for c in combo])
            )
            blocked_df = (
                df_with_block_key.groupBy("block_key")
                .agg(collect_list(struct(*["origin", "id"])).alias("records"))
                .withColumn("pass_id", lit(pass_id))
            )
            blocked_df = blocked_df.filter(size(col("records")) > 1)
            blocking_passes.append(blocked_df)

        multiparty_struct = reduce(
            lambda df1, df2: df1.unionByName(df2), blocking_passes
        )

        # Write the final result to Parquet
        output_path = os.path.join(
            settings.MEDIA_ROOT, "pipelines", str(pipeline_id), "output", "struct"
        )
        multiparty_struct.write.mode("overwrite").parquet(output_path)
        pipeline.output_struct = output_path
        pipeline.save()

    except MatchingPipeline.DoesNotExist:
        pipeline.mark_failed("Pipeline not found")
    except Exception as e:
        msg = f"Unexpected error: {str(e)}"
        logger.error(msg, exc_info=True)
        pipeline.mark_failed(msg)
    else:
        pipeline.mark_completed("done")
    finally:
        spark.stop()


@shared_task(bind=True, autoretry_for=(), retry_kwargs={"max_retries": 0})
def get_matched_data(self, pipeline_id):
    spark = None
    matched_data = None

    try:
        matched_data = MatchedData.objects.get(
            uuid=pipeline_id
        )  # Changed from uuid to id

        spark = (
            SparkSession.builder.appName(f"MatchedData-{pipeline_id}")
            .master("local[*]")
            # Tungsten engine configurations
            .config("spark.sql.tungsten.enabled", "true")
            .config("spark.sql.codegen.wholeStage", "true")
            .config("spark.sql.codegen.factoryMode", "CODEGEN_ONLY")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
            # Memory management
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memoryFraction", "0.8")
            .config("spark.sql.shuffle.partitions", "200")
            # Tungsten off-heap memory
            .config("spark.sql.columnVector.offheap.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryo.unsafe", "true")
            # Vectorized execution
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.parquet.enableVectorizedReader", "true")
            .config("spark.sql.orc.enableVectorizedReader", "true")
            .getOrCreate()
        )

        multi_party_pipeline = matched_data.pipeline

        # Check if output_struct exists and is not empty
        if not multi_party_pipeline.output_struct:
            raise ValueError("Pipeline output_struct is empty or None")

        output_struct_path = os.path.join(
            settings.MEDIA_ROOT, multi_party_pipeline.output_struct
        )
        if not os.path.exists(output_struct_path):
            raise FileNotFoundError(
                f"Output struct path does not exist: {output_struct_path}"
            )

        multiparty_struct = spark.read.parquet(output_struct_path)
        multiparty_struct.cache()

        parties = multi_party_pipeline.get_parties_status()

        processed_dfs = []
        for party in parties:
            user_id = party["user_id"]
            party_path = os.path.join(
                settings.MEDIA_ROOT,
                "pipelines",
                str(multi_party_pipeline.id),
                "participants",
                f"party_{user_id}",
            )

            if not os.path.exists(party_path):
                logger.warning(f"Party data path does not exist: {party_path}")
                continue

            df_tmp = spark.read.parquet(party_path)
            df_tmp.cache()
            processed_dfs.append({"user_id": user_id, "file": df_tmp})

        if not processed_dfs:
            raise ValueError("No valid party data found")

        # Fix the filter expression for updated model structure
        flattened_df = (
            multiparty_struct.filter(
                expr(
                    f"""
                exists(records, x -> x.origin = {matched_data.left_party.pk})
                AND exists(records, x -> x.origin IN ({",".join(map(str, matched_data.right_parties.values_list('id', flat=True)))}))
            """
                )
            )
            .select(explode(col("records")).alias("record"))
            .select(col("record.origin").alias("origin"), col("record.id").alias("id"))
        )

        filtered_df = flattened_df.dropDuplicates(["origin", "id"])

        results = []
        for info_df in processed_dfs:
            origin, tmp_df = info_df["user_id"], info_df["file"]
            tmp_df = tmp_df.withColumn("origin", lit(origin))

            # Filter only the relevant (origin, id) pairs for this dataset
            df_filtered = filtered_df.filter(col("origin") == origin)

            # Check if there are any matches for this origin
            if df_filtered.count() == 0:
                logger.info(f"No matches found for origin {origin}")
                continue

            joined_df = tmp_df.join(
                df_filtered, on=["origin", "id"], how="inner"
            ).select(*COLUMNS)

            results.append(joined_df)

        final_result = reduce(lambda df1, df2: df1.unionByName(df2), results)

        # Create output directory
        output_dir = os.path.join(
            settings.MEDIA_ROOT,
            "pipelines",
            str(multi_party_pipeline.id),
            "output",
            "requested_matches",
        )
        os.makedirs(output_dir, exist_ok=True)

        # Create unique filename based on timestamp
        timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"match_{timestamp}")

        # Write the result
        final_result.coalesce(1).write.mode("overwrite").parquet(output_path)

    except MatchedData.DoesNotExist:
        msg = f"MatchedData with ID {pipeline_id} not found"
        logger.error(msg)
        return {"error": msg}

    except (FileNotFoundError, ValueError) as e:
        msg = f"Data error: {str(e)}"
        logger.error(msg)
        if matched_data:
            matched_data.status = "FAILED"
            matched_data.save()
        return {"error": msg}

    except Exception as e:
        msg = f"Unexpected error: {str(e)}"
        logger.error(msg, exc_info=True)
        if matched_data:
            matched_data.status = "FAILED"
            matched_data.save()
        return {"error": msg}
    else:
        # Update matched data
        matched_data.folder_path = output_path
        matched_data.status = "COMPLETED"
        matched_data.save()

        return {
            "status": "completed",
            "output_path": output_path,
            "matched_data_id": matched_data.uuid,
        }
    finally:
        if spark:
            spark.stop()
