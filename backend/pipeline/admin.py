from django.contrib import admin
from .models import MatchingPipeline, PipelineParty


@admin.register(MatchingPipeline)
class MatchingPipelineAdmin(admin.ModelAdmin):
    """
    Admin configuration for MatchingPipeline model.
    """

    list_display = (
        "name",
        "status",
        "created_by",
        "parties_accepted",
        "total_parties",
        "created_at",
        "updated_at",
    )
    list_filter = ("status", "created_at")
    search_fields = ("name", "created_by__username")
    readonly_fields = ("id", "created_at", "updated_at", "parties_accepted")

    def total_parties(self, obj):
        return obj.total_parties

    total_parties.short_description = "Total Parties"

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related("created_by")


@admin.register(PipelineParty)
class PipelinePartyAdmin(admin.ModelAdmin):
    """
    Admin configuration for PipelineParty model.
    """

    list_display = (
        "pipeline",
        "user",
        "accepted",
        "has_file",
        "joined_at",
        "updated_at",
    )
    list_filter = ("accepted", "joined_at")
    search_fields = ("pipeline__name", "user__username")
    readonly_fields = ("joined_at", "updated_at")

    def has_file(self, obj):
        return bool(obj.file)

    has_file.boolean = True
    has_file.short_description = "File Uploaded"

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related("pipeline", "user")
