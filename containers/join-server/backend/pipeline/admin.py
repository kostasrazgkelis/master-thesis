from django.contrib import admin
from .models import MatchingPipeline, PipelineParticipant


@admin.register(MatchingPipeline)
class MatchingPipelineAdmin(admin.ModelAdmin):
    """
    Admin configuration for MatchingPipeline model.
    """
    list_display = ('name', 'status', 'created_by', 'created_at', 'updated_at')
    list_filter = ('status', 'created_at')
    search_fields = ('name', 'created_by__username')
    readonly_fields = ('created_at', 'updated_at')
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related('created_by')


@admin.register(PipelineParticipant)
class PipelineParticipantAdmin(admin.ModelAdmin):
    """
    Admin configuration for PipelineParticipant model.
    """
    list_display = ('pipeline', 'user', 'status', 'joined_at', 'status_updated_at')
    list_filter = ('status', 'joined_at')
    search_fields = ('pipeline__name', 'user__username')
    readonly_fields = ('joined_at', 'status_updated_at')
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related('pipeline', 'user')
