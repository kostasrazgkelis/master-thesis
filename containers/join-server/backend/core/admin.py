from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import CustomUser

# Unregister the default User admin if it's registered
try:
    admin.site.unregister(User)
except admin.sites.NotRegistered:
    pass


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    """
    Admin configuration for CustomUser model.
    """
    # Add the custom fields to the admin interface
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {
            'fields': ('phone_number', 'date_of_birth', 'bio')
        }),
    )
    
    # Add custom fields to the add user form
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Additional Info', {
            'fields': ('phone_number', 'date_of_birth', 'bio')
        }),
    )
    
    # Display custom fields in the user list
    list_display = UserAdmin.list_display + ('phone_number', 'date_of_birth')
    
    # Make custom fields searchable
    search_fields = UserAdmin.search_fields + ('phone_number',)
