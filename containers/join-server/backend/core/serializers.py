from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    """
    Serializer for CustomUser model.
    Excludes sensitive fields like password.
    """
    
    class Meta:
        model = User
        fields = [
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'phone_number',
            'date_of_birth',
            'bio',
            'is_active',
            'date_joined',
            'last_login'
        ]
        read_only_fields = ['id', 'date_joined', 'last_login']


class UserListSerializer(serializers.ModelSerializer):
    """
    Simplified serializer for user list view.
    Only includes essential fields for better performance.
    """
    
    class Meta:
        model = User
        fields = [
            'id',
            'username',
            'email',
            'first_name',
            'last_name',
            'is_active',
            'date_joined'
        ]
