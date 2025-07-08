from rest_framework import serializers
from django.contrib.auth import get_user_model, authenticate, authenticate

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
            'is_active',
            'date_joined',
            'last_login'
        ]
        read_only_fields = ['id', 'date_joined', 'last_login']
        
    def to_representation(self, instance):
        """
        Add custom fields if they exist in the model.
        """
        data = super().to_representation(instance)
        
        # Safely add custom fields if they exist
        if hasattr(instance, 'phone_number'):
            data['phone_number'] = instance.phone_number
        if hasattr(instance, 'date_of_birth'):
            data['date_of_birth'] = instance.date_of_birth
        if hasattr(instance, 'bio'):
            data['bio'] = instance.bio
            
        return data


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


class LoginSerializer(serializers.Serializer):
    """
    Serializer for user login.
    """
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True)
    
    def validate(self, attrs):
        username = attrs.get('username')
        password = attrs.get('password')
        
        if username and password:
            # Try to authenticate the user
            user = authenticate(username=username, password=password)
            if user:
                if user.is_active:
                    attrs['user'] = user
                else:
                    raise serializers.ValidationError('User account is disabled.')
            else:
                raise serializers.ValidationError('Invalid username or password.')
        else:
            raise serializers.ValidationError('Username and password are required.')
        
        return attrs


class LogoutSerializer(serializers.Serializer):
    """
    Serializer for user logout.
    """
    pass

