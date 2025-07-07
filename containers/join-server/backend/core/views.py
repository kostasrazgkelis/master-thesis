from rest_framework import generics, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from drf_spectacular.utils import extend_schema
from .serializers import UserSerializer, UserListSerializer

User = get_user_model()


class UserListView(generics.ListAPIView):
    """
    API view to retrieve list of all users.
    """
    queryset = User.objects.all()
    serializer_class = UserListSerializer
    permission_classes = [permissions.IsAuthenticated]  # Require authentication
    
    @extend_schema(
        summary="Get all users",
        description="Retrieve a list of all users in the system",
        responses={200: UserListSerializer(many=True)}
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)


class UserDetailView(generics.RetrieveAPIView):
    """
    API view to retrieve a specific user by ID.
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    @extend_schema(
        summary="Get user details",
        description="Retrieve detailed information about a specific user",
        responses={200: UserSerializer()}
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)


@extend_schema(
    summary="Get current user",
    description="Retrieve information about the currently authenticated user",
    responses={200: UserSerializer()}
)
@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def current_user_view(request):
    """
    API view to get the current authenticated user's information.
    """
    serializer = UserSerializer(request.user)
    return Response(serializer.data)
