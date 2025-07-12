from rest_framework import generics, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.contrib.auth import get_user_model, authenticate, login, logout
from rest_framework import status
from drf_spectacular.utils import extend_schema
from .serializers import UserSerializer, UserListSerializer

User = get_user_model()


class UserListView(generics.ListAPIView):
    """
    API view to retrieve list of all users.
    """

    queryset = User.objects.all()
    serializer_class = UserListSerializer
    permission_classes = [permissions.IsAuthenticated]

    @extend_schema(
        summary="Get all users",
        description="Retrieve a list of all users in the system",
        responses={200: UserListSerializer(many=True)},
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
        responses={200: UserSerializer()},
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)


@extend_schema(
    summary="Get current user",
    description="Retrieve information about the currently authenticated user",
    responses={
        200: UserSerializer(),
        401: {"description": "Authentication credentials were not provided."},
    },
)
@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def current_user_view(request):
    """
    API view to get the current authenticated user's information.
    Returns the user that is currently logged in via session or basic auth.
    """

    if not request.user.is_authenticated:
        return Response(
            {"detail": "Authentication credentials were not provided."}, status=401
        )

    serializer = UserSerializer(request.user)
    return Response(serializer.data)


@extend_schema(
    summary="User login",
    description="Authenticate user and create session",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "username": {"type": "string", "description": "Username"},
                "password": {
                    "type": "string",
                    "description": "Password",
                    "format": "password",
                },
            },
            "required": ["username", "password"],
        }
    },
    responses={
        200: UserSerializer(),
        400: {
            "type": "object",
            "properties": {
                "error": {"type": "string"},
                "non_field_errors": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
)
@api_view(["POST"])
@permission_classes([permissions.AllowAny])
def login_view(request):
    """
    API view to authenticate user and create session.
    """
    from .serializers import LoginSerializer

    serializer = LoginSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.validated_data["user"]
        login(request, user)

        # Return user data
        user_serializer = UserSerializer(user)
        return Response(
            {"message": "Login successful", "user": user_serializer.data},
            status=status.HTTP_200_OK,
        )

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@extend_schema(
    summary="User logout",
    description="Logout user and destroy session",
    responses={200: {"type": "object", "properties": {"message": {"type": "string"}}}},
)
@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def logout_view(request):
    """
    API view to logout user and destroy session.
    """
    logout(request)
    return Response({"message": "Logout successful"}, status=status.HTTP_200_OK)
