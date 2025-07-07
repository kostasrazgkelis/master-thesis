from django.urls import path
from .views import UserListView, UserDetailView, current_user_view

urlpatterns = [
    # User endpoints
    path('users/', UserListView.as_view(), name='user-list'),
    path('users/<int:pk>/', UserDetailView.as_view(), name='user-detail'),
    path('users/me/', current_user_view, name='current-user'),
]
