from django.urls import path
from .views import (
    UserListView, UserDetailView, current_user_view, login_view, logout_view
)

urlpatterns = [
    # Authentication endpoints
    path('auth/login/', login_view, name='login'),
    path('auth/logout/', logout_view, name='logout'),

    # User endpoints
    path('users/', UserListView.as_view(), name='user-list'),
    path('users/<int:pk>/', UserDetailView.as_view(), name='user-detail'),
    path('users/me/', current_user_view, name='current-user'),

]
