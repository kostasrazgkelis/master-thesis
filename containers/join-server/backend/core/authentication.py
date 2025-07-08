from rest_framework.authentication import SessionAuthentication


class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    Custom SessionAuthentication that doesn't enforce CSRF for API calls.
    This is useful for API endpoints accessed via Swagger UI.
    """
    
    def enforce_csrf(self, request):
        """
        Don't enforce CSRF for API calls.
        """
        return
    
    def authenticate(self, request):
        """
        Returns a `User` instance if the request session currently has a logged in user.
        Otherwise returns `None`.
        """
        # Get the underlying user from the session
        user = getattr(request._request, 'user', None)
        
        # Unauthenticated, CSRF validation not required
        if not user or not user.is_active:
            return None

        return (user, None)
