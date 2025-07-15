# Generated manually for creating default parties

from django.contrib.auth import get_user_model
from django.db import migrations


def create_default_parties(apps, schema_editor):
    """
    Create 5 default parties: partya, partyb, partyc, partyd, partye
    Each with password: test1234
    """
    User = get_user_model()

    parties = [
        {
            "username": "partya",
            "email": "partya@example.com",
            "first_name": "Party",
            "last_name": "A",
        },
        {
            "username": "partyb",
            "email": "partyb@example.com",
            "first_name": "Party",
            "last_name": "B",
        },
        {
            "username": "partyc",
            "email": "partyc@example.com",
            "first_name": "Party",
            "last_name": "C",
        },
        {
            "username": "partyd",
            "email": "partyd@example.com",
            "first_name": "Party",
            "last_name": "D",
        },
        {
            "username": "partye",
            "email": "partye@example.com",
            "first_name": "Party",
            "last_name": "E",
        },
    ]

    for party_data in parties:
        # Check if user already exists
        if not User.objects.filter(username=party_data["username"]).exists():
            user = User.objects.create_user(
                username=party_data["username"],
                email=party_data["email"],
                first_name=party_data["first_name"],
                last_name=party_data["last_name"],
                password="test1234",
            )
            print(f"Created user: {user.username}")
        else:
            print(f"User {party_data['username']} already exists, skipping...")


def reverse_create_default_parties(apps, schema_editor):
    """
    Remove the default parties if migration is reversed
    """
    User = get_user_model()

    usernames = ["partya", "partyb", "partyc", "partyd", "partye"]

    for username in usernames:
        try:
            user = User.objects.get(username=username)
            user.delete()
            print(f"Deleted user: {username}")
        except User.DoesNotExist:
            print(f"User {username} does not exist, skipping...")


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(
            create_default_parties,
            reverse_create_default_parties,
        ),
    ]
