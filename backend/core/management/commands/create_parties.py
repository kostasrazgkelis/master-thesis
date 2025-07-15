from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Create default parties (partya through partye) with password test1234"

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force recreate users even if they exist (will delete existing)",
        )

    def handle(self, *args, **options):
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

        created_count = 0
        updated_count = 0

        for party_data in parties:
            username = party_data["username"]

            if options["force"] and User.objects.filter(username=username).exists():
                User.objects.filter(username=username).delete()
                self.stdout.write(
                    self.style.WARNING(f"Deleted existing user: {username}")
                )

            user, created = User.objects.get_or_create(
                username=username, defaults=party_data
            )

            if created:
                user.set_password("test1234")
                user.save()
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f"Created user: {username}"))
            else:
                if options["force"]:
                    # Update existing user
                    for key, value in party_data.items():
                        setattr(user, key, value)
                    user.set_password("test1234")
                    user.save()
                    updated_count += 1
                    self.stdout.write(self.style.SUCCESS(f"Updated user: {username}"))
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f"User {username} already exists, skipping..."
                        )
                    )

        if created_count > 0:
            self.stdout.write(
                self.style.SUCCESS(f"Successfully created {created_count} users")
            )

        if updated_count > 0:
            self.stdout.write(
                self.style.SUCCESS(f"Successfully updated {updated_count} users")
            )

        self.stdout.write(
            self.style.SUCCESS("\nAll parties created with password: test1234")
        )
        self.stdout.write("Users: partya, partyb, partyc, partyd, partye")
