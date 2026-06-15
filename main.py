"""IQC Alpha Mining Machine — Terminal Research Platform.

Entry point for the CLI application.
Initializes the database and launches the Typer CLI.
"""

from database.database import engine
from database.models import Base
from cli.commands import app

# Create all tables on startup (safe — skips existing tables)
Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    app()