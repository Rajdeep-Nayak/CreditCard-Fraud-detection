"""ASGI entry point:  uvicorn fraudguard.api.main:app"""

from .app import create_app

app = create_app()
