"""
WSGI configuration for Timetable Generator Backend

This file contains the WSGI application object that can be used by WSGI servers
like Gunicorn, uWSGI, or Apache with mod_wsgi to serve the FastAPI application.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import the FastAPI app from main.py
from main import app

# For WSGI compatibility, we need to wrap the ASGI app
# Install a2wsgi for proper ASGI to WSGI conversion: pip install a2wsgi
try:
    from a2wsgi import ASGIMiddleware

    application = ASGIMiddleware(app)
except ImportError:
    # Fallback: If a2wsgi is not available, this will raise an error
    # indicating that a2wsgi needs to be installed for WSGI deployment
    raise ImportError(
        "a2wsgi is required for WSGI deployment. Install it with: pip install a2wsgi"
    )

# Alternative: If you prefer to use uvicorn's WSGI mode (less recommended)
# from uvicorn.workers import WSGIWorker
# application = WSGIWorker(app)
