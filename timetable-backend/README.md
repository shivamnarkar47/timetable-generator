# Timetable Generator Backend

A FastAPI-based backend service for generating optimized academic timetables using constraint-based algorithms.

## Features

- Constraint-based timetable optimization
- RESTful API for timetable generation
- Support for multiple departments, faculty, and resources
- Handles complex scheduling constraints (teacher availability, room allocation, lab scheduling)

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development

Run the development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Production Deployment

### Using WSGI (Recommended for production)

This backend includes WSGI support for deployment with servers like Gunicorn, uWSGI, or Apache.

1. Install additional WSGI dependencies:
```bash
pip install gunicorn a2wsgi
```

2. Run with Gunicorn:
```bash
gunicorn wsgi:application --bind 0.0.0.0:8000 --workers 4
```

### Using ASGI (Alternative)

For better performance with async operations:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, visit:
- Interactive API docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## Main Endpoint

- `POST /generate-timetable`: Generate an optimized timetable based on provided configuration

See the docstring in `main.py` for detailed request/response format.