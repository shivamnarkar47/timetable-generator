#!/bin/bash

# Timetable Generator Automation Script
# This script sets up and runs both the backend and frontend services

set -e  # Exit on any error

echo "🚀 Starting Timetable Generator..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

if ! command_exists npm; then
    if ! command_exists bun; then
        echo "❌ Neither npm nor bun is installed. Please install Node.js package manager."
        exit 1
    fi
    PACKAGE_MANAGER="bun"
else
    PACKAGE_MANAGER="npm"
fi

echo "📦 Using package manager: $PACKAGE_MANAGER"

# Setup backend
echo "🔧 Setting up backend..."
cd timetable-backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Start backend server in background
echo "🖥️  Starting backend server..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Go back to root directory
cd ..

# Setup frontend
echo "🎨 Setting up frontend..."
cd timetable-frontend

# Install Node dependencies
echo "📥 Installing Node dependencies..."
if [ "$PACKAGE_MANAGER" = "bun" ]; then
    bun install
else
    npm install
fi

# Start frontend dev server in background
echo "🌐 Starting frontend dev server..."
if [ "$PACKAGE_MANAGER" = "bun" ]; then
    bun run dev &
else
    npm run dev &
fi
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

# Go back to root
cd ..

echo ""
echo "✅ Timetable Generator is now running!"
echo "📊 Backend API: http://localhost:8000"
echo "🎯 Frontend App: http://localhost:5173"
echo ""
echo "To stop the services, run:"
echo "kill $BACKEND_PID $FRONTEND_PID"
echo "or use Ctrl+C to stop this script"

# Wait for processes (this keeps the script running)
wait $BACKEND_PID $FRONTEND_PID