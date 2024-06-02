# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirement s.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV ROOT_DIR=/app/code
ENV FLASK_APP=/app/flask/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=80

# Run app.py when the container launches
CMD ["flask", "run"]