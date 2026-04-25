FROM python:3.11

# Set the working directory to /code
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY . .

# Install requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variable for the port
ENV PORT=7860

# Expose the port
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
