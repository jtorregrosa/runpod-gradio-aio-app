# Stage 1: Build the Python environment
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /sources

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for the project
RUN python3 -m venv /opt/venv

# Activate virtual environment and install project dependencies
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements.txt and install dependencies
COPY src/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Stage 2: Build the final image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

# Copy required scripts
COPY pre_start.sh /pre_start.sh
COPY stop_gradio_aio_app.sh /restart_app.sh
COPY relauncher.py /relauncher.py

# Copy the project files
WORKDIR /sources
COPY src /sources

SHELL ["/bin/bash", "-c"]

# Expose the port Gradio will run on
EXPOSE 7860
EXPOSE 8888