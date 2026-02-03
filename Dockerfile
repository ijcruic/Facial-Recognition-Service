FROM nvcr.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04

LABEL maintainer="SKOPE AI"
LABEL classification="UNCLASSIFIED"
LABEL description="Docker container for a Facial Recognition API Service"

WORKDIR /service

# Direct container to never require user input
ENV DEBIAN_FRONTEND=noninteractive
# Base directory for non-essential data files
ENV XDG_CACHE_HOME=/service/xdg_cache

# Use production environment mirrors
COPY ./sources.list /etc/apt/sources.list

# Update packages, distro, and install required dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y python3 python3-pip git libgl1 libglib2.0-0 pandoc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./requirements.txt /service/requirements.txt
COPY ./pip.conf /usr/pip.conf
RUN rm -f /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf && \
    pip3 install --upgrade pip wheel setuptools && \
    pip3 install --upgrade -r /service/requirements.txt

# turn the readme into an HTML file to be served to users
COPY ./README.md /service/README.md
RUN pandoc /service/README.md -f gfm -t html -s -o /service/README.html
RUN rm /service/README.md

COPY ./favicon.ico /service/favicon.ico
COPY ./app /service/app

# Copy model weights into the container
COPY ./models_weights /service/model_weights

# Expose port 80 for the API service
EXPOSE 80

# Start the API using uvicorn
CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "80"]
