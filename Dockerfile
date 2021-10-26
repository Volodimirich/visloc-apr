# Use image with installed anaconda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /usr/app

RUN pip install torch
# Copy project files
COPY . .

ENTRYPOINT ["./keepalive.sh"] 
