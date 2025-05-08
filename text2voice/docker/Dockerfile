# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD requirements.txt /app

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt \
#     -i https://pypi.tuna.tsinghua.edu.cn/simple \
#     --trusted-host pypi.tuna.tsinghua.edu.cn \
#     --default-timeout=300 --retries=5

COPY ./pip_packages /tmp/pip_packages

# Install from local packages
RUN pip install --no-cache-dir \
    --find-links=/tmp/pip_packages \
    -r requirements.txt

# Make port 80 available to the world outside this container
RUN mkdir -p /app/data

EXPOSE 80

RUN echo "successfully installed"

# Run app.py when the container launches
CMD ["python", "server.py"]