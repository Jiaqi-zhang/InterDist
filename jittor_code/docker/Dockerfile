FROM jittor/jittor-cuda-11-1:latest

WORKDIR /workspace
COPY docker/requirements.txt /tmp/interdist-jittor-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/interdist-jittor-requirements.txt
