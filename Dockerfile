FROM python:3.10-slim

RUN pip install torch numpy torchvision torcheval

WORKDIR /workspace
COPY . /workspace

ENTRYPOINT ["python"]


