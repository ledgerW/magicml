FROM python:3.8-slim-buster

RUN pip3 install \
  sentence-transformers \
  pandas \
  numpy

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]