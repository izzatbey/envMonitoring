FROM python:3.11
WORKDIR /usr/src/app
COPY . .
RUN pip install