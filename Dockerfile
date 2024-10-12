FROM python:3.12.6-slim

WORKDIR /code

RUN pip install --no-cache-dir --upgrade pip==24.2

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY . /code

CMD ["python", "app.py"]