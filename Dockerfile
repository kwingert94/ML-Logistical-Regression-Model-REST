FROM python

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY crunchyImputer.sav
COPY crunchyScaler.save
COPY finalized_model.sav
Copy main.py

CMD [ "python", "./main" ]