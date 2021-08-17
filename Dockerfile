FROM python

#WORKDIR /usr/src/app
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH = "$VIRTUAL_ENV/bin:$PATH"


COPY requirements.txt .

RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY /pickleJar/ /pickleJar/.
COPY /config/ /config/.
RUN ls
COPY main.py .
COPY find_variables.py .

CMD [ "/opt/venv/bin/python", "main.py" ]
