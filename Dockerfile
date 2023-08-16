FROM python:3.10
#RUN pip install dicee==0.0.4
RUN pip3 install -r requirements.txt
WORKDIR /dicee
ADD . .
CMD ./main.py
