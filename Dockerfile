FROM python:3.9.11

RUN apt-get update

#copy local code to container image
ENV APP_HOME /app
WORKDIR $APP_HOME 
COPY . ./

#install dependecies
RUN pip install -r requirements.txt 

CMD ['streamlit', 'run', '--server.enableCORS', 'false', 'main.py']





