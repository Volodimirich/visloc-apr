# Use our base image
FROM polevanovairina/visloc-apr:latest

RUN apt update -y && apt install -y unzip

COPY download.sh .

RUN ./download.sh

ENTRYPOINT ["./keepalive.sh"] 
