# Use our base image
FROM polevanovairina/visloc-apr:latest

COPY download.sh .

RUN ./download.sh

ENTRYPOINT ["./keepalive.sh"] 
