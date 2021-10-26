# Use our base image
FROM polevanovairina/visloc-apr:latest

RUN ./download.sh

ENTRYPOINT ["./keepalive.sh"] 
