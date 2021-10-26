FROM polevanovairina/visloc-apr:trained

RUN ./test.sh

ENTRYPOINT ["./keepalive.sh"]
