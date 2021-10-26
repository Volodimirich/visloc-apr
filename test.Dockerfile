FROM polevanovairina/visloc-apr:with_dataset

RUN ./test.sh

ENTRYPOINT ["./keepalive.sh"]
