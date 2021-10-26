FROM polevanovairina/visloc-apr:with_dataset

RUN ./eval.sh

ENTRYPOINT ["./keepalive.sh"]
