FROM polevanovairina/visloc-apr:tested

RUN ./eval.sh

ENTRYPOINT ["./keepalive.sh"]
