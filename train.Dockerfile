# Use our image with dataset
FROM polevanovairina/visloc-apr:with_dataset

RUN ./train.sh

ENTRYPOINT ["./keepalive.sh"]
