IMAGE_NAME=person-detection
CONTAINER_NAME=person-detection-cont

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --name $(CONTAINER_NAME) -p 8000:8000 -d $(IMAGE_NAME)

stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

restart: stop run

clean:
	docker rmi $(IMAGE_NAME) || true
