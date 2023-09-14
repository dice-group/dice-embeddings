TAG=hub.cs.upb.de/dice-research/images/dice-embeddings:1.06
LATEST=hub.cs.upb.de/dice-research/images/dice-embeddings:latest

build:
	docker build --tag $(TAG) --tag $(LATEST) .

push:
	docker push $(TAG)
	docker push $(LATEST)
