.PHONY:	run-gui
run-gui:
	pavucontrol &
	python3 gui/main.py

################################################################
# Containerization stuff
################################################################

# For absolute path usage later
cwd := $(shell pwd)

.PHONY:	docker
docker:
	docker build --tag 'arbfn' .
	docker run \
		--mount type=bind,source="${cwd}",target="/host" \
		-i \
		-t arbfn:latest \

.PHONY:	podman
podman:
	podman build --tag 'arbfn' .
	podman run \
		--mount type=bind,source="${cwd}",target="/host" \
		--mount type=bind,source="/",target="/hostroot" \
		-i \
		-t arbfn:latest \
