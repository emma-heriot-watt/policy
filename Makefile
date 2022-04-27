# Use .env file if exists when running Makefile
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Increase open-file limit when using instances.
# This is only a soft-increase, meaning it only works while the current terminal
# session is active. All other sessions default to basics
.PHONY : increase-open-file-limit
increase-open-file-limit :
	ulimit -S -n unlimited


# Download the necessary checkpoints and build the image that will get submitted to eval.ai
.PHONY : build-teach-inference-image
build-teach-inference-image :
	docker build -t policy-teach-inference-api:latest -f docker/teach-inference.Dockerfile
