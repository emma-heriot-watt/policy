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

# Install docker-compose directly and make path executable
# MUST BE RUN SUDO
.PHONY : install-docker-compose
install-docker-compose :
	curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
	chmod +x /usr/local/bin/docker-compose

.PHONY : setup-new-venv
setup-new-venv :
	# TODO: Check if pyenv installed
	# TODO: Check if python version installed, if not install it
	# TODO: Check if poetry installed, if not install it
	poetry install
