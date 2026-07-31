export HEADROOM_TELEMETRY := off

.PHONY: aider golem

aider:
	headroom wrap aider \
		--watch-files \
		--no-show-model-warnings \
		--no-gitignore \
		--config .aider.conf.yml