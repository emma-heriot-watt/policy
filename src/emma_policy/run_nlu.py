import sys

from emma_common.api.gunicorn import create_gunicorn_server
from emma_common.api.instrumentation import instrument_app
from emma_common.aws.cloudwatch import add_cloudwatch_handler_to_logger
from emma_common.logging import InstrumentedInterceptHandler, setup_logging, setup_rich_logging

from emma_policy.commands.run_simbot_nlu import ApiSettings, app as api


def main() -> None:
    """Run the EMMA NLU API."""
    settings = ApiSettings()

    if settings.traces_to_opensearch:
        instrument_app(api, settings.opensearch_service_name, settings.otlp_endpoint)
        setup_logging(sys.stdout, InstrumentedInterceptHandler())
    else:
        setup_rich_logging(rich_traceback_show_locals=False)

    server = create_gunicorn_server(api, settings.host, settings.port, settings.workers)

    if settings.log_to_cloudwatch:
        add_cloudwatch_handler_to_logger(
            boto3_profile_name=settings.aws_profile,
            log_stream_name=settings.watchtower_log_stream_name,
            log_group_name=settings.watchtower_log_group_name,
            send_interval=1,
            enable_trace_logging=settings.traces_to_opensearch,
        )

    server.run()


if __name__ == "__main__":
    main()
