# Build context: repository root. One image backs every pipeline Lambda;
# CDK overrides CMD per function.
FROM public.ecr.aws/lambda/python:3.12

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /bin/uv
COPY processor /tmp/processor
RUN uv pip install --no-cache --target "${LAMBDA_TASK_ROOT}" /tmp/processor && rm -rf /tmp/processor

CMD ["atl06_processor.handlers.convert_handler"]
