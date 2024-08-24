FROM debian:bookworm-slim AS rdkit-build-env

RUN apt-get update \
 && apt-get install -y \
 libxrender1 \
 libsm6 \
 libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

FROM python:3.8-slim

# Copy necessary files from the previous stage
COPY --from=rdkit-build-env /usr/local /usr/local
COPY --from=rdkit-build-env /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]
