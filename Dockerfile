# MiniVectorDB — build from source, run in stdin/stdout mode.
# No EXPOSE / no port mapping: the judge talks to the process over pipes.
# All dependencies are installed at build time; the runtime needs no network.

FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ libc6-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY src/ ./src/
# Static libstdc++/libgcc so the runtime stage needs no compiler libraries.
RUN mkdir -p /out && g++ -O3 -std=c++17 -static-libstdc++ -static-libgcc \
        -o /out/mvdb src/main.cpp

FROM ubuntu:24.04
RUN mkdir -p /data
COPY --from=build /out/mvdb /usr/local/bin/mvdb
# /data is the only writable path the engine uses; the judge bind-mounts it.
VOLUME ["/data"]
ENTRYPOINT ["/usr/local/bin/mvdb", "/data"]