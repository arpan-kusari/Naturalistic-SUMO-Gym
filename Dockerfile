# syntax=docker/dockerfile:1
FROM ubuntu:18.04

ENV SUMO_VERSION 1.10.0
ENV SUMO_HOME /opt/sumo

# Install system dependencies.
RUN apt-get update && apt-get -qq install \
    wget \
    g++ \
    cmake \
    libxerces-c-dev \
    libfox-1.6-0 libfox-1.6-dev \
    libgdal-dev \
    libproj-dev \
    libgl2ps-dev
    
RUN apt-get update
RUN apt-get install -y \
    python3-pip

# Download and extract source code
RUN wget http://downloads.sourceforge.net/project/sumo/sumo/version%20$SUMO_VERSION/sumo-src-$SUMO_VERSION.tar.gz
RUN tar xzf sumo-src-$SUMO_VERSION.tar.gz && \
    mv sumo-$SUMO_VERSION $SUMO_HOME && \
    rm sumo-src-$SUMO_VERSION.tar.gz

# Configure and build from source.
RUN cd $SUMO_HOME && \
    mkdir build/cmake-build && \
    cd build/cmake-build &&\
    cmake ../.. &&\
    make -j$(nproc)


WORKDIR /sumo_gym
COPY /sumo_gym .

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt