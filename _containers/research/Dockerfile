# Dockerfile

FROM jupyter/pyspark-notebook:spark-3.3.2

# Install Java 11 (JDK)
USER root
RUN apt-get update && apt-get install -y openjdk-11-jdk && \
    update-alternatives --install /usr/bin/java java /usr/lib/jvm/java-11-openjdk-amd64/bin/java 1 && \
    update-alternatives --set java /usr/lib/jvm/java-11-openjdk-amd64/bin/java

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Install curl
RUN apt-get update && apt-get install -y curl

# Install GraphFrames JAR
ENV GRAPHFRAMES_VERSION=0.8.2-spark3.3-s_2.12
RUN mkdir -p /opt/graphframes && \
    curl -L -o /opt/graphframes/graphframes.jar \
    https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.3-s_2.12/graphframes-0.8.2-spark3.3-s_2.12.jar

ENV SPARK_EXTRA_CLASSPATH="/opt/graphframes/graphframes.jar"

# Back to non-root for Jupyter
USER $NB_UID
