FROM ubuntu:16.04

LABEL maintainer="Tim Xiao <xiaozhenzhong0708@hotmail.com>"

# Install some dependencies for python
RUN apt-get update && apt-get install -y \
                  python3-pip \
                  python3-dev \
                  && \
        apt-get clean && \
        apt-get autoremove && \
        rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip3 --no-cache-dir install \
        numpy \
        scipy \
        Pillow

# Install TensorFlow CPU version with v1.3.0 with python 3.5
RUN pip3 --no-cache-dir install --upgrade \
        https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp35-cp35m-linux_x86_64.whl

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y \
                  build-essential \
                  cmake \
                  git \
                  wget \
                  unzip \
                  pkg-config \
                  libswscale-dev \
                  libtbb2 \
                  libtbb-dev \
                  libjpeg-dev \
                  libpng-dev \
                  libtiff-dev \
                  libjasper-dev \
                  libavformat-dev \
                  libpq-dev \
                  && \
        apt-get clean && \
        apt-get autoremove && \
        rm -rf /var/lib/apt/lists/*

# Install OpenCV
WORKDIR /

ENV OPENCV_VERSION="3.4.0"

RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DENABLE_AVX=ON \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
&& make install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}

# Install Flask
RUN pip3 --no-cache-dir install flask

COPY ./ /app/

WORKDIR /app/

EXPOSE 5000

ENTRYPOINT ["python3", "server.py"]
