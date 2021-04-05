# Pull the base image with python 3.7 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.7

# Install OS packages for Pillow-SIMD
RUN yum -y install tar gzip zlib freetype-devel \
    gcc \
    ghostscript \
    lcms2-devel \
    libffi-devel \
    libimagequant-devel \
    libjpeg-devel \
    libraqm-devel \
    libtiff-devel \
    libwebp-devel \
    make \
    openjpeg2-devel \
    rh-python36 \
    rh-python36-python-virtualenv \
    sudo \
    tcl-devel \
    tk-devel \
    tkinter \
    which \
    xorg-x11-server-Xvfb \
    zlib-devel \
    && yum clean all

# Copy the requirements.txt file to the container
COPY requirements-tf.txt ./

# Install the python requirements from requirements.txt
RUN python3.7 -m pip install -r requirements-tf.txt
# Replace Pillow with Pillow-SIMD to take advantage of AVX2
RUN pip uninstall -y pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# Copy lambda handlers and any libs
COPY similarity/api.py ./
RUN mkdir libs
COPY libs ./libs

# Preload a tfhub model if we want
RUN mkdir -p models/use-large/1
RUN curl -L https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed -o ./models/use.tar.gz
RUN tar -xf models/use.tar.gz -C models/use-large/1
RUN rm -r models/use.tar.gz

# Download ImageNet labels
#RUN curl https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt -o ./model/ImageNetLabels.txt

# Set the CMD to your handler
CMD ["api.free_text_query"]