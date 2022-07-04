# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8

# Copy the requirements.txt file to the container
COPY requirements-st.txt ./

# Install the python requirements from requirements.txt
RUN python3.8 -m pip install -r requirements-st.txt

# Copy lambda handlers and any libs
COPY similarity/api.py ./
RUN mkdir libs
COPY libs ./libs

# Set the CMD to your handler
CMD ["api.free_text_query"]