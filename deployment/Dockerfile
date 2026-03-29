# Use a highly compatible python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Step 1: Install dependencies first (for caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Step 2: Create a non-root user (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Step 3: Set up the app directory
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Step 4: Run the application
# We use 7860 as it is the default port for Hugging Face Spaces
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
