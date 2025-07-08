# EasyReadTogether Backend

This repository contains the backend services and demonstration interface for the EasyReadTogether project. The goal of this project is to convert complex documents into the EasyRead format, making information more accessible.

The system utilizes Large Language Models to simplify text, provides Text-to-Speech (TTS) capabilities, and automatically associates relevant images with the simplified content.

## Table of Contents

- [EasyReadTogether Backend](#easyreadtogether-backend)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Running the FastAPI Server](#running-the-fastapi-server)
    - [Running the Streamlit Demo](#running-the-streamlit-demo)
    - [Running Evaluations](#running-evaluations)
  - [Project Structure](#project-structure)
  - [Deployment](#deployment)

## Features

-   **Text Simplification:** Converts complex text into EasyRead format using AWS Bedrock (Meta Llama 3 models).
-   **Language Support:** Handles content in English and Swahili.
-   **FastAPI Backend:** Provides API endpoints for simplification, TTS, and analytics.
-   **Text-to-Speech (TTS):** Generates audio from simplified text using the Kokoro TTS engine (`src/tts.py`).
-   **Image Association:** Automatically finds and associates relevant images with simplified text blocks using sentence embeddings.
-   **Readability Evaluation:** Calculates readability scores (Flesch-Kincaid, SMOG, Dale-Chall) for the output (`evaluate.py`).
-   **Streamlit Demo:** A user interface for testing the conversion process (`app.py`).

## Prerequisites

-   Python 3.8+
-   AWS Account with access to Amazon Bedrock (specifically Llama 3 models).
-   AWS Credentials configured for Boto3.
-   (Optional) CUDA-enabled GPU for faster local processing (TTS and embeddings).

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/your-repo/easyreadtogether-backend.git
    cd easyreadtogether-backend
    ```

2.  Create and activate a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application requires environment variables for AWS access and other settings.

1.  Create a `.env` file in the root directory by copying the example file:

    ```bash
    cp .env.example .env
    ```

2.  Edit the `.env` file and provide your credentials:

    ```
    AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY
    AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_KEY
    AWS_IMAGE_BUCKET=YOUR_S3_BUCKET_FOR_IMAGES # Used by image processing scripts
    MOCK_PASSWORD=A_PASSWORD_FOR_MOCK_AUTH # Used for mock authentication in the API
    ```

Note: The AWS credentials must have permissions for Amazon Bedrock (for text generation) and S3 (if using the image upload scripts).

## Usage

### Running the FastAPI Server

The FastAPI server (`server.py`) provides the core API endpoints for simplification, TTS, and image association.

To start the server, use Uvicorn:

```bash
uvicorn server:app --host 0.0.0.0 --port 8050 --reload
```

The API will be available at `http://localhost:8050`.

Key Endpoints:
-   `POST /api/simplify`: Accepts text or a file and returns simplified content blocks with associated images.
-   `POST /api/listen`: Accepts text and returns a WAV audio stream (TTS).
-   `GET/POST /api/analytics`: Handles basic usage tracking.

### Running the Streamlit Demo

The Streamlit application (`app.py`) provides a graphical interface to test the text simplification models and view readability metrics.

```bash
streamlit run app.py
```

The application will open in your web browser, typically at `http://localhost:8501`.

### Running Evaluations

The `evaluate.py` script can be used from the command line to generate EasyRead text from an input file and calculate its readability metrics.

Example usage:

```bash
python evaluate.py \
    -f "./data/input_document.txt" \
    -m "meta.llama3-3-70b-instruct-v1:0" \
    -p "./prompts/prompt_2.txt" \
    -o "./output/simplified_output.md"
```

Additionally, `run_eval.sh` is provided to run batch evaluations across multiple input files, models, and prompts.

## Project Structure

-   `app.py`: The Streamlit demonstration interface.
-   `server.py`: The FastAPI backend application.
-   `model_aws.py`: Interface for interacting with AWS Bedrock models (Llama 3).
-   `model_swa.py`: Handles Swahili translations using Hugging Face models.
-   `evaluate.py`: Contains logic for calculating readability metrics using `textstat`.
-   `src/tts.py`: Text-to-Speech implementation using Kokoro.
-   `src/image_process/`: Scripts used for collecting EasyRead images, generating descriptions, creating embeddings, and uploading to S3.
-   `run_eval.sh`: Shell script for batch evaluation of models and prompts.

## Deployment

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) configured to automatically deploy the FastAPI application to an AWS EC2 instance when changes are pushed to the `main` branch. This requires setting up SSH secrets (`EC2_SSH_KEY`, `EC2_HOST`, `EC2_USER`) in the GitHub repository settings.