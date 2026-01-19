---
title: SciCoQA Discrepancy Detection
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# ScicoQA Discrepancy Detection Demo

Detect discrepancies between scientific papers and their code implementations using GPT OSS 20B.

## Features

- **Fast Paper Processing**: Uses arxiv2md for quick HTML-based paper conversion (no OCR needed)
- **Full Repository Analysis**: Clones and processes entire GitHub repositories
- **High-Reasoning Detection**: Uses GPT OSS 20B with high reasoning effort for accurate discrepancy detection
- **Clean Output**: Automatically removes references and formats results nicely

## Usage

1. Enter an arXiv URL or paper ID (e.g., `https://arxiv.org/abs/2006.12834` or `2006.12834`)
2. Enter the corresponding GitHub repository URL
3. Click "Detect Discrepancies"
4. View the detected discrepancies in a formatted list

## Environment Variables

Set the following in your HuggingFace Space settings:

- `OPENROUTER_API_KEY`: Your OpenRouter API key for GPT OSS 20B access

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENROUTER_API_KEY=your_key_here

# Run the app
streamlit run app.py
```

## Architecture

The demo replicates the inference pipeline from the main ScicoQA codebase:

1. **Paper Processing**: arxiv2md API converts arXiv papers to markdown
2. **Code Loading**: Clones GitHub repos and processes code files
3. **Token Management**: Calculates token usage and truncates code if needed
4. **Prompt Construction**: Builds prompt using the `discrepancy_generation` template
5. **LLM Inference**: Calls GPT OSS 20B via OpenRouter with high reasoning effort
6. **Parsing**: Extracts discrepancies from YAML output

## Deployment

This app is configured for HuggingFace Spaces deployment using Docker. The Dockerfile includes all necessary dependencies and system packages.
