# Contributing to ElbowVision

Thank you for your interest in contributing to ElbowVision! This document provides guidelines for contributing to this project.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Node.js 18+

### Development Setup

1. Clone the repository:

```bash
git clone https://github.com/tomosoko/ElbowVision.git
cd ElbowVision
```

2. Start the development environment with Docker:

```bash
docker-compose up --build
```

This launches:
- **API** at http://localhost:8000
- **Frontend** at http://localhost:3000

### Running without Docker

```bash
# API
cd elbow-api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd elbow-frontend
npm install && npm run dev
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# API tests
cd elbow-api && python -m pytest tests/ -v
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`:

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure all tests pass.
3. Update documentation if your changes affect the public API or usage.
4. Submit a pull request to `main` with a clear description of the changes.

### PR Guidelines

- Keep PRs focused on a single change.
- Write descriptive commit messages.
- Add tests for new functionality.
- Ensure CI passes before requesting review.

## Reporting Issues

- Use GitHub Issues to report bugs or request features.
- Include steps to reproduce, expected behavior, and actual behavior for bugs.
- For DICOM/medical imaging issues, do **not** include patient data.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
