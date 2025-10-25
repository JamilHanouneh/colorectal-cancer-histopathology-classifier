# Contributing to Colorectal Cancer Histopathology Classifier

Thank you for your interest in contributing! This project is part of academic research on medical image analysis.

## How to Contribute

### Reporting Issues
- Use the issue tracker to report bugs or propose features
- Provide detailed descriptions and reproducible examples
- Include system information (OS, Python version, PyTorch version)

### Code Contributions
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes following the code style
4. Add tests if applicable
5. Run tests: `pytest tests/ -v`
6. Commit your changes (`git commit -m 'Add feature'`)
7. Push to your branch (`git push origin feature/your-feature`)
8. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use `black` for formatting: `black src/ --line-length 100`
- Use `flake8` for linting: `flake8 src/ --max-line-length=100`
- Add docstrings to all functions and classes

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Maintain or improve code coverage

## Development Setup

