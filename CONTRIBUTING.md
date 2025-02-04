# Contributing to EmbodyHub

Thank you for your interest in contributing to EmbodyHub! This document provides guidelines and standards for contributing to the project.

## Code Style Guidelines

### Language Requirements

- All code comments, documentation, commit messages, and API documentation MUST be written in English
- Use clear and concise English for better international collaboration
- Avoid colloquialisms and region-specific expressions

### Documentation Standards

1. **Code Comments**
   - Write meaningful comments that explain WHY, not WHAT
   - Use complete sentences with proper punctuation
   - Keep comments up-to-date with code changes

2. **Function Documentation**
   ```python
   def process_data(input_data: np.ndarray) -> Dict[str, Any]:
       """Process input data and return analysis results.
       
       Args:
           input_data: Input array containing sensor data
           
       Returns:
           Dictionary containing processed results
           
       Raises:
           ValueError: If input_data is empty or malformed
       """
   ```

3. **Module Documentation**
   - Include a module-level docstring explaining the purpose
   - List key classes and functions
   - Provide usage examples when appropriate

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Keep functions focused and single-purpose
- Maintain consistent indentation

## Pull Request Process

1. Ensure all comments and documentation are in English
2. Update relevant documentation
3. Add tests for new features
4. Follow the existing code style

## Communication

- Use English for all project communications
- Be clear and professional in issue discussions
- Provide context and examples when reporting bugs

## Questions?

If you have any questions about these guidelines, please open an issue for clarification.