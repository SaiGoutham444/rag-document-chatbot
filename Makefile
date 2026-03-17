# ─────────────────────────────────────────────────────────────────
# RAG Document Chatbot — Developer Shortcuts
#
# Usage: make <command>
# Example: make run    ← starts the app
#          make test   ← runs all tests
#
# IMPORTANT: Makefile uses TABS (not spaces) for indentation.
# If you see "missing separator" errors, check your editor isn't
# converting tabs to spaces.
# ─────────────────────────────────────────────────────────────────

# The .PHONY declaration tells make these aren't real files.
# Without it, if a file named "test" existed, "make test" would
# think it's already up-to-date and do nothing.
.PHONY: install run test eval lint format clean help

# Default target — runs when you type just "make" with no command
help:
	@echo "╔══════════════════════════════════════════╗"
	@echo "║   RAG Document Chatbot — Make Commands   ║"
	@echo "╠══════════════════════════════════════════╣"
	@echo "║  make install  — Install all dependencies ║"
	@echo "║  make run      — Start the Streamlit app  ║"
	@echo "║  make test     — Run tests with coverage  ║"
	@echo "║  make eval     — Run RAG evaluation       ║"
	@echo "║  make lint     — Check code style         ║"
	@echo "║  make format   — Auto-fix code style      ║"
	@echo "║  make clean    — Remove generated files   ║"
	@echo "╚══════════════════════════════════════════╝"

# install: Sets up the project from scratch.
# Run this once after cloning the repo or creating the venv.
install:
	# Install main application dependencies
	pip install -r requirements.txt
	# Install development tools (pytest, black, ruff)
	pip install -r requirements-dev.txt
	@echo "✅ Installation complete. Copy .env.example to .env and add your keys."

# run: Starts the Streamlit web application.
# Opens browser automatically at http://localhost:8501
run:
	streamlit run app.py

# test: Runs the full test suite with coverage reporting.
# -v flag: verbose output (shows each test name as it runs)
# --cov=src: measures coverage of code in src/ folder
# --cov-report=term-missing: shows which specific lines are untested
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# eval: Runs the RAG evaluation pipeline against ground truth dataset.
# This is what the CI quality gate runs on every PR.
eval:
	python tests/eval/run_evals.py

# lint: Checks code quality WITHOUT making changes.
# Use this to see what would change before committing.
# ruff: catches bugs, unused imports, undefined variables
# black --check: reports formatting issues without fixing them
lint:
	ruff check src/ tests/
	black --check src/ tests/

# format: Auto-fixes all code style issues.
# Run this before committing to ensure clean code.
# black: reformats all files to consistent style
# ruff --fix: auto-fixes all fixable linting issues
format:
	black src/ tests/
	ruff check --fix src/ tests/

# clean: Removes all auto-generated files.
# Use this to start fresh or before a production deployment.
# -rf flag: recursive + force (no error if folder doesn't exist)
clean:
	rm -rf chroma_db/
	rm -rf bm25_index/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ All generated files removed."