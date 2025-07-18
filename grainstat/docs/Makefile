# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html livehtml linkcheck spelling

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*
	@echo "Build directory cleaned."

# Build HTML documentation
html:
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Build HTML with automatic reload (requires sphinx-autobuild)
livehtml:
	@echo "Starting live documentation server..."
	@command -v sphinx-autobuild >/dev/null 2>&1 || { \
		echo "sphinx-autobuild is required. Install it with:"; \
		echo "  pip install sphinx-autobuild"; \
		exit 1; \
	}
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--host 0.0.0.0 \
		--port 8000 \
		--open-browser \
		--delay 1 \
		--ignore "*.swp" \
		--ignore "*.pdf" \
		--ignore "*.log" \
		--ignore "*.out" \
		--ignore "*.toc" \
		--ignore "*.aux" \
		--ignore "*.idx" \
		--ignore "*.ind" \
		--ignore "*.ilg" \
		--ignore "*.tex"

# Check for broken links
linkcheck:
	@echo "Checking for broken links..."
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)
	@echo "Link check finished. Check $(BUILDDIR)/linkcheck/output.txt for results."

# Check spelling (requires sphinxcontrib-spelling)
spelling:
	@echo "Checking spelling..."
	@command -v sphinx-build >/dev/null 2>&1 || { \
		echo "sphinxcontrib-spelling is required. Install it with:"; \
		echo "  pip install sphinxcontrib-spelling"; \
		exit 1; \
	}
	$(SPHINXBUILD) -b spelling "$(SOURCEDIR)" "$(BUILDDIR)/spelling" $(SPHINXOPTS)
	@echo "Spelling check finished. Check $(BUILDDIR)/spelling/ for results."

# Build LaTeX (for PDF generation)
latex:
	@echo "Building LaTeX documentation..."
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS)
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."

# Build PDF (requires LaTeX)
pdf: latex
	@echo "Building PDF documentation..."
	@command -v pdflatex >/dev/null 2>&1 || { \
		echo "pdflatex is required for PDF generation."; \
		echo "Install LaTeX distribution (e.g., texlive-latex-base on Ubuntu)."; \
		exit 1; \
	}
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "PDF build finished. The PDF is in $(BUILDDIR)/latex/GrainStat.pdf."

# Build EPUB
epub:
	@echo "Building EPUB documentation..."
	$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS)
	@echo "Build finished. The EPUB file is in $(BUILDDIR)/epub/."

# Build for Read the Docs (includes multiple formats)
rtd:
	@echo "Building documentation for Read the Docs..."
	$(MAKE) clean
	$(MAKE) html
	$(MAKE) epub
	@echo "Read the Docs build finished."

# Install documentation dependencies
install-deps:
	@echo "Installing documentation dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed."

# Comprehensive build (clean + html + linkcheck)
all: clean html linkcheck
	@echo "Comprehensive build completed."

# Development build (faster, skip some checks)
dev:
	@echo "Building documentation for development..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) -E -a
	@echo "Development build finished."

# Check documentation quality
check: linkcheck spelling
	@echo "Documentation quality checks completed."

# Generate API documentation from source code
apidoc:
	@echo "Generating API documentation from source code..."
	@command -v sphinx-apidoc >/dev/null 2>&1 || { \
		echo "sphinx-apidoc is required. It's included with Sphinx."; \
		exit 1; \
	}
	sphinx-apidoc -o . ../grainstat --force --separate
	@echo "API documentation generated. Run 'make html' to build."

# Serve documentation locally
serve:
	@echo "Serving documentation on http://localhost:8080"
	@command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || { \
		echo "Python is required to serve documentation."; \
		exit 1; \
	}
	@if [ -d "$(BUILDDIR)/html" ]; then \
		cd $(BUILDDIR)/html && python3 -m http.server 8080 2>/dev/null || python -m SimpleHTTPServer 8080; \
	else \
		echo "Documentation not built. Run 'make html' first."; \
		exit 1; \
	fi

# Show build statistics
stats:
	@echo "Documentation Statistics"
	@echo "========================"
	@echo "Source files:"
	@find . -name "*.rst" -o -name "*.md" | wc -l
	@echo "Total lines in source files:"
	@find . -name "*.rst" -o -name "*.md" -exec cat {} \; | wc -l
	@if [ -d "$(BUILDDIR)/html" ]; then \
		echo "Built HTML files:"; \
		find $(BUILDDIR)/html -name "*.html" | wc -l; \
		echo "Total HTML size:"; \
		du -sh $(BUILDDIR)/html; \
	else \
		echo "No built documentation found."; \
	fi

# Validate documentation structure
validate:
	@echo "Validating documentation structure..."
	@echo "Checking for missing files..."
	@for file in index.rst installation.rst quickstart.rst api.rst; do \
		if [ ! -f "$$file" ]; then \
			echo "Missing required file: $$file"; \
			exit 1; \
		fi; \
	done
	@echo "Checking for broken internal references..."
	@grep -r ":doc:" . --include="*.rst" | while read line; do \
		ref=$$(echo "$$line" | sed -n 's/.*:doc:`\([^`]*\)`.*/\1/p'); \
		if [ -n "$$ref" ] && [ ! -f "$$ref.rst" ]; then \
			echo "Broken reference: $$ref in $$line"; \
		fi; \
	done
	@echo "Documentation structure validation completed."

# Watch for changes and rebuild (requires inotify-tools on Linux)
watch:
	@echo "Watching for changes..."
	@command -v inotifywait >/dev/null 2>&1 || { \
		echo "inotifywait is required. Install inotify-tools."; \
		echo "On Ubuntu: sudo apt-get install inotify-tools"; \
		echo "On macOS: brew install fswatch (then use 'make watch-mac')"; \
		exit 1; \
	}
	@while true; do \
		inotifywait -r -e modify,create,delete . --include=".*\.(rst|py|md)$$" && \
		echo "Change detected, rebuilding..." && \
		$(MAKE) html; \
	done

# Watch for changes on macOS (requires fswatch)
watch-mac:
	@echo "Watching for changes on macOS..."
	@command -v fswatch >/dev/null 2>&1 || { \
		echo "fswatch is required. Install it with:"; \
		echo "  brew install fswatch"; \
		exit 1; \
	}
	@fswatch -o . | while read f; do \
		echo "Change detected, rebuilding..." && \
		$(MAKE) html; \
	done

# Create a new tutorial template
new-tutorial:
	@read -p "Enter tutorial name (e.g., 'custom_analysis'): " name; \
	if [ -z "$$name" ]; then \
		echo "Tutorial name cannot be empty."; \
		exit 1; \
	fi; \
	tutorial_file="tutorials/$$name.rst"; \
	if [ -f "$$tutorial_file" ]; then \
		echo "Tutorial $$tutorial_file already exists."; \
		exit 1; \
	fi; \
	echo "Creating tutorial: $$tutorial_file"; \
	mkdir -p tutorials; \
	cat > "$$tutorial_file" << 'EOF'
Tutorial Title
==============

**Estimated time:** XX minutes

**Learning objectives:**
- Objective 1
- Objective 2
- Objective 3

Prerequisites
-------------

- Prerequisite 1
- Prerequisite 2

Introduction
------------

Brief introduction to the tutorial content.

Step 1: First Step
------------------

Description of the first step.

.. code-block:: python

   # Example code
   from grainstat import GrainAnalyzer

   analyzer = GrainAnalyzer()

Step 2: Second Step
-------------------

Description of the second step.

Conclusion
----------

Summary of what was learned.

Next Steps
----------

- Link to related tutorials
- Suggestions for further learning

EOF
	echo "Tutorial template created: $$tutorial_file"

# Help with common tasks
info:
	@echo "GrainStat Documentation Build System"
	@echo "===================================="
	@echo ""
	@echo "Common commands:"
	@echo "  make html          - Build HTML documentation"
	@echo "  make livehtml      - Build and serve with auto-reload"
	@echo "  make clean         - Clean build directory"
	@echo "  make linkcheck     - Check for broken links"
	@echo "  make serve         - Serve built documentation"
	@echo "  make all           - Clean, build, and check"
	@echo ""
	@echo "Development commands:"
	@echo "  make dev           - Fast development build"
	@echo "  make watch         - Watch for changes and rebuild"
	@echo "  make validate      - Validate documentation structure"
	@echo "  make stats         - Show build statistics"
	@echo ""
	@echo "Advanced commands:"
	@echo "  make pdf           - Build PDF documentation"
	@echo "  make epub          - Build EPUB documentation"
	@echo "  make apidoc        - Generate API docs from source"
	@echo "  make spelling      - Check spelling"
	@echo ""
	@echo "Requirements:"
	@echo "  make install-deps  - Install documentation dependencies"
	@echo ""
	@echo "For more options, run: make help"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)