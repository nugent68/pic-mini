# Makefile for 3D Particle-in-Cell Simulation

# Compiler
CXX = g++

# Compiler flags
# -O3: Maximum optimization
# -march=native: Optimize for the current CPU architecture
# -std=c++11: Use C++11 standard
# -Wall: Enable all warnings
# -fopenmp: Enable OpenMP for parallelization (optional)
CXXFLAGS_RELEASE = -O3 -march=native -std=c++11 -Wall
CXXFLAGS_DEBUG = -g -O0 -std=c++11 -Wall -DDEBUG

# Executable names
TARGET_RELEASE = pic3d_stable
TARGET_DEBUG = pic3d_stable_debug

# Source files
SOURCES = pic3d_stable.cpp

# Default target
all: release

# Release build (optimized)
release: $(SOURCES)
	$(CXX) $(CXXFLAGS_RELEASE) $(SOURCES) -o $(TARGET_RELEASE)
	@echo "Release build complete: $(TARGET_RELEASE)"

# Debug build
debug: $(SOURCES)
	$(CXX) $(CXXFLAGS_DEBUG) $(SOURCES) -o $(TARGET_DEBUG)
	@echo "Debug build complete: $(TARGET_DEBUG)"

# Parallel build with OpenMP
parallel: $(SOURCES)
	$(CXX) $(CXXFLAGS_RELEASE) -fopenmp $(SOURCES) -o $(TARGET_RELEASE)_omp
	@echo "Parallel build complete: $(TARGET_RELEASE)_omp"

# Run the simulation
run: release
	./$(TARGET_RELEASE)

# Run debug version
run-debug: debug
	./$(TARGET_DEBUG)

# Clean build artifacts
clean:
	rm -f $(TARGET_RELEASE) $(TARGET_DEBUG) $(TARGET_RELEASE)_omp
	rm -f *.o
	rm -f pic3d_stable_diagnostics.txt
	rm -f particles_*.txt
	@echo "Clean complete"

# Help target
help:
	@echo "Available targets:"
	@echo "  make          - Build optimized release version"
	@echo "  make release  - Build optimized release version"
	@echo "  make debug    - Build debug version with debugging symbols"
	@echo "  make parallel - Build with OpenMP parallelization"
	@echo "  make run      - Build and run release version"
	@echo "  make run-debug- Build and run debug version"
	@echo "  make clean    - Remove all build artifacts"
	@echo "  make help     - Show this help message"

.PHONY: all release debug parallel run run-debug clean help
