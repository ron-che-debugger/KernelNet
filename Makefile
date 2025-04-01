# Compiler
NVCC = "D:/External Installations/bin/nvcc.exe"

# Compute capability
CUDA_ARCH = -gencode=arch=compute_52,code=\"sm_52,compute_52\"

# Include directories
INCLUDES = -I"./include" \
           -I"D:/External Installations/include" \
           -I"D:/External_Apps/GLEW/include" \
           -I"D:/External_Apps/GLFW/include" \
           -I"D:/External_Apps/JSON/json/include" \
           -I"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/include" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/ucrt" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/shared" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/um" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/winrt" \
           -I"C:/Program Files (x86)/Windows Kits/10/Include/10.0.22621.0/cppwinrt"

# Preprocessor macros
DEFINES = -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS

# Host compiler flags with spaces (switched to /MDd for debug runtime)
XCOMPILER_FLAGS = "/std:c++17 /EHsc /W3 /nologo /O2 /MDd"

# Define the host compiler
CCBIN = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64/cl.exe"

# NVCC compile flags (added -cudart static)
NVCC_COMPILE_FLAGS = $(CUDA_ARCH) -G -g -std=c++17 $(INCLUDES) $(DEFINES) \
                     -Xcompiler $(XCOMPILER_FLAGS) \
                     -cudart static --use-local-env --machine 64 --compile -x cu \
                     -ccbin $(CCBIN)

# Linker flags (unchanged)
LINKER_FLAGS = -Xlinker /LIBPATH:\"C:/Program\ Files/Microsoft\ Visual\ Studio/2022/Community/VC/Tools/MSVC/14.41.34120/lib/x64\" \
               -Xlinker /LIBPATH:\"C:/Program\ Files\ \(x86\)/Windows\ Kits/10/Lib/10.0.22621.0/ucrt/x64\" \
               -Xlinker /LIBPATH:\"C:/Program\ Files\ \(x86\)/Windows\ Kits/10/Lib/10.0.22621.0/um/x64\"

# NVCC link flags (added -cudart static)
NVCC_LINK_FLAGS = $(CUDA_ARCH) -G -g $(INCLUDES) $(DEFINES) \
                  -Xcompiler $(XCOMPILER_FLAGS) \
                  -cudart static --use-local-env --machine 64 \
                  -ccbin $(CCBIN)

# Directories
SRCDIR = src
OBJDIR = obj

# Target executable
TARGET = tensor_example.exe

# Source files (adjust filenames as necessary)
SOURCES = $(SRCDIR)/tensor.cpp $(SRCDIR)/dense.cpp $(SRCDIR)/optimizer.cpp $(SRCDIR)/main.cpp

# Object files
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.obj,$(SOURCES))

# Build the target executable
$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCC_LINK_FLAGS) -o $@ $^ $(LINKER_FLAGS)

### Compile Rules ###

# Rule for compiling C++ source files
$(OBJDIR)/%.obj: $(SRCDIR)/%.cpp
	@mkdir -p "$(OBJDIR)"
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c $< -o $@

# Rule for compiling CUDA source files (if any)
$(OBJDIR)/%.obj: $(SRCDIR)/%.cu
	@mkdir -p "$(OBJDIR)"
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c $< -o $@

# Run target
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(OBJDIR)/*.obj
	rm -f $(TARGET)
	rm -f $(TARGET:.exe=.exp) $(TARGET:.exe=.lib) $(TARGET:.exe=.pdb)
	rm -f vc*.pdb

# Phony targets
.PHONY: all clean run