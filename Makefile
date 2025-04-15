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

# Preprocessor macros: add KERNELNET_EXPORTS to export public symbols from the DLL.
DEFINES = -DWIN32 -DWIN64 -D_DEBUG -D_CONSOLE -D_MBCS -DKERNELNET_EXPORTS

# Host compiler flags with spaces
XCOMPILER_FLAGS = "/std:c++17 /EHsc /W3 /nologo /O2 /MDd"

# Define the host compiler
CCBIN = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64/cl.exe"

# NVCC compile flags (added -cudart static)
NVCC_COMPILE_FLAGS = $(CUDA_ARCH) -G -g -std=c++17 $(INCLUDES) $(DEFINES) \
                     -Xcompiler $(XCOMPILER_FLAGS) \
                     -cudart static --use-local-env --machine 64 --compile -x cu \
                     -ccbin $(CCBIN)

# Linker flags: add shared flag to build a DLL
NVCC_LINK_FLAGS = $(CUDA_ARCH) -G -g -shared $(INCLUDES) $(DEFINES) \
                  -Xcompiler $(XCOMPILER_FLAGS) \
                  -cudart static --use-local-env --machine 64 \
                  -ccbin $(CCBIN)

LINKER_FLAGS = -Xlinker /LIBPATH:\"C:/Program\ Files/Microsoft\ Visual\ Studio/2022/Community/VC/Tools/MSVC/14.41.34120/lib/x64\" \
               -Xlinker /LIBPATH:\"C:/Program\ Files\ \(x86\)/Windows\ Kits/10/Lib/10.0.22621.0/ucrt/x64\" \
               -Xlinker /LIBPATH:\"C:/Program\ Files\ \(x86\)/Windows\ Kits/10/Lib/10.0.22621.0/um/x64\"

# Directories
SRCDIR = src
OBJDIR = obj

# Target DLL (the import library will typically be kernelnet.lib)
TARGET = kernelnet.dll

# Source files (adjust filenames as necessary)
SOURCES = $(SRCDIR)/tensor.cu $(SRCDIR)/dense.cu $(SRCDIR)/conv2d.cu $(SRCDIR)/optimizer.cu $(SRCDIR)/autograd.cu\
          $(SRCDIR)/maxpool.cu $(SRCDIR)/softmax.cu $(SRCDIR)/sigmoid.cu $(SRCDIR)/tanh.c $(SRCDIR)/lstm.cu\
          $(SRCDIR)/lstm_wrapper.cpp $(SRCDIR)/sequential.cpp $(SRCDIR)/relu.cu $(SRCDIR)/embedding.cu $(SRCDIR)/trainer.cpp

# Assuming your main file is only used for testing your DLL, you might build it separately.
# For the library DLL, we exclude main.cpp.
LIB_SOURCES = $(filter-out $(SRCDIR)/main.cpp, $(SOURCES))

# Object files
CU_SOURCES = $(wildcard $(SRCDIR)/*.cu)
CPP_SOURCES = $(wildcard $(SRCDIR)/*.cpp)

# For DLL, exclude main.cpp from object list:
LIB_CU_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.obj,$(filter-out $(SRCDIR)/main.cu,$(CU_SOURCES)))
LIB_CPP_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.obj,$(filter-out $(SRCDIR)/main.cpp,$(CPP_SOURCES)))
OBJS = $(LIB_CU_OBJS) $(LIB_CPP_OBJS)

# Build the target DLL
$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	$(NVCC) $(NVCC_LINK_FLAGS) -o $@ $^ $(LINKER_FLAGS)

### Compile Rules ###

# Rule for compiling C++ source files
$(OBJDIR)/%.obj: $(SRCDIR)/%.cpp
	@mkdir -p "$(OBJDIR)"
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c $< -o $@

# Rule for compiling CUDA source files
$(OBJDIR)/%.obj: $(SRCDIR)/%.cu
	@mkdir -p "$(OBJDIR)"
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJDIR)/*.obj
	rm -f $(TARGET)
	rm -f $(TARGET:.dll=.lib) $(TARGET:.dll=.exp) $(TARGET:.dll=.pdb)
	rm -f vc*.pdb

# Phony targets
.PHONY: all clean