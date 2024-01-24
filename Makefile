# Compiler and Flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++20 -Wall -fopenmp -I/usr/include/python3.11 \
            -I/usr/local/include -I./imgui -I./imgui-sfml \
            -Iinclude -I/home/haakolau/.local/lib/python3.11/site-packages/pybind11/include \
            -I/usr/include/eigen3
NVCCFLAGS = -I/usr/local/cuda/include -I/usr/include/python3.11 \
            -I/usr/local/include -I./imgui -I./imgui-sfml \
            -Iinclude -I/home/haakolau/.local/lib/python3.11/site-packages/pybind11/include \
            -I/usr/include/eigen3

CUDA_LIB_PATH = /usr/local/cuda/lib64
LFLAGS = -L$(CUDA_LIB_PATH) -lcudart -fopenmp

# Directories
SRC_DIR = src
OBJ_DIR = obj
INCLUDE_DIR = include
CUDA_SRC_DIR = cuda_src

# Libraries
SFML = -lsfml-graphics -lsfml-window -lsfml-system
OPENGL_LIBS = -lGL
PYTHON_LIB = -L/usr/lib/python3.11/config-3.11-x86_64-linux-gnu -lpython3.11
LIBS = -L/usr/local/lib $(SFML) $(OPENGL_LIBS) $(PYTHON_LIB)

# All cpp and corresponding obj files
IMGUI_SRC = $(wildcard ./imgui/*.cpp) $(wildcard ./imgui-sfml/*.cpp)
SRC = $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRC = $(wildcard $(CUDA_SRC_DIR)/*.cu)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o) $(CUDA_SRC:$(CUDA_SRC_DIR)/%.cu=$(OBJ_DIR)/%.o) $(IMGUI_SRC:%.cpp=%.o)

# Executable Name
EXEC = Kuramoto

# Rules
all: $(OBJ)
	$(CXX) $(OBJ) $(LIBS) $(LFLAGS) -o $(EXEC)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-rm -f $(OBJ_DIR)/*.o $(EXEC) *.o