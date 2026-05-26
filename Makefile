CXX = g++
CXXFLAGS = -O3 -fopenmp -I.
TARGET = render

all: $(TARGET)

$(TARGET): main.cpp lbfgs.c
	$(CXX) $(CXXFLAGS) main.cpp lbfgs.c -o $(TARGET)

clean:
	rm -f $(TARGET) *.o test_*.png test_*.svg toto*.png toto.svg