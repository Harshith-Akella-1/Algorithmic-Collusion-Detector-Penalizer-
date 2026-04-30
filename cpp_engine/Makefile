# Makefile for the C++ LOB Trading Engine
#
# Usage:
#   make          — build the engine binary
#   make clean    — remove build artifacts
#   make smoke    — build and run smoke test
#   make batch    — build and generate 100 episodes per class
#
# Requires: g++ with C++14 support (GCC >= 5.x)

CXX      = g++
CXXFLAGS = -std=c++14 -O2 -Wall -Wextra -Wno-unused-parameter
LDFLAGS  =

SRCS = lob.cpp market_tape.cpp participants.cpp simulation.cpp main.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = engine

.PHONY: all clean smoke batch

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Dependencies (manual; no header dependency tracking for simplicity)
lob.o: lob.cpp lob.hpp
market_tape.o: market_tape.cpp market_tape.hpp lob.hpp
participants.o: participants.cpp participants.hpp lob.hpp
simulation.o: simulation.cpp simulation.hpp lob.hpp market_tape.hpp participants.hpp
main.o: main.cpp simulation.hpp lob.hpp market_tape.hpp

clean:
	del /Q *.o $(TARGET).exe sample_orders.csv sample_trades.csv 2>NUL || true

smoke: $(TARGET)
	./$(TARGET) --mode smoke

batch: $(TARGET)
	./$(TARGET) --mode batch --episodes 100 --output cpp_dataset
