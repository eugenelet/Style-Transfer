RCS = $(wildcard *.cpp)
PROGS = $(patsubst %.cpp,%,$(SRCS))
OBJS = $(SRCS:.cpp=.o)
TEMPS = $(SRCS:.cpp=.txt)
SRC = main_taskbar.cpp
CFLAGS = `pkg-config --cflags --libs opencv` -O3 -g
LDFLAGS = `pkg-config --libs opencv`
OUT = Style_Transfer

all: $(OUT)

$(OUT): $(SRC)
	g++ $(SRC) $(CFLAGS)  -o $@

clean:
	@rm -f $(PROGS) $(OBJS) $(TEMPS) $(OUT)
	@echo "Limpo!"
