CFLAGS = -O3 -Wall -Wall -Wextra -pedantic -fPIC
TARGET = liboptimize.so

all:
	$(CC) $(CFLAGS) native/optimize.c -shared -o $(TARGET)

clean:
	$(RM) native/*.o $(TARGET)
