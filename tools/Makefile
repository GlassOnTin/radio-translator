# SPDX-License-Identifier: AGPL-3.0-or-later
# Build the SDRplay IQ capture tool against API v3 (headers/libs in /usr/local).
CC      ?= gcc
CFLAGS  ?= -O2 -Wall
INCS     = -I/usr/local/include
LIBS     = -L/usr/local/lib
LDLIBS   = -lsdrplay_api -lpthread -lrt

all: sdrplay_cap

sdrplay_cap: sdrplay_cap.c
	$(CC) $(CFLAGS) $(INCS) -o $@ $< $(LIBS) $(LDLIBS)

clean:
	rm -f sdrplay_cap

.PHONY: all clean
