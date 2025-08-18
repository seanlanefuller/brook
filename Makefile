# Simple Makefile for brook project
CC = gcc
#CFLAGS = -O2 -Wall
CFLAGS = -O3 -march=native -ffast-math -funroll-loops -fomit-frame-pointer
OBJDIR = bin

OBJS = $(OBJDIR)/brook.o $(OBJDIR)/model.o $(OBJDIR)/interface.o \
	$(OBJDIR)/data.o $(OBJDIR)/token.o $(OBJDIR)/train.o $(OBJDIR)/predict.o \
	 $(OBJDIR)/util.o

all: brook

gen: generate.py
	python generate.py

brook: $(OBJS)
	$(CC) $(CFLAGS) -o brook $(OBJS) -lm

$(OBJDIR)/util.o: util.c brook.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c util.c -o $(OBJDIR)/util.o

$(OBJDIR)/brook.o: brook.c brook.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c brook.c -o $(OBJDIR)/brook.o

$(OBJDIR)/model.o: model.c brook.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c model.c -o $(OBJDIR)/model.o

$(OBJDIR)/interface.o: interface.c brook.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c interface.c -o $(OBJDIR)/interface.o

$(OBJDIR)/data.o: data.c brook.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c data.c -o $(OBJDIR)/data.o

$(OBJDIR)/token.o: token.c brook.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c token.c -o $(OBJDIR)/token.o

$(OBJDIR)/train.o: train.c brook.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c train.c -o $(OBJDIR)/train.o

$(OBJDIR)/predict.o: predict.c brook.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c predict.c -o $(OBJDIR)/predict.o

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	-rm -f brook $(OBJDIR)/*.o

#	-rm weights.bin

.PHONY: all
