# Find news makefile
GO			:= go
GOPATH	:= $(shell pwd)
BUILD		:= "build"
DEPENDS := github.com/balzaczyy/golucene

.PHONY: index build

export GOPATH $(GOPATH)

all: env_info deps find_news

env_info:
	$(GO) env

deps:
	$(GO) get $(DEPENDS)

%: %.go
	$(GO) build -o $(BUILD)/$@

index:
	$(BUILD)/find_news -index_base index/bbc -src_base data/bbc/politics

search:
	$(BUILD)/find_news -index_base index/bbc -content Apple

clearn:
	rm -rf build index
