# Find news makefile
GOPATH	:= $(shell pwd)
GO			:= go
BUILD		:= "build"
DEPENDS := github.com/balzaczyy/golucene

.PHONY: index build

all: env_info deps datasets find_news

env_info:
	$(GO) env

datasets:
	$(shell data/get.sh)
	@echo "datasets done"

deps:
	$(GO) get $(DEPENDS)

%: %.go
	$(GO) build -o $(BUILD)/$@

index:
	$(BUILD)/find_news -index_base index/bbc -src_base data/bbc/politics

search:
	$(BUILD)/find_news -index_base index/bbc -content Apple

clean:
	rm -rf build index
