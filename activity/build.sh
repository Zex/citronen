#!/bin/bash

export GOPATH=`pwd`

go build -work route.go
go build -work client.go

